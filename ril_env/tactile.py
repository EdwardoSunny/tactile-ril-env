"""
Tactile sensor support for the A31301 streaming protocol over USB serial.

Each sensor board exposes 9 three-axis Hall taxels. The on-device firmware
emits frames as line-oriented serial:

    BEGIN_STREAM ... units=raw,rate_ms=10
    S,<ts_ms>,<idx>,<addr>,<connected>,<x>,<y>,<z>
    ...
    END_STREAM

This module reads one or more of those boards in dedicated mp.Process workers
and publishes the latest (n_taxels, 3) frame to a SharedMemoryRingBuffer per
sensor. The XArmController consumes those ring buffers to clamp gripper
closing when contact exceeds a configurable threshold.

Outsiders calling `XArmController.step(pose, grasp)` see no API change — the
safety wrapper only intervenes when tactile readings cross the threshold.
"""

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import serial
except ImportError:
    serial = None

from multiprocessing.managers import SharedMemoryManager

# SharedMemoryRingBuffer pulls in the `atomics` C extension. Keep this import
# lazy so the pure-Python helpers (TactileConfig, evaluate_safety, ...) remain
# usable in environments that don't have the SHM stack installed.


logger = logging.getLogger(__name__)


# Lines containing any of these patterns indicate an ESP32 reboot/crash.
_REBOOT_PATTERNS = (
    "rst:0x",
    "boot:0x",
    "configsip:",
    "ets ",
    "Guru Meditation",
    "Backtrace:",
    "assert failed:",
    "abort()",
    "panic_handler",
    "LoadProhibited",
    "StoreProhibited",
    "IllegalInstruction",
    "IntegerDivideByZero",
)


@dataclass
class TactileConfig:
    """Configuration for one or more A31301 boards and the gripper safety wrapper."""

    # One serial port per board. Default: two boards on /dev/ttyUSB{0,1}.
    ports: List[str] = field(
        default_factory=lambda: ["/dev/ttyUSB0", "/dev/ttyUSB1"]
    )
    baud: int = 115200
    # Number of taxels per board (A31301 boards used here have 9).
    n_taxels: int = 9

    # How to reduce the (n_sensors, n_taxels, 3) tensor to a scalar contact metric:
    #   "max_abs_z"  : max(|Z|)            across connected taxels of all sensors
    #   "max_norm"   : max(||XYZ||)        across connected taxels of all sensors
    #   "sum_abs_z"  : sum(|Z|)            across connected taxels of all sensors
    safety_metric: str = "max_abs_z"
    # Trip threshold. Units depend on what the device is streaming (raw counts by default).
    safety_threshold: float = 2000.0
    # Tactile reading older than this is treated as unsafe (i.e. clamp closing).
    stale_after_sec: float = 0.2

    # Optional device-side configuration sent on stream start.
    set_device_units: Optional[str] = None  # "raw" | "mt" | "g"
    set_device_rate_hz: Optional[int] = None  # clamped to [1, 100]

    verbose: bool = False


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

def _is_reboot(line: str) -> bool:
    return any(p in line for p in _REBOOT_PATTERNS)


def _parse_sample(line: str):
    """Parse a single ``S,...`` row. Returns (ts_ms, idx, connected, x, y, z) or None."""
    if not line.startswith("S,"):
        return None
    parts = line.split(",")
    if len(parts) != 8:
        return None
    try:
        return (
            int(parts[1]),
            int(parts[2]),
            int(parts[4]),
            float(parts[5]),
            float(parts[6]),
            float(parts[7]),
        )
    except Exception:
        return None


def compute_safety_metric(
    states: Sequence[Dict],
    metric: str,
    stale_after_sec: float,
) -> Tuple[float, bool]:
    """Reduce per-sensor ring-buffer snapshots to a scalar contact metric.

    Returns (metric_value, all_fresh). ``all_fresh`` is False if any sensor has
    no connected taxels or has not produced a frame within ``stale_after_sec``.
    The caller should treat ``all_fresh=False`` as unsafe.
    """
    now = time.time()
    per_sensor: List[float] = []
    all_fresh = True
    for st in states:
        host_ts = float(st.get("host_timestamp", 0.0))
        conn = np.asarray(st.get("connected", np.zeros(0)), dtype=np.int32)
        xyz = np.asarray(st.get("xyz", np.zeros((0, 3))), dtype=np.float32)
        if (
            conn.size == 0
            or not np.any(conn)
            or host_ts <= 0.0
            or (now - host_ts) > stale_after_sec
        ):
            all_fresh = False
            continue
        mask = conn > 0
        if metric == "max_abs_z":
            per_sensor.append(float(np.max(np.abs(xyz[mask, 2]))))
        elif metric == "max_norm":
            per_sensor.append(float(np.max(np.linalg.norm(xyz[mask], axis=1))))
        elif metric == "sum_abs_z":
            per_sensor.append(float(np.sum(np.abs(xyz[mask, 2]))))
        else:
            raise ValueError(f"Unknown safety_metric: {metric!r}")
    if not per_sensor:
        return 0.0, False
    if metric == "sum_abs_z":
        return float(sum(per_sensor)), all_fresh
    return float(max(per_sensor)), all_fresh


def evaluate_safety(
    states: Sequence[Dict],
    config: TactileConfig,
) -> Tuple[float, bool]:
    """Returns (metric_value, is_safe_to_close).

    ``is_safe_to_close=False`` -> caller must not increase grasp closure. Stale
    or missing tactile data forces ``is_safe_to_close=False`` (fail-safe).
    """
    metric_val, fresh = compute_safety_metric(
        states, config.safety_metric, config.stale_after_sec
    )
    if not fresh:
        return metric_val, False
    return metric_val, metric_val <= config.safety_threshold


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

class TactileSensor(mp.Process):
    """One mp.Process per A31301 board.

    The child process owns the serial port, parses frames, and publishes the
    latest 9-taxel xyz frame to a SharedMemoryRingBuffer the parent can read.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        port: str,
        config: TactileConfig,
    ):
        super().__init__(name=f"TactileSensor-{port}")
        if serial is None:
            raise ImportError(
                "pyserial is required for TactileSensor; "
                "install with `pip install pyserial` or via environment.yml."
            )
        from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

        self.port = port
        self.config = config
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

        n = config.n_taxels
        example = {
            "xyz": np.zeros((n, 3), dtype=np.float32),
            "connected": np.zeros(n, dtype=np.int32),
            "device_ts_ms": np.int64(0),
            # host_timestamp=0.0 -> always counted as stale until first real put.
            "host_timestamp": 0.0,
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=32,
            get_time_budget=0.2,
            put_desired_frequency=100,
        )

        if config.verbose:
            logger.setLevel(logging.DEBUG)

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(5)

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_state(self, k=None):
        if k is None:
            return self.ring_buffer.get()
        return self.ring_buffer.get_last_k(k)

    # ------------------------------------------------------------------
    # Child-process entry point
    # ------------------------------------------------------------------

    def run(self):
        cfg = self.config
        try:
            ser = serial.Serial(
                self.port, cfg.baud, timeout=1, rtscts=False, dsrdtr=False
            )
            # Hold DTR/RTS to avoid auto-resetting the ESP32 on connect.
            try:
                ser.setDTR(True)
                ser.setRTS(True)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[TactileSensor {self.port}] open failed: {e}")
            self.ready_event.set()
            return

        def send_cmd(line: str):
            try:
                ser.write((line + "\n").encode())
                ser.flush()
            except Exception as e:
                logger.error(f"[TactileSensor {self.port}] cmd '{line}' failed: {e}")

        if cfg.set_device_units:
            send_cmd(f"CMD,SET,UNITS,{cfg.set_device_units}")
        if cfg.set_device_rate_hz is not None:
            hz = max(1, min(100, int(cfg.set_device_rate_hz)))
            send_cmd(f"CMD,SET,RATE_HZ,{hz}")
        send_cmd("CMD,GET,STATE")

        self.ready_event.set()

        n = cfg.n_taxels
        frame_xyz = np.zeros((n, 3), dtype=np.float32)
        frame_conn = np.zeros(n, dtype=np.int32)
        seen = np.zeros(n, dtype=bool)
        current_ts = None
        started = False

        try:
            while not self.stop_event.is_set():
                try:
                    raw = ser.readline().decode(errors="ignore")
                except Exception as e:
                    logger.error(f"[TactileSensor {self.port}] read error: {e}")
                    break

                if not raw:
                    continue
                line = raw.strip()
                if not line:
                    continue

                if _is_reboot(line):
                    logger.warning(
                        f"[TactileSensor {self.port}] reboot indicator: {line}"
                    )
                    started = False
                    current_ts = None
                    seen[:] = False
                    continue

                if line.startswith("BEGIN_STREAM"):
                    started = True
                    current_ts = None
                    frame_xyz[:] = 0
                    frame_conn[:] = 0
                    seen[:] = False
                    logger.info(f"[TactileSensor {self.port}] stream begin: {line}")
                    continue

                if line.startswith("END_STREAM"):
                    started = False
                    logger.info(f"[TactileSensor {self.port}] stream end")
                    continue

                if not started:
                    continue

                parsed = _parse_sample(line)
                if parsed is None:
                    continue
                ts_ms, idx, conn, x, y, z = parsed
                if not (0 <= idx < n):
                    continue

                if current_ts is None:
                    current_ts = ts_ms

                # Frame boundary: ts_ms changed -> publish accumulated frame and start a new one.
                if ts_ms != current_ts:
                    self._publish(frame_xyz, frame_conn, current_ts)
                    frame_xyz[:] = 0
                    frame_conn[:] = 0
                    seen[:] = False
                    current_ts = ts_ms

                frame_xyz[idx] = (x, y, z)
                frame_conn[idx] = conn
                seen[idx] = True

                # Early publish once all taxels for this frame have arrived.
                if seen.all():
                    self._publish(frame_xyz, frame_conn, current_ts)
                    frame_xyz[:] = 0
                    frame_conn[:] = 0
                    seen[:] = False
                    current_ts = None

        finally:
            try:
                ser.close()
            except Exception:
                pass
            logger.info(f"[TactileSensor {self.port}] worker exited")

    def _publish(self, xyz: np.ndarray, conn: np.ndarray, ts_ms: int):
        self.ring_buffer.put(
            {
                "xyz": xyz.copy(),
                "connected": conn.copy(),
                "device_ts_ms": np.int64(ts_ms),
                "host_timestamp": time.time(),
            }
        )


# ---------------------------------------------------------------------------
# Multi-sensor container
# ---------------------------------------------------------------------------

class TactileSensors:
    """Bundle of N TactileSensor workers, one per port in ``config.ports``.

    Use as a context manager. Pass the instance to ``XArmController(tactile=...)``
    or ``RealEnv(tactile=...)`` to wire up gripper safety; external API for
    ``step(pose, grasp)`` stays unchanged.
    """

    def __init__(self, shm_manager: SharedMemoryManager, config: TactileConfig):
        self.config = config
        self.sensors: List[TactileSensor] = [
            TactileSensor(shm_manager, port, config) for port in config.ports
        ]

    @property
    def ring_buffers(self):
        return [s.ring_buffer for s in self.sensors]

    @property
    def is_ready(self):
        return all(s.is_ready for s in self.sensors)

    def start(self, wait=True):
        for s in self.sensors:
            s.start(wait=wait)

    def stop(self, wait=True):
        for s in self.sensors:
            s.stop(wait=wait)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_latest(self) -> List[Dict]:
        """Latest snapshot per sensor (parent-process convenience)."""
        return [s.get_state() for s in self.sensors]

    def safety(self) -> Tuple[float, bool]:
        """(metric_value, is_safe_to_close) using the configured metric/threshold."""
        return evaluate_safety(self.get_latest(), self.config)
