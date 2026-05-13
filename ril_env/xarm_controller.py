import time
import enum
import multiprocessing as mp
import scipy.spatial.transform as st
import numpy as np
import logging

from typing import List, Optional
from xarm.wrapper import XArmAPI
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from dataclasses import dataclass, field
from ril_env.spacemouse import Spacemouse
from ril_env.tactile import TactileConfig, TactileSensors, evaluate_safety


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class XArmConfig:
    robot_ip: str = "192.168.1.223"
    frequency: int = 30
    position_gain: float = 2.0
    orientation_gain: float = 2.0
    home_pos: List[int] = field(default_factory=lambda: [0, 0, 0, 70, 0, 70, 0])
    home_speed: float = 50.0
    tcp_maxacc: int = 5000
    # Gripper position bounds in xArm SDK units. grasp=0.0 -> open_pos, grasp=1.0 -> close_pos.
    gripper_open_pos: int = 850
    gripper_close_pos: int = 0
    # Minimum change in grasp (in [0,1]) before re-issuing a gripper command.
    # Prevents spamming the gripper API every tick from small numeric noise.
    gripper_eps: float = 0.01
    verbose: bool = False  # switch off


def _apply_grasp(arm, grasp, previous_grasp, config):
    """Map grasp in [0,1] to a continuous gripper position and command it if changed.

    Returns the new previous_grasp value (unchanged if below epsilon).
    """
    grasp = float(np.clip(grasp, 0.0, 1.0))
    if abs(grasp - previous_grasp) < config.gripper_eps:
        return previous_grasp
    open_pos = config.gripper_open_pos
    close_pos = config.gripper_close_pos
    target = int(round(open_pos + grasp * (close_pos - open_pos)))
    code = arm.set_gripper_position(target, wait=False)
    if code != 0:
        logger.error(f"Error in set_gripper_position({target}): {code}")
        raise RuntimeError(f"Error in set_gripper_position({target}): {code}")
    return grasp


class XArm:
    """
    At this point, this is legacy code. If you want to record
    data, please use the multiprocessing-emabled XArmController.
    """

    def __init__(
        self,
        xarm_config: XArmConfig,
        tactile: Optional[TactileSensors] = None,
    ):
        self.config = xarm_config
        self.tactile = tactile
        self.init = False
        self._safety_active = False

        self.current_position = None
        self.current_orientation = None
        self.previous_grasp = 0.0

        if self.config.verbose:
            logger.setLevel(logging.DEBUG)

    @property
    def is_ready(self):
        return self.init

    def initialize(self):
        self.arm = XArmAPI(self.config.robot_ip)
        arm = self.arm

        arm.connect()
        arm.clean_error()
        arm.clean_warn()

        code = arm.motion_enable(enable=True)
        if code != 0:
            logger.error(f"Error in motion_enable: {code}")
            raise RuntimeError(f"Error in motion_enable: {code}")

        arm.set_tcp_maxacc(self.config.tcp_maxacc)

        code = arm.set_mode(1)
        if code != 0:
            logger.error(f"Error in set_mode: {code}")
            raise RuntimeError(f"Error in set_mode: {code}")

        code = arm.set_state(0)
        if code != 0:
            logger.error(f"Error in set_state: {code}")
            raise RuntimeError(f"Error in set_state: {code}")

        code, state = arm.get_state()
        if code != 0:
            logger.error(f"Error getting robot state: {code}")
            raise RuntimeError(f"Error getting robot state: {code}")
        if state != 0:
            logger.error(f"Robot is not ready to move. Current state: {state}")
            raise RuntimeError(f"Robot is not ready to move. Current state: {state}")
        else:
            logger.info(f"Robot is ready to move. Current state: {state}")

        err_code, warn_code = arm.get_err_warn_code()
        if err_code != 0 or warn_code != 0:
            logger.error(
                f"Error code: {err_code}, Warning code: {warn_code}. Cleaning error and warning."
            )
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_state(0)

        code = arm.set_gripper_mode(0)
        if code != 0:
            logger.error(f"Error in set_gripper_mode: {code}")
            raise RuntimeError(f"Error in set_gripper_mode: {code}")

        code = arm.set_gripper_enable(True)
        if code != 0:
            logger.error(f"Error in set_gripper_enable: {code}")
            raise RuntimeError(f"Error in set_gripper_enable: {code}")

        code = arm.set_gripper_speed(1000)
        if code != 0:
            logger.error(f"Error in set_gripper_speed: {code}")
            raise RuntimeError(f"Error in set_gripper_speed: {code}")

        self.init = True
        time.sleep(3)
        self.home()
        time.sleep(3)
        logger.info("Successfully initialized xArm.")

    def shutdown(self):
        if not self.init:
            logger.error("shutdown() called on an uninitialized xArm.")
            return
        self.home()
        self.arm.disconnect()
        logger.info("xArm shutdown complete.")

    def home(self):
        logger.info("Homing robot.")
        if not self.init:
            logger.error("xArm not initialized.")
            raise RuntimeError("xArm not initialized.")

        arm = self.arm
        arm.set_mode(0)
        arm.set_state(0)
        code = arm.set_gripper_position(850, wait=False)
        if code != 0:
            logger.error(f"Error in set_gripper_position (open, homing): {code}")
            raise RuntimeError(f"Error in set_gripper_position (open, homing): {code}")

        code = arm.set_servo_angle(
            angle=self.config.home_pos, speed=self.config.home_speed, wait=True
        )
        if code != 0:
            logger.error(f"Error in set_servo_angle (homing): {code}")
            raise RuntimeError(f"Error in set_servo_angle (homing): {code}")
        arm.set_mode(1)
        arm.set_state(0)

        code, pose = arm.get_position()
        if code != 0:
            logger.error(f"Failed to query initial pose: {code}")
            raise RuntimeError(f"Failed to query initial pose: {code}")
        else:
            self.current_position = np.array(pose[:3])
            self.current_orientation = np.array(pose[3:])
            logger.debug(
                f"Initial pose set: pos={self.current_position}, ori={self.current_orientation}"
            )

    def step(self, dpos, drot, grasp):
        if not self.init:
            logger.error("xArm not initialized. Use it in a 'with' block")
            raise RuntimeError("xArm not initialized. Use it in a 'with' block")

        dpos *= self.config.position_gain
        drot *= self.config.orientation_gain

        curr_rot = st.Rotation.from_euler("xyz", self.current_orientation, degrees=True)
        delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
        final_rot = delta_rot * curr_rot

        self.current_orientation = final_rot.as_euler("xyz", degrees=True)
        self.current_position += dpos

        logger.debug(f"Current position: {self.current_position}")
        logger.debug(f"Current orientation: {self.current_orientation}")

        code = self.arm.set_servo_cartesian(
            np.concatenate((self.current_position, self.current_orientation)),
            is_radian=False,
        )
        if code != 0:
            logger.error(f"Error in set_servo_cartesian in step(): {code}")
            raise RuntimeError(f"Error in set_servo_cartesian in step(): {code}")

        # Tactile safety: if contact exceeds threshold, don't allow closing further.
        if self.tactile is not None:
            metric_val, is_safe = self.tactile.safety()
            if not is_safe and grasp > self.previous_grasp:
                if not self._safety_active:
                    logger.warning(
                        f"[XArm] tactile safety engaged "
                        f"(metric={metric_val:.2f}); clamping grasp to "
                        f"previous_grasp={self.previous_grasp:.3f}"
                    )
                self._safety_active = True
                grasp = self.previous_grasp
            else:
                self._safety_active = False

        self.previous_grasp = _apply_grasp(
            self.arm, grasp, self.previous_grasp, self.config
        )

    def get_state(self):
        state = {}

        code, actual_pose = self.arm.get_position(is_radian=False)
        if code != 0:
            logger.error(f"Error getting TCP pose: code {code}")
            raise RuntimeError(f"Error getting TCP pose: code {code}")
        state["ActualTCPPose"] = actual_pose

        tcp_speed_attr = self.arm.realtime_tcp_speed
        state["ActualTCPSpeed"] = tcp_speed_attr() if callable(tcp_speed_attr) else tcp_speed_attr

        code, actual_angles = self.arm.get_servo_angle(is_radian=False)
        if code != 0:
            logger.error(f"Error getting joint angles: code {code}")
            raise RuntimeError(f"Error getting joint angles: code {code}")
        state["ActualQ"] = actual_angles

        joint_speeds_attr = self.arm.realtime_joint_speeds
        state["ActualQd"] = joint_speeds_attr() if callable(joint_speeds_attr) else joint_speeds_attr

        return state

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


class Command(enum.Enum):
    STOP = 0
    STEP = 1
    HOME = 2
    # May add more commands here. e.g. SCHEDULE_WAYPOINT


class XArmController(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        xarm_config: XArmConfig,
        tactile: Optional[TactileSensors] = None,
    ):
        super().__init__(name="XArmController")

        # Tactile safety wiring. Only the picklable pieces are stored on self;
        # the TactileSensors mp.Process workers are managed by the parent.
        if tactile is not None:
            self._tactile_ring_buffers = list(tactile.ring_buffers)
            self._tactile_config: Optional[TactileConfig] = tactile.config
        else:
            self._tactile_ring_buffers = None
            self._tactile_config = None

        self.config = xarm_config
        self.robot_ip = xarm_config.robot_ip
        self.frequency = xarm_config.frequency
        self.position_gain = xarm_config.position_gain
        self.orientation_gain = xarm_config.orientation_gain
        self.home_pos = xarm_config.home_pos
        self.home_speed = xarm_config.home_speed
        self.tcp_maxacc = xarm_config.tcp_maxacc
        self.verbose = xarm_config.verbose

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        # Events for synchronization
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

        # Build Input Queue
        queue_example = {
            "cmd": Command.STEP.value,
            "target_pose": np.zeros(6, dtype=np.float64),
            "grasp": 0.0,
            "duration": 0.0,
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=queue_example, buffer_size=256
        )

        # Build Ring Buffer.
        try:
            arm_temp = XArmAPI(self.robot_ip)
            arm_temp.connect()
            arm_temp.clean_error()
            arm_temp.clean_warn()
            arm_temp.set_tcp_maxacc(xarm_config.tcp_maxacc)
            code = arm_temp.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"motion_enable error: {code}")
            code = arm_temp.set_mode(1)
            if code != 0:
                raise RuntimeError(f"set_mode error: {code}")
            code = arm_temp.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state error: {code}")

            state_example = {}

            # Get TCPPose: use get_position.
            code, pos = arm_temp.get_position(is_radian=False)
            if code == 0:
                state_example["TCPPose"] = np.array(pos[:6], dtype=np.float64)
            else:
                state_example["TCPPose"] = np.zeros(6, dtype=np.float64)

            # Get TCPSpeed: use realtime_tcp_speed.
            try:
                if callable(arm_temp.realtime_tcp_speed):
                    tcp_speed = arm_temp.realtime_tcp_speed()
                else:
                    tcp_speed = arm_temp.realtime_tcp_speed
                state_example["TCPSpeed"] = np.array(tcp_speed, dtype=np.float64)
            except Exception:
                state_example["TCPSpeed"] = np.zeros(6, dtype=np.float64)

            # Get JointAngles: use get_servo_angle()
            code, angles = arm_temp.get_servo_angle(is_radian=False)
            if code == 0:
                state_example["JointAngles"] = np.array(angles, dtype=np.float64)
            else:
                state_example["JointAngles"] = np.zeros(7, dtype=np.float64)

            # Get JointSpeeds: handle callable or value directly.
            try:
                if callable(arm_temp.realtime_joint_speeds):
                    joint_speeds = arm_temp.realtime_joint_speeds()
                else:
                    joint_speeds = arm_temp.realtime_joint_speeds
                state_example["JointSpeeds"] = np.array(joint_speeds, dtype=np.float64)
            except Exception:
                state_example["JointSpeeds"] = np.zeros(7, dtype=np.float64)

            # Robot timestamp (absolute for now).
            state_example["robot_receive_timestamp"] = time.time()

            # Initialize our grasp state.
            self.previous_grasp = 0.0
            state_example["Grasp"] = self.previous_grasp
            # 0.0 / 1.0 flag set when tactile safety clamped a closing command.
            state_example["TactileSafetyActive"] = 0.0
            state_example["TactileMetric"] = 0.0

            self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=state_example,
                get_max_k=128,
                get_time_budget=0.2,
                put_desired_frequency=self.frequency,
            )

            # Disconnect the temporary connection; the main loop will reconnect.
            arm_temp.disconnect()

        except Exception as e:
            logger.error(f"Error during initial state fetch: {e}")
            raise e

        # Store the last target pose; initialize it from the example.
        self.last_target_pose = state_example["TCPPose"]

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait(3)
            assert self.is_alive(), "XArmController did not start correctly."
        logger.debug(f"[XArmController] Process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
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
            logger.debug("[XArmController] In get_state(), k is None")
            return self.ring_buffer.get()
        else:
            return self.ring_buffer.get_last_k(k)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def step(self, pose, grasp):
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)
        grasp = float(np.clip(grasp, 0.0, 1.0))

        cmd = {
            "cmd": Command.STEP.value,
            "target_pose": pose,
            "grasp": grasp,
            "duration": 0.02,
            "target_time": time.time() + 0.02,
        }

        self.input_queue.put(cmd)

    def run(self):
        try:
            logger.info(f"[XArmController] Connecting to xArm at {self.robot_ip}")
            arm = XArmAPI(self.robot_ip)
            arm.connect()
            arm.clean_error()
            arm.clean_warn()
            arm.set_tcp_maxacc(self.tcp_maxacc)

            code = arm.motion_enable(True)
            if code != 0:
                raise RuntimeError(f"[XArmController] motion_enable error: {code}")
            code = arm.set_mode(1)
            if code != 0:
                raise RuntimeError(f"[XArmController] set_mode error: {code}")
            code = arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"[XArmController] set_state error: {code}")

            code, pos = arm.get_position(is_radian=False)
            if code == 0:
                self.last_target_pose = np.array(pos[:6], dtype=np.float64)
            else:
                logger.error(
                    "[XArmController] Failed to get initial position; defaulting to zeros."
                )
                self.last_target_pose = np.zeros(6, dtype=np.float64)

            start_time = time.time()
            self.ready_event.set()

            dt = 1.0 / self.frequency
            iter_idx = 0
            # Edge tracker so we only log on transitions into the unsafe state.
            safety_active_prev = False

            while not self.stop_event.is_set():
                grasp = self.previous_grasp
                t_start = time.time()

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: commands[key][i] for key in commands}
                    cmd = command["cmd"]
                    if cmd == Command.STOP.value:
                        logger.debug("[XArmController] Received STOP command.")
                        self.stop_event.set()
                        break
                    elif cmd == Command.STEP.value:
                        target_pose = np.array(command["target_pose"], dtype=np.float64)
                        grasp = command["grasp"]
                        self.last_target_pose = target_pose
                        logger.debug(f"[XArmController] New target pose: {target_pose}")
                    elif cmd == Command.HOME.value:
                        # Currently, there are some issues here. It is best to move closer
                        # to home before homing, otherwise it is *very* dangerous.
                        logger.info("[XArmController] Received HOME command.")
                        arm.set_mode(0)
                        arm.set_state(0)
                        code = arm.set_gripper_position(850, wait=False)
                        if code != 0:
                            logger.error(
                                f"Error in set_gripper_position (HOME open): {code}"
                            )
                        code = arm.set_servo_angle(
                            angle=self.home_pos, speed=self.home_speed, wait=True
                        )
                        arm.set_mode(1)
                        arm.set_state(0)
                        code, pos = arm.get_position(is_radian=False)
                        if code == 0:
                            self.last_target_pose = np.array(pos[:6], dtype=np.float64)
                    else:
                        logger.error(f"[XArmController] Unknown command: {cmd}")

                # If the last command wasn't STOP or HOME, we do a servo step
                code = arm.set_servo_cartesian(
                    list(self.last_target_pose), is_radian=False
                )

                # Tactile safety: refuse to close beyond previous_grasp if the
                # contact metric is over threshold (or readings are stale).
                tactile_metric = 0.0
                safety_active = False
                if self._tactile_ring_buffers is not None and self._tactile_config is not None:
                    try:
                        tactile_states = [rb.get() for rb in self._tactile_ring_buffers]
                        tactile_metric, is_safe = evaluate_safety(
                            tactile_states, self._tactile_config
                        )
                    except Exception as e:
                        logger.error(f"[XArmController] tactile read failed: {e}")
                        is_safe = False
                    if not is_safe and grasp > self.previous_grasp:
                        grasp = self.previous_grasp
                        safety_active = True
                if safety_active and not safety_active_prev:
                    logger.warning(
                        f"[XArmController] tactile safety engaged "
                        f"(metric={tactile_metric:.2f}, "
                        f"threshold={self._tactile_config.safety_threshold:.2f}); "
                        f"holding grasp at {self.previous_grasp:.3f}"
                    )
                safety_active_prev = safety_active

                # Update gripper (continuous mapping; epsilon-gated).
                self.previous_grasp = _apply_grasp(
                    arm, grasp, self.previous_grasp, self.config
                )

                if code != 0:
                    logger.error(f"[XArmController] set_servo_cartesian error: {code}")

                # Update robot state
                state = {}
                code, pos = arm.get_position(is_radian=False)
                if code == 0:
                    state["TCPPose"] = np.array(pos[:6], dtype=np.float64)
                else:
                    state["TCPPose"] = np.zeros(6, dtype=np.float64)

                try:
                    if callable(arm.realtime_tcp_speed):
                        tcp_speed = arm.realtime_tcp_speed()
                    else:
                        tcp_speed = arm.realtime_tcp_speed
                    state["TCPSpeed"] = np.array(tcp_speed, dtype=np.float64)
                except Exception as e:
                    logger.error(f"Error in realtime_tcp_speed: {e}")
                    state["TCPSpeed"] = np.zeros(6, dtype=np.float64)

                code, angles = arm.get_servo_angle(is_radian=False)
                if code == 0:
                    state["JointAngles"] = np.array(angles, dtype=np.float64)
                else:
                    state["JointAngles"] = np.zeros(7, dtype=np.float64)

                try:
                    if callable(arm.realtime_joint_speeds):
                        joint_speeds = arm.realtime_joint_speeds()
                    else:
                        joint_speeds = arm.realtime_joint_speeds
                    state["JointSpeeds"] = np.array(joint_speeds, dtype=np.float64)
                except Exception as e:
                    logger.error(f"Error in realtime_joint_speeds: {e}")
                    state["JointSpeeds"] = np.zeros(7, dtype=np.float64)

                state["Grasp"] = self.previous_grasp
                state["TactileSafetyActive"] = 1.0 if safety_active else 0.0
                state["TactileMetric"] = float(tactile_metric)
                state["robot_receive_timestamp"] = time.time() - start_time

                # Update ring buffer (data)
                self.ring_buffer.put(state)

                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                iter_idx += 1
                logger.debug(
                    f"[XArmController] Iteration {iter_idx} at {1.0/(time.time()-t_start):.2f} Hz"
                )
        except Exception as e:
            logger.error(f"[XArmController] Exception in control loop: {e}")
        finally:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
                logger.info(
                    f"[XArmController] Disconnected from xArm at {self.robot_ip}"
                )
            except Exception as e:
                logger.error(f"[XArmController] Cleanup error: {e}")
            self.ready_event.set()


def main():
    with SharedMemoryManager() as shm_manager, Spacemouse(deadzone=0.4) as sm:
        xarm_config = XArmConfig()
        xarm_ctrl = XArmController(
            shm_manager=shm_manager,
            xarm_config=xarm_config,
        )
        xarm_ctrl.start(wait=True)
        print("XArmController started and ready.")

        # Keep our local "target_pose" so orientation accumulates properly
        current_target_pose = xarm_ctrl.last_target_pose.copy()

        last_timestamp = None
        try:
            while True:
                loop_start = time.monotonic()

                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]
                drot = sm_state[3:]
                grasp = sm.grasp

                # Right button -> HOME
                # DO NOT MOVE THE SPACEMOUSE WHILE CLICKING THIS.
                # The HOME command is buggy and best avoided...
                """
                if sm.is_button_pressed(1):
                    command = {
                        "cmd": Command.HOME.value,
                        "target_pose": np.zeros(6, dtype=np.float64),
                        "grasp": 0.0,
                        "duration": 0.0,
                        "target_time": time.time(),
                    }
                    xarm_ctrl.input_queue.put(command)

                    time.sleep(1.0)
                    updated_state = xarm_ctrl.get_state()
                    new_pose = updated_state["TCPPose"]
                    current_target_pose[:] = new_pose
                    continue
                """

                dpos *= xarm_ctrl.position_gain
                drot *= xarm_ctrl.orientation_gain

                curr_orientation = current_target_pose[3:]
                curr_rot = st.Rotation.from_euler("xyz", curr_orientation, degrees=True)
                delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
                final_rot = delta_rot * curr_rot

                current_target_pose[:3] += dpos
                current_target_pose[3:] = final_rot.as_euler("xyz", degrees=True)

                # This is a workaround that does not use home_pos
                # This will have to be fixed later.
                # This is also like insanely dangerous...
                """
                if sm.is_button_pressed(1):
                    current_target_pose = [
                        475.791901,
                        -1.143693,
                        244.719421,
                        179.132906,
                        -0.010084,
                        0.77567,
                    ]
                """

                command = {
                    "cmd": Command.STEP.value,
                    "target_pose": current_target_pose,
                    "grasp": grasp,
                    "duration": 0.02,
                    "target_time": time.time() + 0.02,
                }
                xarm_ctrl.input_queue.put(command)

                # Check the ring buffer to see if the child updated
                state = xarm_ctrl.get_state(k=1)
                logger.debug(f"Most recent state: {state}")
                ts = state.get("robot_receive_timestamp")[0]
                if ts != last_timestamp:
                    logger.debug(f"Ring buffer updated, time: {ts:.3f}")
                    last_timestamp = ts

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0, 0.02 - elapsed))
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            xarm_ctrl.stop(wait=True)
            print("XArmController stopped.")


if __name__ == "__main__":
    main()
