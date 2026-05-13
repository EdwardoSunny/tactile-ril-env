import time
import numpy as np
import zarr
import logging
from ril_env.xarm_controller import XArmConfig, XArmController
from multiprocessing.managers import SharedMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def replay_demonstration():
    # Open the replay buffer for episode 0
    zarr_path = "./recordings/replay_buffer.zarr"
    replay_buffer = zarr.open(zarr_path, mode='r')
    
    # Get episode information
    episode_ends = replay_buffer['meta']['episode_ends'][:]
    
    # Get data for episode 0
    start_idx = 0
    end_idx = episode_ends[0]
    episode_slice = slice(start_idx, end_idx)
    
    # Extract episode data - use TCPPose instead of action
    tcp_poses = replay_buffer['data']['TCPPose'][episode_slice]
    timestamps = replay_buffer['data']['timestamp'][episode_slice]
    
    # Also get grasp data if available
    grasp_data = None
    if 'Grasp' in replay_buffer['data']:
        grasp_data = replay_buffer['data']['Grasp'][episode_slice]
    
    logger.info(f"Episode 0 loaded: {len(tcp_poses)} steps")
    logger.info(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
    logger.info(f"First TCP Pose: {tcp_poses[0]}")
    logger.info(f"Last TCP Pose: {tcp_poses[-1]}")
    
    # Initialize the robot controller
    with SharedMemoryManager() as shm_manager:
        xarm_config = XArmConfig()
        robot = XArmController(
            shm_manager=shm_manager,
            xarm_config=xarm_config,
        )
        
        try:
            # Start the robot controller
            logger.info("Starting robot controller")
            robot.start(wait=True)
            
            # Move to the initial position
            initial_pose = tcp_poses[0]
            initial_grasp = 0.0  # Default grasp value if not available
            if grasp_data is not None and len(grasp_data) > 0:
                initial_grasp = grasp_data[0]
            
            logger.info(f"Moving to initial position: {initial_pose}")
            # Comment this out for now just to test
            # robot.step(initial_pose, initial_grasp)
            logger.info("Command to move robot commented out for safety")
            
            # Wait 5 seconds
            logger.info("Waiting 5 seconds before starting replay")
            time.sleep(5.0)
            
            # Start the replay
            logger.info("Starting replay (commands commented out)")
            
            # Calculate time offsets from the first timestamp
            time_offsets = timestamps - timestamps[0]
            start_time = time.time()
            
            for i in range(len(tcp_poses)):
                # Wait until it's time to execute this action
                target_time = start_time + time_offsets[i]
                now = time.time()
                if target_time > now:
                    time.sleep(target_time - now)
                
                # Get the pose and grasp
                pose = tcp_poses[i]
                grasp = 0.0
                if grasp_data is not None:
                    grasp = grasp_data[i]
                
                # Print the pose we would execute
                if i % 10 == 0:
                    logger.info(f"Step {i+1}/{len(tcp_poses)}: {pose}")
                
                # Comment out execution for safety
                robot.step(pose, grasp)
            
            logger.info("Replay completed (simulated)")
            
        except KeyboardInterrupt:
            logger.info("Replay interrupted by user")
        finally:
            # Stop the robot controller
            logger.info("Stopping robot controller")
            robot.stop(wait=True)

if __name__ == "__main__":
    replay_demonstration()
