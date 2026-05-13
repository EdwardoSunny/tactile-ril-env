import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Path to your replay buffer
zarr_path = Path("recordings/replay_buffer.zarr")

# Open the Zarr archive
replay_buffer = zarr.open(str(zarr_path), mode='r')

# Print available episode count
episode_ends = replay_buffer['meta']['episode_ends'][:]
print(f"Total episodes: {len(episode_ends)}")

# Get info about the last episode
if len(episode_ends) > 0:
    # If we have multiple episodes, calculate start index
    start_idx = 0
    if len(episode_ends) > 1:
        start_idx = episode_ends[-2]
    
    # End index is the last episode's end
    end_idx = episode_ends[-1]
    
    print(f"Episode length: {end_idx - start_idx} steps")
    
    # Get data for the latest episode
    episode_slice = slice(start_idx, end_idx)
    
    # Get TCP Poses
    tcp_poses = replay_buffer['data']['TCPPose'][episode_slice]
    timestamps = replay_buffer['data']['timestamp'][episode_slice]
    
    # Print shape and sample
    print(f"TCP Poses shape: {tcp_poses.shape}")
    print(f"First TCP Pose: {tcp_poses[0]}")
    print(f"Last TCP Pose: {tcp_poses[-1]}")
    
    # Plot TCP position over time
    plt.figure(figsize=(12, 6))
    
    # Extract x, y, z positions
    x_pos = tcp_poses[:, 0]
    y_pos = tcp_poses[:, 1]
    z_pos = tcp_poses[:, 2]
    print(f"x, y, z 0 {x_pos[0]} {y_pos[0]} {z_pos[0]}")
    # Time in seconds from the start
    time_seconds = timestamps - timestamps[0]
    
    plt.plot(time_seconds, x_pos, label='X Position')
    plt.plot(time_seconds, y_pos, label='Y Position')
    plt.plot(time_seconds, z_pos, label='Z Position')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position (mm)')
    plt.title('Robot End-Effector Position')
    plt.legend()
    plt.grid(True)
    
    # Save and show
    # plt.savefig('tcp_position_plot.png')
    plt.show()
