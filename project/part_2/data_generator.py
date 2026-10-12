import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final.rollout_loader import load_rollouts

from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists

# Create training and testing directories
TRAINING_DIR = FINAL_DIR / "training_trajectories"
TESTING_DIR = FINAL_DIR / "testing_trajectories"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
TESTING_DIR.mkdir(parents=True, exist_ok=True)

PRINT_PLOTS = False  # Set to True to enable plotting
RECORDING = True  # Set to True to enable data recording

downsample_rate = 2

NUM_TRAJECTORIES = 200
TRAIN_SPLIT = 0.7  # 70% for training, 15% for validation, 15% for testing


def generate_random_trajectories(num_trajectories, cube_position, workspace_radius, 
                                init_joint_angles, z_range=[0.05, 0.3]):
    
    np.random.seed(42)  # For reproducibility
    
    list_of_desired_cartesian_positions = []
    list_of_desired_cartesian_orientations = []
    list_of_type_of_control = []
    list_of_duration_per_desired_cartesian_positions = []
    list_of_initialjoint_positions = []
    
    cube_x, cube_y, cube_z = cube_position
    
    for i in range(num_trajectories):
        # Using range from -90° to +90° (in front of robot)
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        
        radius = np.random.uniform(workspace_radius * 0.7, workspace_radius * 1.2)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Random height within specified range
        z = np.random.uniform(z_range[0], z_range[1])
        
        desired_pos = np.array([x, y, z])
        
        # Vary orientations slightly 
        if i % 3 == 0:
            # Straight down 
            desired_ori = [0.0, 0.0, 0.0, 1.0]
        elif i % 3 == 1:
            # Slight rotation around z-axis
            angle_z = np.random.uniform(-np.pi/6, np.pi/6)
            desired_ori = [0.0, 0.0, np.sin(angle_z/2), np.cos(angle_z/2)]
        else:
            # Slight tilt for varied approach angles
            angle_tilt_x = np.random.uniform(-np.pi/8, np.pi/8)
            angle_tilt_y = np.random.uniform(-np.pi/8, np.pi/8)
            desired_ori = [np.sin(angle_tilt_x/2), np.sin(angle_tilt_y/2), 0.0, 
                          np.cos(angle_tilt_x/2) * np.cos(angle_tilt_y/2)]
        
        duration = np.random.uniform(3.0, 7.0)
        
        control_choice = np.random.choice(["pos", "pos", "pos", "pos"], 
                                         p=[0.25, 0.25, 0.25, 0.25])
        
        list_of_desired_cartesian_positions.append(desired_pos.tolist())
        list_of_desired_cartesian_orientations.append(desired_ori)
        list_of_type_of_control.append(control_choice)
        list_of_duration_per_desired_cartesian_positions.append(duration)
        list_of_initialjoint_positions.append(init_joint_angles)
    
    return (list_of_desired_cartesian_positions, 
            list_of_desired_cartesian_orientations,
            list_of_type_of_control,
            list_of_duration_per_desired_cartesian_positions,
            list_of_initialjoint_positions)


def main():
    conf_file_name = "pandaconfig.json"  
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.join(parent_dir)
    
    print(f"Looking for config in: {root_dir}")
    
    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    print(f"Initial cartesian position: {init_cartesian_pos}")
    
    # Check joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")

    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint vel limits: {joint_vel_limits}")
    
    # Define cube position (at 90 degrees - forward from robot)
    cube_position = [0.8, 0.0, 0.0]  # [x, y, z] 
    
    # Define 180-degree semicircular workspace
    workspace_radius = 0.8  # 80cm radius from robot base
    z_range = [0.0, 0.8]  # Height range: 0cm to 80cm above table
    
    print(f"\nGenerating {NUM_TRAJECTORIES} random trajectories...")
    print(f"Workspace: 180° semicircle (cube at center/90°)")
    print(f"Cube position: {cube_position}")
    print(f"Workspace radius: {workspace_radius}m")
    print(f"Height range: {z_range[0]}m to {z_range[1]}m")
    
    # Generate random trajectories
    (list_of_desired_cartesian_positions,
     list_of_desired_cartesian_orientations,
     list_of_type_of_control,
     list_of_duration_per_desired_cartesian_positions,
     list_of_initialjoint_positions) = generate_random_trajectories(
        NUM_TRAJECTORIES, cube_position, workspace_radius, init_joint_angles, z_range
    )
    
    # Determine train/test split
    num_train = int(NUM_TRAJECTORIES * TRAIN_SPLIT)
    train_indices = set(range(num_train))
    
    print(f"\nTraining trajectories: {num_train} (indices 0-{num_train-1})")
    print(f"Testing trajectories: {NUM_TRAJECTORIES - num_train} (indices {num_train}-{NUM_TRAJECTORIES-1})")
    
    # Command and control loop
    cmd = MotorCommands()  


    kp_pos = 100  
    kp_ori = 100  

    kp = 1000
    kd = 100

    q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []

    current_time = 0  
    time_step = sim.GetTimeStep()

    successful_trajectories = 0

    for i in range(len(list_of_desired_cartesian_positions)):
        print(f"\nExecuting trajectory {i+1}/{NUM_TRAJECTORIES}...")

        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        
        if list_of_initialjoint_positions[i] is None:
            init_position = init_joint_angles
        else:
            init_position = list_of_initialjoint_positions[i]
            
        diff_kin = CartesianDiffKin(dyn_model, controlled_frame_name, init_position, 
                                    desired_cartesian_pos, np.zeros(3), desired_cartesian_ori, 
                                    np.zeros(3), time_step, type_of_control, kp_pos, kp_ori, 
                                    np.array(joint_vel_limits))
        steps = int(duration_per_desired_cartesian_pos / time_step)

        # Reinitialize the robot to the initial position
        sim.ResetPose()
        if init_position is not None:
            sim.SetjointPosition(init_position)
            
        # Data collection loop
        for t in range(steps):
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            tau_mes = np.asarray(sim.GetMotorTorques(0), dtype=float)

            pd_d = [0.0, 0.0, 0.0]  
            ori_d_des = [0.0, 0.0, 0.0]  
            
            
            q_des, qd_des_clip = CartesianDiffKin(dyn_model, controlled_frame_name, q_mes, 
                                                  desired_cartesian_pos, pd_d, desired_cartesian_ori, 
                                                  ori_d_des, time_step, type_of_control, kp_pos, 
                                                  kp_ori, np.array(joint_vel_limits))
            
            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  
            sim.Step(cmd, "torque")  

            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            if RECORDING:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                q_d_all.append(q_des)
                qd_d_all.append(qd_des_clip)
                tau_mes_all.append(tau_mes)
                cart_pos_all.append(cart_pos)
                cart_ori_all.append(cart_ori)

            time.sleep(time_step)  
            current_time += time_step
    
        current_time = 0

        if len(q_mes_all) > 0:    
            # Downsample data
            q_mes_all_downsampled = q_mes_all[::downsample_rate]
            qd_mes_all_downsampled = qd_mes_all[::downsample_rate]
            q_d_all_downsampled = q_d_all[::downsample_rate]
            qd_d_all_downsampled = qd_d_all[::downsample_rate]
            tau_mes_all_downsampled = tau_mes_all[::downsample_rate]
            cart_pos_all_downsampled = cart_pos_all[::downsample_rate]
            cart_ori_all_downsampled = cart_ori_all[::downsample_rate]

            time_array = [time_step * downsample_rate * j for j in range(len(q_mes_all_downsampled))]

            # Determine if this is a training or testing trajectory
            if i in train_indices:
                save_dir = TRAINING_DIR
                data_type = "training"
            else:
                save_dir = TESTING_DIR
                data_type = "testing"

            # Save data to pickle file
            filename = save_dir / f"data_{i}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump({
                    'time': time_array,
                    'q_mes_all': q_mes_all_downsampled,
                    'qd_mes_all': qd_mes_all_downsampled,
                    'q_d_all': q_d_all_downsampled,
                    'qd_d_all': qd_d_all_downsampled,
                    'tau_mes_all': tau_mes_all_downsampled,
                    'cart_pos_all': cart_pos_all_downsampled,
                    'cart_ori_all': cart_ori_all_downsampled,
                    'desired_cartesian_pos': desired_cartesian_pos,
                    'desired_cartesian_ori': desired_cartesian_ori,
                    'control_type': type_of_control,
                    'duration': duration_per_desired_cartesian_pos
                }, f)
            print(f"Saved {data_type} trajectory to {filename}")
            successful_trajectories += 1

        # Reinitialize data storage lists
        q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []

        if PRINT_PLOTS and i < 3:  
            print("Plotting downsampled data...")
            # Plot joint positions
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(q_mes_all_downsampled[0])):
                joint_positions = [q[joint_idx] for q in q_mes_all_downsampled]
                plt.plot(time_array, joint_positions, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Positions (rad)')
            plt.title(f'Trajectory {i}: Joint Positions')
            plt.legend()
            plt.grid(True)
            plt.show()

    print(f"\n{'='*60}")
    print(f"Data generation complete!")
    print(f"Successfully generated {successful_trajectories}/{NUM_TRAJECTORIES} trajectories")
    print(f"Training data: {TRAINING_DIR}")
    print(f"Testing data: {TESTING_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
    
    print("\nTesting rollout loader...")
    train_indices = list(range(min(4, int(NUM_TRAJECTORIES * TRAIN_SPLIT))))
    test_indices = list(range(int(NUM_TRAJECTORIES * TRAIN_SPLIT), 
                             min(int(NUM_TRAJECTORIES * TRAIN_SPLIT) + 2, NUM_TRAJECTORIES)))
    
    if train_indices:
        rls_train = load_rollouts(indices=train_indices, directory=TRAINING_DIR)
        print(f"Loaded {len(rls_train)} training rollouts")
        if len(rls_train) > 0:
            print(f"First training rollout - time points: {len(rls_train[0].time)}, "
                  f"positions: {len(rls_train[0].q_mes_all)}, "
                  f"velocities: {len(rls_train[0].qd_mes_all)}")
    
    if test_indices:
        rls_test = load_rollouts(indices=test_indices, directory=TESTING_DIR)
        print(f"Loaded {len(rls_test)} testing rollouts")
        if len(rls_test) > 0:
            print(f"First testing rollout - time points: {len(rls_test[0].time)}, "
                  f"positions: {len(rls_test[0].q_mes_all)}, "
                  f"velocities: {len(rls_test[0].qd_mes_all)}")