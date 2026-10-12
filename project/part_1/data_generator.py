import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle
import torch


import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final.rollout_loader import load_rollouts
from final.dmlp import DMLP
from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent  # this is .../final
FINAL_DIR.mkdir(parents=True, exist_ok=True)  # safe if it already exists


PRINT_PLOTS = False  # Set to True to enable plotting
RECORDING = True  # Set to True to enable data recording

# downsample rate needs to be bigger than one (is how much I steps I skip when i downsample the data)
#在对数据进行下采样（downsampling）时，每隔 2 个时间步或数据点取一个样本，也就是跳过 1 个，保留 1 个。
downsample_rate = 2
NUM_TRAJECTORIES = 200

# Function to get downsample rate from the user without blocking the simulation loop
def get_downsample_rate():
    try:
        rate = int(input("Enter downsample rate (integer >=1): "))
        if rate < 1:
            print("Invalid downsample rate. Must be >= 1.")
            return None
        return rate
    except ValueError:
        print("Please enter a valid integer.")
        return None

def generate_random_trajectories(num_trajectories, init_joint_angles):
    
    np.random.seed(42)  # For reproducibility
    
    list_of_desired_cartesian_positions = []
    list_of_desired_cartesian_orientations = []
    list_of_type_of_control = []
    list_of_duration_per_desired_cartesian_positions = []
    list_of_initialjoint_positions = []
    
    for i in range(num_trajectories):
        x = np.random.uniform(-0.85, 0.85)
        y = np.random.uniform(-0.85, 0.85)
        z = np.random.uniform(0, 0.6)
        
        desired_pos = np.array([x, y, z])
        
        # # Vary orientations slightly 
        # if i % 3 == 0:
        #     # Straight down 
        #     desired_ori = [0.0, 0.0, 0.0, 1.0]
        # elif i % 3 == 1:
        #     # Slight rotation around z-axis
        #     angle_z = np.random.uniform(-np.pi/6, np.pi/6)
        #     desired_ori = [0.0, 0.0, np.sin(angle_z/2), np.cos(angle_z/2)]
        # else:
        #     # Slight tilt for varied approach angles
        #     angle_tilt_x = np.random.uniform(-np.pi/8, np.pi/8)
        #     angle_tilt_y = np.random.uniform(-np.pi/8, np.pi/8)
        #     desired_ori = [np.sin(angle_tilt_x/2), np.sin(angle_tilt_y/2), 0.0, 
        #                   np.cos(angle_tilt_x/2) * np.cos(angle_tilt_y/2)]
        
        # Vary duration for diversity in motion speed
        duration = 5 #np.random.uniform(3.0, 7.0)
        
        control_choice = "pos"
        orientation = [0.0, 0.0, 0.0, 1.0]
        
        list_of_desired_cartesian_positions.append(desired_pos.tolist())
        list_of_desired_cartesian_orientations.append(orientation)
        list_of_type_of_control.append(control_choice)
        list_of_duration_per_desired_cartesian_positions.append(duration)
        list_of_initialjoint_positions.append(init_joint_angles)
    
    return (list_of_desired_cartesian_positions, 
            list_of_desired_cartesian_orientations,
            list_of_type_of_control,
            list_of_duration_per_desired_cartesian_positions,
            list_of_initialjoint_positions)


def main():

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext = root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos,init_R = dyn_model.ComputeFK(init_joint_angles,controlled_frame_name)
    # print init joint
    print(f"Initial joint angles: {init_joint_angles}")
    
    # check joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")


    joint_vel_limits = sim.GetBotJointsVelLimit()
    # increase the joint vel limits to not trigger warning in the simulation
    #joint_vel_limits = [vel * 100 for vel in joint_vel_limits]
    
    print(f"joint vel limits: {joint_vel_limits}")
    
    # desired value for regulation
    q_des =  init_joint_angles
    qd_des_clip = np.zeros(num_joints)
    
    
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # P controller high level
    kp_pos = 100  # position
    kp_ori = 0    # orientation

    # PD controller gains low level (feedback gain)
    kp = 1000
    kd = 100

    # desired cartesian position
    list_of_desired_cartesian_positions = [[0.5,0.0,0.1], 
                                           [0.4,0.2,0.1], 
                                           [0.4,-0.2,0.1], 
                                           [0.5,0.0,0.1]]
    # desired cartesian orientation in quaternion (XYZW)
    list_of_desired_cartesian_orientations = [[0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0, 1.0]]
    list_of_type_of_control = ["pos", "pos", "pos", "pos"] # "pos",  "ori" or "both"
    list_of_duration_per_desired_cartesian_positions = [5.0, 5.0, 5.0, 5.0] # in seconds
    list_of_initialjoint_positions = [init_joint_angles, init_joint_angles, init_joint_angles, init_joint_angles]


    # Generate random trajectories
    (list_of_desired_cartesian_positions,
     list_of_desired_cartesian_orientations,
     list_of_type_of_control,
     list_of_duration_per_desired_cartesian_positions,
     list_of_initialjoint_positions) = generate_random_trajectories(
        NUM_TRAJECTORIES, init_joint_angles
    )

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []
    angle_diff_all, velocity_diff_all,  tau_cmd_all = [], [], []

    current_time = 0  # Initialize current time
    time_step = sim.GetTimeStep()
    # 加载模型FINAL_DIR / f"data_{i}.pkl"
    with open(FINAL_DIR / f'rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open(FINAL_DIR / f'mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)





    for i in range(len(list_of_desired_cartesian_positions)):

        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration_per_desired_cartesian_pos = list_of_duration_per_desired_cartesian_positions[i]
        type_of_control = list_of_type_of_control[i]
        if list_of_initialjoint_positions[i] is None:
            init_position = init_joint_angles
        else:
            init_position = list_of_initialjoint_positions[i]
        diff_kin = CartesianDiffKin(dyn_model, controlled_frame_name, init_position, desired_cartesian_pos, np.zeros(3), desired_cartesian_ori, np.zeros(3), time_step, type_of_control, kp_pos, kp_ori, np.array(joint_vel_limits))
        steps = int(duration_per_desired_cartesian_pos/time_step)

        # reinitialize the robot to the initial position
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
            #current measured torque
            tau_mes = np.asarray(sim.GetMotorTorques(0),dtype=float)

            pd_d = [0.0, 0.0, 0.0]  # Desired linear velocity
            ori_d_des = [0.0, 0.0, 0.0]  # Desired angular velocity
            # Compute desired joint positions and velocities using Cartesian differential kinematics

            
            x = np.array(np.concatenate([q_mes, list_of_desired_cartesian_positions[i]])).reshape(1, -1)
            # y_pred = rf_model.predict(x)
            # #print('y_pred',t, y_pred)
            # q_des = np.array(y_pred[0,:7])
            # qd_des_clip = np.array(y_pred[0,7:14])
            # print('q_des_p', t, q_des, qd_des_clip)


            # mlp.eval()  # 切换到推理模式（关闭 dropout、BN 等）
            # x_tensor = torch.tensor(x, dtype=torch.float32)  # 转换为 Tensor
            # with torch.no_grad():
            #     y_pred = mlp(x_tensor)
            # q_des = np.array(y_pred[0,:7])
            # qd_des_clip = np.array(y_pred[0,7:14])
            # print('q_des_p',t, q_des, qd_des_clip)
            q_des, qd_des_clip = CartesianDiffKin(dyn_model,controlled_frame_name,q_mes, desired_cartesian_pos, pd_d, desired_cartesian_ori, ori_d_des, time_step, "pos",  kp_pos, kp_ori, np.array(joint_vel_limits))
            #print('q_des',q_des, qd_des_clip)
            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command


            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            
            # Conditional data recording
            if RECORDING:
                q_mes_all.append(q_mes)
                qd_mes_all.append(qd_mes)
                q_d_all.append(q_des)
                qd_d_all.append(qd_des_clip)
                tau_mes_all.append(tau_mes)
                cart_pos_all.append(cart_pos)
                cart_ori_all.append(cart_ori)
                angle_diff_all.append(q_des-q_mes)
                velocity_diff_all.append(qd_des_clip-qd_mes)
                tau_cmd_all.append(tau_cmd)

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step
            #print("Current time in seconds:", current_time)
    
        current_time = 0  # Reset current time for potential future use

        if len(q_mes_all) > 0:    
            print("Preparing to save data...")
            # Downsample data
            # Plot the downsampled data
            
            q_mes_all_downsampled = q_mes_all[::downsample_rate]
            qd_mes_all_downsampled = qd_mes_all[::downsample_rate]
            q_d_all_downsampled = q_d_all[::downsample_rate]
            qd_d_all_downsampled = qd_d_all[::downsample_rate]
            tau_mes_all_downsampled = tau_mes_all[::downsample_rate]
            cart_pos_all_downsampled = cart_pos_all[::downsample_rate]
            cart_ori_all_downsampled = cart_ori_all[::downsample_rate]
            angle_diff_all_downsampled = angle_diff_all[::downsample_rate]
            velocity_diff_all_downsampled = velocity_diff_all[::downsample_rate]
            tau_cmd_all_downsampled = tau_cmd_all[::downsample_rate]

            
            target_position = desired_cartesian_pos


            #q_mes_tar_all_downsampled = q_mes_all_downsampled.extend(desired_cartesian_pos)

            time_array = [time_step * downsample_rate * i for i in range(len(q_mes_all_downsampled))]

            # Save data to pickle file and for name use the current iteration number
            filename = FINAL_DIR / f"data_{i}.pkl"
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
                    'angle_diff_all': angle_diff_all_downsampled,
                    'velocity_diff_all': velocity_diff_all_downsampled,
                    'tau_cmd_all': tau_cmd_all_downsampled,
                    'target_position': target_position
                }, f)
            print(f"Data saved to {filename}")

            # Reinitialize data storage lists
        q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all = [], [], [], [], [], [], []
        angle_diff_all, velocity_diff_all,  tau_cmd_all = [], [], []

        if PRINT_PLOTS:
            print("Plotting downsampled data...")
            # Plot joint positions
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(q_mes_all_downsampled[0])):
                joint_positions = [q[joint_idx] for q in q_mes_all_downsampled]
                plt.plot(time_array, joint_positions, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Positions (rad)')
            plt.title('Downsampled Joint Positions')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot joint velocities
            plt.figure(figsize=(12, 6))
            for joint_idx in range(len(qd_mes_all_downsampled[0])):
                joint_velocities = [qd[joint_idx] for qd in qd_mes_all_downsampled]
                plt.plot(time_array, joint_velocities, label=f'Joint {joint_idx+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Velocities (rad/s)')
            plt.title('Downsampled Joint Velocities')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    
    

if __name__ == '__main__':
    main()
    # test rollout loader
    rls = load_rollouts(indices=[0,1,2,3], directory=FINAL_DIR)  # looks for ./data_1.pkl or ./1.pkl, up to 4
    print(f"Loaded {len(rls)} rollouts")
    print("First rollout keys lengths:",len(rls[0].time),len(rls[0].q_mes_all),len(rls[0].velocity_diff_all))