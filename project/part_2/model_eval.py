import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from final.rollout_loader import load_rollouts
from pathlib import Path
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

# ----- Load trained MLP model -----
class DeepMLP_Tunable(nn.Module):
    """Must match the architecture from training"""
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_hidden_sizes = [64, 128, 64]

mlp_model = DeepMLP_Tunable(input_dim=10, output_dim=14, hidden_sizes=best_hidden_sizes).to(device)
mlp_model.load_state_dict(torch.load('best_mlp_tuned.pth'))
mlp_model.eval()



norm_stats = np.load('normalization_stats.npz')
X_mean = norm_stats['X_mean']
X_std_safe = norm_stats['X_std_safe']
y_mean = norm_stats['y_mean']
y_std_safe = norm_stats['y_std_safe']
print("Loaded normalization stats from file")



FINAL_DIR = Path(__file__).resolve().parent
FINAL_DIR.mkdir(parents=True, exist_ok=True)

PRINT_PLOTS = True
RECORDING = True
downsample_rate = 2

def find_actual_start(traj, motion_threshold=0.002):
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    moving_indices = np.where(diffs > motion_threshold)[0]
    if len(moving_indices) > 0:
        return moving_indices[0]
    return 0 



def split_trajectory(traj, jump_threshold=0.05):
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    breaks = np.where(diffs > jump_threshold)[0] + 1  # points where reset happens
    segments = np.split(traj, breaks)
    segments = [seg for seg in segments if len(seg) > 2]
    return segments

def plot_segments(ax, traj, color, label):
    segments = split_trajectory(traj)
    for i, seg in enumerate(segments):
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],color=color, linewidth=2,label=label if i == 0 else None)  # only label first segment

def trim_after_reaching_target(traj, target, threshold=0.05):
    dists = np.linalg.norm(traj - target, axis=1)
    close_idx = np.where(dists < threshold)[0]
    if len(close_idx) > 0:
        stop_idx = close_idx[0] + 1
        return traj[:stop_idx]
    return traj

def plot_segments(ax, traj, color, label, target=None, stop_threshold=None):
    segments = split_trajectory(traj)
    for i, seg in enumerate(segments):
        if target is not None and stop_threshold is not None:
            seg = trim_after_reaching_target(seg, target, stop_threshold)
        if len(seg) > 1:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color=color, linewidth=2,
                    label=label if i == 0 else None)


def main():
    conf_file_name = "pandaconfig.json"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.join(parent_dir)
    
    print(f"Looking for config in: {root_dir}")
    
    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)

    source_names = ["pybullet"]

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    # Check joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    joint_vel_limits = sim.GetBotJointsVelLimit()
    
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    print(f"Joint vel limits: {joint_vel_limits}")
    
    # Command and control loop
    cmd = MotorCommands()

    # P controller high level
    kp_pos = 100
    kp_ori = 0

    # PD controller gains low level
    kp = 1000
    kd = 100

    # Test trajectories (completely new, not seen in training)
    list_of_desired_cartesian_positions = [
        [0.55, 0.20, 0.15], 
        [0.50, -0.25, 0.10], 
        [0.60, 0.0, 0.20],    
        [0.45, 0.25, 0.08],  
        [0.52, -0.15, 0.25], 
        [0.58, 0.18, 0.12] 
    ]

    list_of_desired_cartesian_orientations = [[0.0, 0.0, 0.0, 1.0]] * 6
    list_of_duration_per_desired_cartesian_positions = [5.0] * 6
    list_of_initialjoint_positions = [init_joint_angles] * 6

    time_step = sim.GetTimeStep()

    comparison_results = []

    for i in range(len(list_of_desired_cartesian_positions)):
        desired_cartesian_pos = np.array(list_of_desired_cartesian_positions[i])
        desired_cartesian_ori = np.array(list_of_desired_cartesian_orientations[i])
        duration = list_of_duration_per_desired_cartesian_positions[i]
        init_position = list_of_initialjoint_positions[i]

        steps = int(duration / time_step)

        # --- Run 1: Classic Controller ---
        print(f"\n{'='*60}")
        print(f"[Trajectory {i+1}/{len(list_of_desired_cartesian_positions)}] Classic Controller")
        print(f"Target: {desired_cartesian_pos}")
        print(f"{'='*60}")
        
        sim.ResetPose()
        sim.SetjointPosition(init_position)

        classic_trajectory = []
        for t in range(steps):
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)

            pd_d = np.zeros(3)
            ori_d_des = np.zeros(3)

            q_des, qd_des_clip = CartesianDiffKin(
                dyn_model, controlled_frame_name, q_mes, desired_cartesian_pos,
                pd_d, desired_cartesian_ori, ori_d_des, time_step,
                "pos", kp_pos, kp_ori, np.array(joint_vel_limits)
            )

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
            sim.Step(cmd, "torque")
            time.sleep(time_step)

            classic_trajectory.append(cart_pos.copy())

        classic_trajectory = np.array(classic_trajectory)
        classic_final_error = np.linalg.norm(classic_trajectory[-1] - desired_cartesian_pos)
        print(f"Classic final error: {classic_final_error:.6f} m")

        # --- Run 2: ML Model Controller ---
        print(f"\n{'='*60}")
        print(f"[Trajectory {i+1}/{len(list_of_desired_cartesian_positions)}] ML Model Controller")
        print(f"{'='*60}")
        
        sim.ResetPose()
        sim.SetjointPosition(init_position)

        ml_trajectory = []
        target_reached = False
        target_threshold = 0.06
        min_steps_before_check = int(1.0 / time_step)
        hold_steps = 0
        
        for t in range(steps):
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            cart_pos, cart_ori = dyn_model.ComputeFK(q_mes, controlled_frame_name)

            error_distance = np.linalg.norm(cart_pos - desired_cartesian_pos)
            
            if not target_reached and t > min_steps_before_check and error_distance < target_threshold:
                target_reached = True
                print(f"  Target reached at step {t}/{steps} (error: {error_distance:.6f} m)")
            
            if target_reached:
                hold_steps += 1
                q_des = q_mes
                qd_des_clip = np.zeros(7)  # Zero velocities to stop
            else:
                X_input = np.concatenate([q_mes, desired_cartesian_pos])
                X_input_norm = (X_input - X_mean) / X_std_safe
                X_tensor = torch.tensor(X_input_norm, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred_norm = mlp_model(X_tensor).cpu().numpy().flatten()
                
                # Denormalize predictions
                y_pred = y_pred_norm * y_std_safe + y_mean
                
                q_des = y_pred[:7]
                qd_des_clip = y_pred[7:14]

            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)
            sim.Step(cmd, "torque")
            time.sleep(time_step)

            ml_trajectory.append(cart_pos.copy())

        ml_trajectory = np.array(ml_trajectory)
        ml_final_error = np.linalg.norm(ml_trajectory[-1] - desired_cartesian_pos)
        print(f"ML model final error: {ml_final_error:.6f} m")

        # Store results
        comparison_results.append({
            'target': desired_cartesian_pos,
            'classic_trajectory': classic_trajectory,
            'ml_trajectory': ml_trajectory,
            'classic_error': classic_final_error,
            'ml_error': ml_final_error
        })

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        target = np.array(desired_cartesian_pos)

        # Plot the trajectories
        plot_segments(ax, classic_trajectory, 'b', 'Classic Controller',
                    target=target, stop_threshold=0.02)
        plot_segments(ax, ml_trajectory, 'g', 'ML Model',
                    target=target, stop_threshold=0.02)

        start_idx = find_actual_start(classic_trajectory)
        true_start = classic_trajectory[start_idx]

        ax.scatter(*target, c='r', s=200, marker='*', label='Target', zorder=10)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title("3D Trajectory Comparison (Trimmed at Target)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=25, azim=45)
        plt.tight_layout()
        plt.savefig('trajectory_trimmed.png', dpi=150, bbox_inches='tight')
        plt.show()

            
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    classic_errors = [r['classic_error'] for r in comparison_results]
    ml_errors = [r['ml_error'] for r in comparison_results]
    
    print(f"Classic Controller:")
    print(f"  Mean error: {np.mean(classic_errors):.6f} m")
    print(f"  Max error: {np.max(classic_errors):.6f} m")
    print(f"  Min error: {np.min(classic_errors):.6f} m")
    
    print(f"\nML Model Controller:")
    print(f"  Mean error: {np.mean(ml_errors):.6f} m")
    print(f"  Max error: {np.max(ml_errors):.6f} m")
    print(f"  Min error: {np.min(ml_errors):.6f} m")
    
    improvement = ((np.mean(classic_errors) - np.mean(ml_errors)) / np.mean(classic_errors)) * 100
    if improvement > 0:
        print(f"\nML model is {improvement:.1f}% better on average!")
    else:
        print(f"\nClassic controller is {-improvement:.1f}% better on average")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()