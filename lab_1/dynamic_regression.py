import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []
    tau_mes_link7_all = []
    regressor_link7_all = []
    regressor_1to6_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)
        #print("torque", tau_mes.shape)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        regressor = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes,qdd_mes)
        #print("regessor",regressor.shape)
        regressor_for_1to6 = regressor[:, :60]#(7,60)
        regressor_for_7 = regressor[:, 60:] #(7,10)
        regressor_all.append(regressor)
        regressor_1to6_all.append(regressor_for_1to6)
        regressor_link7_all.append(regressor_for_7)
        
        tau_mes_all.append(tau_mes)
        tau_mes_link7_all.append(tau_mes[6])
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
        
        # if len(regressor_all)==0:
        #     regressor_all = regressor
        #     regressor_link7_all = regressor_for_7
        #     regressor_1to6_all = regressor_for_1to6
        #     print("regressor_link7",regressor_link7_all.shape)
        # else:
        #     regressor_all=np.vstack((regressor_all,regressor))
        #     regressor_link7_all=np.vstack((regressor_link7_all,regressor_for_7))
        #     regressor_1to6_all=np.vstack((regressor_1to6_all,regressor_for_1to6))
        #     print("regressor_link7",regressor_link7_all.shape)
        # if len(tau_mes_all)==0:
        #     tau_mes_all = tau_mes.reshape(7,1)
        #     #tau_mes_link7_all = tau_mes[6].reshape(1,1)
        #     print("tau_mes_all-0",tau_mes_all.shape)
        # else:
        #     tau_mes_all=np.vstack((tau_mes_all,tau_mes.reshape(7,1)))
        #     #tau_mes_link7_all=np.vstack((tau_mes_link7_all,tau_mes[6].reshape(1,1)))
    # give up data in the first second for regression
    regressor_stack = np.vstack(regressor_all[1000:])
    regressor_1to6_stack  = np.vstack(regressor_1to6_all[1000:])
    regressor_link7_stack = np.vstack(regressor_link7_all[1000:])
    tau_mes_stack = np.array(tau_mes_all[1000:]).reshape(-1,1)
    tau_mes_link7_all = np.array(tau_mes_link7_all).reshape(-1,1)
    # remain all data for evaluation
    regressor_all = np.vstack(regressor_all)
    regressor_1to6_all  = np.vstack(regressor_1to6_all)
    regressor_link7_all = np.vstack(regressor_link7_all)
    tau_mes_all = np.array(tau_mes_all).reshape(-1,1)
    print("regressor_stack",regressor_stack.shape)
    print("tau_mes_stack",tau_mes_stack.shape)
    print("regressor_link7_stack",regressor_link7_stack.shape)
    print("regressor_1to6_stack ",regressor_1to6_stack.shape)
    print("tau_mes_link7_all",tau_mes_link7_all.shape)

    #a_pred = np.linalg.pinv(regressor_stack)@tau_mes_stack
    a_pred = np.linalg.pinv(regressor_all)@tau_mes_all
    print("pseudoinverse calculated a", a_pred.shape,a_pred.reshape(7,10))
    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    params = []
    for i in range(num_joints):
        params.append(np.asarray(dyn_model.pin_model.inertias[i+1].toDynamicParameters(), dtype=float).flatten())
    params = np.asarray(params).flatten()
    print("a true",params)
    a_params_1to6 = params[:10 * (num_joints - 1)].reshape(60,1)
    #τ=Y1−6​a1−6​+Y7​a7​
    a_link7 = np.linalg.pinv(regressor_link7_stack) @ (tau_mes_stack - (regressor_1to6_stack  @ a_params_1to6))
    #a_link7 = np.linalg.pinv(regressor_link7_all) @ (tau_mes_all - (regressor_1to6_all  @ a_params_1to6))
    #a_link7 = np.linalg.pinv(regressor_link7_all)@tau_mes_link7_all
    print("pseudoinverse calculated a_link7", a_link7.shape, a_link7)
    
    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file     
    tau_mes_link7_pred = (regressor_1to6_all @ a_params_1to6) + (regressor_link7_all@a_link7)
    tau_mes_link7_pred = tau_mes_link7_pred[6::7]
    print("tau_mes_link7_pred",tau_mes_link7_pred.shape,tau_mes_link7_pred)
    # print("tau_mes",np.array(tau_mes_all).shape,tau_mes_all)
    tau_mes_all_pred = regressor_all@a_pred
    # Extract actual dynamic a parameters from the urdf file
    

    # save data
    np.savez('./week_1_2/robot_data.npz', Y_link7=regressor_link7_all, Y=regressor_all, 
            a_link7=a_link7,a_pred=a_pred, a_true=params,
            u_link7=tau_mes_link7_all,u_link7_pred=tau_mes_link7_pred,u=tau_mes_all,u_pred=tau_mes_all_pred)
    # TODO plot the torque prediction error for each joint (optional)
    

if __name__ == '__main__':
    main()
