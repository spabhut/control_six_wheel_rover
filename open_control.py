import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RoverController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Real width matches the XML where y=0.25 and y=-0.25 (0.25 * 2 = 0.5)
        real_width = 0.50
        slip_factor = 2.2
        
        self.r = 0.1
        self.width = slip_factor * real_width
        
        # FIXED: Mapped exact actuator names from six_wheel_rover.xml
        self.fl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_motor")
        self.ml_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ml_motor")
        self.rl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rl_motor")
        
        self.fr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fr_motor")
        self.mr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mr_motor")
        self.rr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rr_motor")

    def move(self, v_linear, w_angular):
        left_vel  = (v_linear - (self.width / 2.0) * w_angular) / self.r
        right_vel = (v_linear + (self.width / 2.0) * w_angular) / self.r

        # Apply left velocities
        self.data.ctrl[self.fl_id] = left_vel
        self.data.ctrl[self.ml_id] = left_vel
        self.data.ctrl[self.rl_id] = left_vel
        
        # Apply right velocities
        self.data.ctrl[self.fr_id] = right_vel
        self.data.ctrl[self.mr_id] = right_vel
        self.data.ctrl[self.rr_id] = right_vel

def main():
    model = mujoco.MjModel.from_xml_path("six_wheel_rover.xml")
    data = mujoco.MjData(model)
    rover = RoverController(model, data)

    path_x = []
    path_y = []
    times = []

    print("Starting Six-Wheel Rover Simulation...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = data.time 
            
            path_x.append(data.qpos[0])
            path_y.append(data.qpos[1])
            times.append(t)
                
            if t < 2.0:
                v, w = 1.0, 0.0
            elif t < 3.0:
                v, w = 0.0, 1.57
            elif t < 5.0:
                v, w = 1.0, 0.0
            elif t < 6.0:
                v, w = 0.0, 1.57
            elif t < 8.0:
                v, w = 1.0, 0.0
            elif t < 9.0:
                v, w = 0.0, 1.57
            elif t < 11.0:
                v, w = 1.0, 0.0
            elif t < 12.0:
                v, w = 0.0, 1.57
            else:
                v, w = 0.0, 0.0
                if t > 12.0: break
            
            rover.move(v, w)
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time.sleep(model.opt.timestep)

    # Plotting the trajectory
    plt.figure(figsize=(10, 10))
    plt.plot(path_x, path_y, label='Actual Path', color='blue', linewidth=2)
    plt.scatter(path_x[0], path_y[0], color='green', label='Start', zorder=5, s=100)
    plt.scatter(path_x[-1], path_y[-1], color='black', label='End', zorder=5, s=100)

    plt.title("Six-Wheel Rover Trajectory")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    plt.savefig("rover_path.png")
    print("Plot saved as 'rover_path.png'")

if __name__ == "__main__":
    main()