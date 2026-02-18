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
        
        real_width = 0.55
        slip_factor = 8.897475
        
        self.r = 0.1
        self.width = slip_factor * real_width
        
        self.lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lf_motor")
        self.lm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lm_motor")
        self.lr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lr_motor")
        
        self.rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rf_motor")
        self.rm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rm_motor")
        self.rr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rr_motor")

    def move(self, v_linear, w_angular):
        left_vel  = (v_linear - (self.width / 2.0) * w_angular) / self.r
        right_vel = (v_linear + (self.width / 2.0) * w_angular) / self.r

        self.data.ctrl[self.lf_id] = left_vel
        self.data.ctrl[self.lm_id] = left_vel
        self.data.ctrl[self.lr_id] = left_vel
        
        self.data.ctrl[self.rf_id] = right_vel
        self.data.ctrl[self.rm_id] = right_vel
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
            elif t < 2.5:
                v, w = 0.0, 0.0
            elif t < 3.5:
                v, w = 0.0, 1.57
            elif t < 4.0:
                v, w = 0.0, 0.0
                
            elif t < 6.0:
                v, w = 1.0, 0.0
            elif t < 6.5:
                v, w = 0.0, 0.0
            elif t < 7.5:
                v, w = 0.0, 1.57
            elif t < 8.0:
                v, w = 0.0, 0.0
                
            elif t < 10.0:
                v, w = 1.0, 0.0
            elif t < 10.5:
                v, w = 0.0, 0.0
            elif t < 11.5:
                v, w = 0.0, 1.57
            elif t < 12.0:
                v, w = 0.0, 0.0
                
            elif t < 14.0:
                v, w = 1.0, 0.0
            elif t < 14.5:
                v, w = 0.0, 0.0
            elif t < 15.5:
                v, w = 0.0, 1.57
                
            else:
                v, w = 0.0, 0.0
                if t > 16: break 
            
            rover.move(v, w)
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time.sleep(model.opt.timestep)

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