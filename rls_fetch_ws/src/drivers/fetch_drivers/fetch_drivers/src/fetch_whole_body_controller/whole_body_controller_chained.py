#!/usr/bin/env python3
import rospy
import numpy as np
import math
import tf.transformations as tf_trans
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
import tf2_ros

def generate_timed_trajectory(path, max_vel, max_acc):
    if not path or len(path) < 2:
        pose_dim, vel_dim = len(max_vel) + 1, len(max_vel)
        pose = path[0] if path else np.zeros(pose_dim)
        return [{'time': 0.0, 'pose': pose, 'velocity': np.zeros(vel_dim)}]

    safe_max_acc = np.maximum(max_acc, 1e-6)
    safe_max_vel = np.maximum(max_vel, 1e-6)

    segment_durations = []
    for i in range(len(path) - 1):
        p_start, p_end = path[i], path[i+1]
        delta_dist_xy = np.linalg.norm(p_end[:2] - p_start[:2])
        delta_rot_joints = np.abs(p_end[2:] - p_start[2:])
        deltas = np.concatenate(([delta_dist_xy], delta_rot_joints))
        times_per_dim = np.zeros_like(deltas)
        for j in range(len(deltas)):
            delta_d, v_max, a_max = deltas[j], safe_max_vel[j], safe_max_acc[j]
            t_accel = v_max / a_max
            d_accel = 0.5 * a_max * t_accel**2
            if delta_d > 2 * d_accel:
                t_coast = (delta_d - 2 * d_accel) / v_max
                times_per_dim[j] = 2 * t_accel + t_coast
            else:
                times_per_dim[j] = 2 * np.sqrt(delta_d / a_max)
        segment_duration = np.max(times_per_dim) if times_per_dim.size > 0 else 1e-3
        segment_durations.append(max(segment_duration, 1e-3))

    times = [0.0]
    for d in segment_durations:
        times.append(times[-1] + d)

    trajectory = []
    for i, (t, pose) in enumerate(zip(times, path)):
        velocity = np.zeros(len(max_vel))
        if i < len(path) - 1:
            p_start, p_end = path[i], path[i+1]
            duration = segment_durations[i]
            velocity[0] = np.linalg.norm(p_end[:2] - p_start[:2]) / duration
            velocity[1:] = (p_end[2:] - p_start[2:]) / duration
        trajectory.append({'time': t, 'pose': pose, 'velocity': velocity})
    return trajectory

class ChainedFormController:
    def __init__(self, params):
        self.k1, self.k2, self.k3 = params['k1'], params['k2'], params['k3']

    def wrap_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def compute_control(self, current_state, ref_pose, ref_vel):
        x_r, y_r, theta_r = ref_pose
        v_r, omega_r = ref_vel
        x, y, theta = current_state

        dx, dy = x - x_r, y - y_r
        cos_theta_r, sin_theta_r = math.cos(theta_r), math.sin(theta_r)
        x_e = dx * cos_theta_r + dy * sin_theta_r
        y_e = -dx * sin_theta_r + dy * cos_theta_r
        theta_e = self.wrap_angle(theta - theta_r)

        if abs(theta_e) > 1.4:
            return [0.0, -2.0 * theta_e]
            
        z1, z2, z3 = x_e, y_e, math.tan(theta_e)
        v_r_active = v_r if abs(v_r) > 1e-2 else 1e-2
        
        w1 = -self.k1 * abs(v_r_active) * (z1 + z2 * z3)
        w2 = -self.k2 * v_r_active * z2 - self.k3 * abs(v_r_active) * z3

        cos_theta_e = math.cos(theta_e)
        u1 = (w1 + v_r) / cos_theta_e
        u2 = w2 * (cos_theta_e**2) + omega_r
        
        return [max(0.0, u1), u2]

class WholeBodyController:
    def __init__(self):
        rospy.init_node('fetch_chained_form_controller')
        self.joint_states, self.arm_traj, self.base_traj = None, None, None
        self.executing, self.start_time = False, None
        self.timed_trajectory = []
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.params = self.load_parameters()
        self.controller = ChainedFormController(self.params)
        self.max_vel_vec = np.concatenate(([self.params['max_linear_vel']], [self.params['max_angular_vel']], self.params['max_joint_velocities']))
        self.max_acc_vec = np.concatenate(([self.params['max_linear_acc']], [self.params['max_angular_acc']], self.params['max_joint_accelerations']))
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.joint_vel_pub = rospy.Publisher('/arm_with_torso_controller/joint_velocity_controller/command', JointState, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        rospy.Subscriber('/fetch_whole_body_controller/arm_path', JointTrajectory, self.arm_cb)
        rospy.Subscriber('/fetch_whole_body_controller/base_path', JointTrajectory, self.base_cb)
        self.execution_timer = None
        rospy.loginfo("Chained Form Controller Initialized.")

    def load_parameters(self):
        param_names = ['control_rate', 'max_linear_vel', 'max_angular_vel', 'max_joint_velocities', 
                       'max_linear_acc', 'max_angular_acc', 'max_joint_accelerations', 
                       'k1', 'k2', 'k3', 'position_tolerance', 'angle_tolerance']
        params = {p: rospy.get_param(f'~{p}') for p in param_names}
        return params

    def joint_cb(self, msg):
        if len(msg.name) > 8: self.joint_states = dict(zip(msg.name, msg.position))
    def arm_cb(self, msg): self.arm_traj = msg; self.check_trajectories()
    def base_cb(self, msg): self.base_traj = msg; self.check_trajectories()

    def check_trajectories(self):
        if self.arm_traj is None or self.base_traj is None: return
        merged_path = [np.concatenate([b.positions, a.positions]) for b, a in zip(self.base_traj.points, self.arm_traj.points)]
        self.timed_trajectory = generate_timed_trajectory(merged_path, self.max_vel_vec, self.max_acc_vec)
        rospy.loginfo(f"Generated synced trajectory. Duration: {self.timed_trajectory[-1]['time']:.2f}s.")
        self.arm_traj = None; self.base_traj = None
        self.start_trajectory()

    def start_trajectory(self):
        if self.executing: return
        self.executing, self.start_time = True, rospy.Time.now()
        self.execution_timer = rospy.Timer(rospy.Duration(1.0 / self.params['control_rate']), self.trajectory_execution_cb)

    def get_current_state(self):
        t = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0))
        pos, q = t.transform.translation, t.transform.rotation
        _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
        j_names = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        joint_state = [self.joint_states.get(name, 0) for name in j_names]
        return [pos.x, pos.y, yaw], joint_state

    def trajectory_execution_cb(self, event):
        # 1. GET CURRENT STATE
        if self.joint_states is None: return
        current_base_state, _ = self.get_current_state()
        if current_base_state is None: return
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()

        # 2. CHECK FOR TRAJECTORY COMPLETION
        final_goal_pose = self.timed_trajectory[-1]['pose']
        pos_error = np.linalg.norm(np.array(current_base_state[:2]) - final_goal_pose[:2])
        orient_error = abs(self.controller.wrap_angle(current_base_state[2] - final_goal_pose[2]))
        
        if (elapsed_time > self.timed_trajectory[-1]['time'] - 0.2 and 
            pos_error < self.params['position_tolerance'] and 
            orient_error < self.params['angle_tolerance']):
            rospy.loginfo("Goal reached within tolerance.")
            self.stop_execution()
            return
        if elapsed_time > self.timed_trajectory[-1]['time'] + 2.0:
            rospy.logwarn("Trajectory time has elapsed. Forcing stop.")
            self.stop_execution()
            return

        # 3. INTERPOLATE REFERENCE TRAJECTORY (The new, clearer way)
        ref_times = [p['time'] for p in self.timed_trajectory]
        ref_x_coords = [p['pose'][0] for p in self.timed_trajectory]
        ref_y_coords = [p['pose'][1] for p in self.timed_trajectory]
        ref_thetas_unwrapped = np.unwrap([p['pose'][2] for p in self.timed_trajectory])
        ref_linear_vels = [p['velocity'][0] for p in self.timed_trajectory]
        ref_angular_vels = [p['velocity'][1] for p in self.timed_trajectory]
        ref_arm_vels = np.array([p['velocity'][2:] for p in self.timed_trajectory])

        interp_x = np.interp(elapsed_time, ref_times, ref_x_coords)
        interp_y = np.interp(elapsed_time, ref_times, ref_y_coords)
        interp_theta = self.controller.wrap_angle(np.interp(elapsed_time, ref_times, ref_thetas_unwrapped))
        ref_pose = [interp_x, interp_y, interp_theta]
        
        interp_v = np.interp(elapsed_time, ref_times, ref_linear_vels)
        interp_omega = np.interp(elapsed_time, ref_times, ref_angular_vels)
        ref_vel = [interp_v, interp_omega]

        # 4. COMPUTE AND PUBLISH CONTROL COMMANDS
        base_command = self.controller.compute_control(current_base_state, ref_pose, ref_vel)
        
        arm_vel_command = np.zeros(8)
        for j in range(8):
            arm_vel_command[j] = np.interp(elapsed_time, ref_times, ref_arm_vels[:, j])

        self.publish_commands(base_command, arm_vel_command)

    def stop_execution(self):
        if not self.executing: return
        self.publish_commands([0.0, 0.0], np.zeros(8))
        self.executing = False
        if self.execution_timer: self.execution_timer.shutdown(); self.execution_timer = None
        rospy.loginfo("Trajectory execution finished.")

    def publish_commands(self, base_cmd, arm_cmd):
        twist = Twist()
        twist.linear.x, twist.angular.z = base_cmd[0], base_cmd[1]
        self.cmd_vel_pub.publish(twist)
        js = JointState(); js.header.stamp = rospy.Time.now()
        js.name = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        js.velocity = list(arm_cmd)
        self.joint_vel_pub.publish(js)

if __name__ == '__main__':
    wbc = WholeBodyController()
    rospy.spin()