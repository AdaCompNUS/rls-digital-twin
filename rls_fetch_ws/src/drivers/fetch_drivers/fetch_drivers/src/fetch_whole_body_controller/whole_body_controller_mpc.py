#!/usr/bin/env python3
import rospy
import casadi as ca
import numpy as np
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker
import tf2_ros
import tf.transformations as tf_trans
from std_msgs.msg import Bool


class WholeBodyMPC:
    def __init__(self, params):
        self.N = params['prediction_horizon']
        self.dt = 0.05
        
        # Dimensions
        self.n_arm_joints = 7
        self.n_torso_joints = 1
        self.n_base_pose = 3
        self.nx = self.n_base_pose + self.n_torso_joints + self.n_arm_joints
        self.nu = 2 + 1 + self.n_arm_joints  # base_vel(2) + torso_vel + arm_vels
        
        # Limits
        self.max_linear_vel = params['max_linear_vel']
        self.max_angular_vel = params['max_angular_vel']
        self.max_joint_vel = np.array(params['max_joint_velocities'])
        
        # State and control weights
        self.Q_state = np.array(params['Q_state'])
        self.R_control = np.array(params['R_control'])
        
        # Slack weights
        self.slack_dynamics_weight = params.get('slack_dynamics_weight', 1000.0)
        
        # Terminal cost weights
        self.P_state = np.array(params['P_state'])
        
        # Debug flag
        self.debug = params['debug']
        
        if self.debug:
            rospy.loginfo("Initialized MPC with weights:")
            rospy.loginfo(f"Q_state = {self.Q_state}")
            rospy.loginfo(f"R_control = {self.R_control}")
        
        # Set up the optimization problem
        self.setup_optimization()
        
    def setup_optimization(self):
        """Set up the MPC optimization problem with a more advanced tracking cost."""
        opti = ca.Opti()
        
        # Decision variables
        X = opti.variable(self.nx, self.N+1)
        U = opti.variable(self.nu, self.N)
        
        # Add slack variables for dynamics constraints
        slack_dynamics = opti.variable(self.nx, self.N)
        
        # Parameters for reference trajectory and initial state
        X_ref = opti.parameter(self.nx, self.N+1)
        X0 = opti.parameter(self.nx)
        
        # Cost function
        cost = 0
        for i in range(self.N):
            # Calculate the error vector in the world frame
            world_frame_error = X[:,i] - X_ref[:,i]
            ref_yaw = X_ref[2,i]

            # Create the symbolic rotation matrix to transform world-frame error
            # into the reference vehicle's body frame (Frenet frame).
            Rot = ca.vertcat(
                ca.horzcat(ca.cos(ref_yaw),  ca.sin(ref_yaw)),
                ca.horzcat(-ca.sin(ref_yaw), ca.cos(ref_yaw))
            )
            
            # Transform the position error [dx, dy] into the reference frame
            frenet_pos_error = ca.mtimes(Rot, world_frame_error[:2])
            
            # Extract the meaningful, decoupled errors
            cross_track_error = frenet_pos_error[1] # Error perpendicular to the desired direction

            # Yaw error cost using the (1 - cos(error)) metric for robustness
            yaw_error_cost = self.Q_state[2] * (1 - ca.cos(world_frame_error[2]))

            # Joint position error cost (remains the same)
            joint_error = world_frame_error[3:]
            joint_cost = self.Q_state[3:].reshape(1, self.nx-3) @ (joint_error * joint_error)

            # Combined weighted state cost using the Frenet-frame errors
            # Penalize cross-track error to stay on the path, not along-track error.
            frenet_pos_cost = self.Q_state[1] * cross_track_error**2
            
            state_cost = frenet_pos_cost + yaw_error_cost + joint_cost

            cost += state_cost
            
            # Control costs
            control_cost = self.R_control.reshape(1, self.nu) @ (U[:,i] * U[:,i])
            cost += control_cost
            
            # Penalties for slack variables
            cost += self.slack_dynamics_weight * ca.sumsqr(slack_dynamics[:,i])
        
        # Add a terminal cost to ensure the final state is reached accurately
        terminal_error = X[:, self.N] - X_ref[:, self.N]
        
        # Terminal position cost (world frame)
        terminal_pos_cost = self.P_state[0] * terminal_error[0]**2 + self.P_state[1] * terminal_error[1]**2

        # Terminal yaw cost (using 1-cos to handle wrapping)
        terminal_yaw_cost = self.P_state[2] * (1 - ca.cos(terminal_error[2]))
        
        # Terminal joint cost
        terminal_joint_error = terminal_error[3:]
        terminal_joint_cost = self.P_state[3:].reshape(1, self.nx-3) @ (terminal_joint_error * terminal_joint_error)

        cost += terminal_pos_cost + terminal_yaw_cost + terminal_joint_cost
        
        # Dynamics constraints with slack variables (unchanged)
        for i in range(self.N):
            x = X[:,i]
            u = U[:,i]
            dx = ca.vertcat(
                u[0] * ca.cos(x[2]),  # x
                u[0] * ca.sin(x[2]),  # y
                u[1],                 # theta
                u[2],                 # torso
                u[3:]                 # arm joints
            )
            opti.subject_to(X[:,i+1] == x + dx * self.dt + slack_dynamics[:,i])
        
        # Input constraints
        for i in range(self.N):
            opti.subject_to(opti.bounded(0, U[0,i], self.max_linear_vel))
            opti.subject_to(U[1,i] <= self.max_angular_vel)
            opti.subject_to(U[1,i] >= -self.max_angular_vel)
            for j in range(self.n_torso_joints + self.n_arm_joints):
                opti.subject_to(U[2+j,i] <= self.max_joint_vel[j])
                opti.subject_to(U[2+j,i] >= -self.max_joint_vel[j])
        
        # Non-negativity constraints for slack variables (unchanged)
        opti.subject_to(opti.bounded(0, slack_dynamics, 1.0))
        
        # Initial state constraint (no slack) (unchanged)
        opti.subject_to(X[:,0] == X0)
        
        # Set the objective (unchanged)
        opti.minimize(cost)
        
        # Store the optimization problem and variables/parameters (unchanged)
        self.opti = opti
        self.cost = cost
        self.X = X
        self.U = U
        self.X_ref = X_ref
        self.X0 = X0
        self.slack_dynamics = slack_dynamics
    
    def solve(self, x0, x_ref):
        """Solve the MPC problem"""
        # Set parameter values
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.X_ref, x_ref)
        
        # Set solver options
        opts = {
            'ipopt.print_level': 0 if not self.debug else 3,
            'print_time': 0 if not self.debug else 1,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_obj_change_tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes'
        }
        
        # Create and call the solver
        self.opti.solver('ipopt', opts)
        sol = self.opti.solve()
        
        # Extract solution
        X_opt = sol.value(self.X)
        U_opt = sol.value(self.U)
        
        # Debug information
        if self.debug:
            slack_dynamics = sol.value(self.slack_dynamics)
            
            if np.any(slack_dynamics > 1e-6):
                rospy.logwarn(f"Dynamics slack active, max: {np.max(slack_dynamics)}")
        
        return X_opt, U_opt


class WholeBodyController:
    def __init__(self):
        rospy.init_node('fetch_whole_body_controller')
        
        # State tracking
        self.joint_states = None
        self.arm_traj = []
        self.base_traj = []
        self.merged_traj = []
        self.executing = False
        self.last_waypoint_idx = 0
        
        # Create TF buffer with longer cache time for better performance
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Load parameters
        self.params = self.load_parameters()
        self.mpc = WholeBodyMPC(self.params)
        
        # Frame names
        self.map_frame = self.params.get('map_frame', 'map')
        self.base_frame = self.params.get('base_frame', 'base_link')
        
        # ROS interfaces
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.joint_vel_pub = rospy.Publisher(
            '/arm_with_torso_controller/joint_velocity_controller/command', JointState, queue_size=1)
        self.execution_finished_pub = rospy.Publisher('/fetch_whole_body_controller/execution_finished', Bool, queue_size=1)
        
        # Debug publishers for paths
        if self.params['debug']:
            self.ref_path_pub = rospy.Publisher('/ref_path_viz', Marker, queue_size=1)
            self.mpc_path_pub = rospy.Publisher('/mpc_path_viz', Marker, queue_size=1)
            self.global_path_pub = rospy.Publisher('/global_path_viz', Marker, queue_size=1)
        
        # Subscribe to sensor topics
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        
        # Subscribe to trajectory topics
        rospy.Subscriber('/fetch_whole_body_controller/arm_path', JointTrajectory, self.arm_cb)
        rospy.Subscriber('/fetch_whole_body_controller/base_path', JointTrajectory, self.base_cb)
        rospy.Subscriber('/fetch_whole_body_controller/stop', Bool, self.stop_cb)
        
        rospy.loginfo("Waiting for the 'map' to 'base_link' transform...")
        try:
            # This will block until the transform is available, with a timeout.
            self.tf_buffer.can_transform(self.map_frame, self.base_frame, rospy.Time(0), timeout=rospy.Duration(15.0))
            rospy.loginfo("Transform is now available.")
        except tf2_ros.TransformException as e:
            rospy.logerr(f"Could not get transform after 15 seconds. Is a localization node (e.g., amcl) running? Error: {e}")
            # Depending on your needs, you might want to shut down or raise an exception
            rospy.signal_shutdown("TF transform not available")
            return
        
        # Create timers for execution and obstacle processing
        self.execution_timer = None
        
        rospy.loginfo("Whole Body Controller initialized")

    def load_parameters(self):
        return {
            'control_rate': rospy.get_param('~control_rate', 10.0),
            'prediction_horizon': rospy.get_param('~prediction_horizon', 10),
            'max_linear_vel': rospy.get_param('~max_linear_vel', 1.5),
            'max_angular_vel': rospy.get_param('~max_angular_vel', 1.0),
            'max_joint_velocities': rospy.get_param('~max_joint_velocities', 
                [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            'Q_state': rospy.get_param('~Q_state', 
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            'R_control': rospy.get_param('~R_control', 
                [2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            'slack_dynamics_weight': rospy.get_param('~slack_dynamics_weight', 1000.0),
            'base_pos_threshold': rospy.get_param('~base_pos_threshold', 0.05),
            'min_waypoint_distance': rospy.get_param('~min_waypoint_distance', 0.05),
            'trajectory_end_threshold': rospy.get_param('~trajectory_end_threshold', 0.05),
            'P_state': rospy.get_param('~P_state', [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
            'debug': rospy.get_param('~debug', True),
            'map_frame': rospy.get_param('~map_frame', 'map'),
            'base_frame': rospy.get_param('~base_frame', 'base_link')
        }

    def get_base_pose_from_tf(self):
        """Get robot pose from TF"""
        transform = self.tf_buffer.lookup_transform(
            self.map_frame,
            self.base_frame,
            rospy.Time(0),
            rospy.Duration(0.1)
        )
        
        pos = transform.transform.translation
        q = transform.transform.rotation
        _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        return [pos.x, pos.y, yaw]

    def joint_cb(self, msg):
        # Only process messages with length > 2 to filter out gripper messages
        if len(msg.name) > 2:
            self.joint_states = dict(zip(msg.name, msg.position))

    def arm_cb(self, msg):
        self.arm_traj = {
            'points': [p.positions for p in msg.points]
        }
        self.check_trajectories()

    def base_cb(self, msg):
        self.base_traj = {
            'points': [p.positions for p in msg.points]
        }
        self.check_trajectories()

    def stop_cb(self, msg):
        if msg.data and self.executing:
            rospy.loginfo("Received stop command, stopping trajectory execution.")
            self.stop_execution(success=False)

    def check_trajectories(self):
        if not self.arm_traj or not self.base_traj:
            return
            
        if len(self.arm_traj['points']) != len(self.base_traj['points']):
            rospy.logerr("Trajectory mismatch: different number of points!")
            return
        
        # Do not start if trajectories are empty
        if not self.arm_traj['points']:
            rospy.logwarn("Received empty trajectories, not starting execution.")
            return
        
        # Create merged trajectory
        self.merged_traj = [
            np.concatenate([base, arm])
            for base, arm in zip(self.base_traj['points'], self.arm_traj['points'])
        ]
        
        if self.params['debug']:
            # Visualize global path (blue)
            self.publish_path_as_marker(self.merged_traj, self.global_path_pub, 0.0, 0.0, 1.0, 2)
        
        # Pre-compute distances between consecutive waypoints for the base
        self.base_waypoint_distances = []
        for i in range(len(self.base_traj['points'])-1):
            p1 = np.array(self.base_traj['points'][i][:2])  # x, y of waypoint i
            p2 = np.array(self.base_traj['points'][i+1][:2])  # x, y of waypoint i+1
            self.base_waypoint_distances.append(np.linalg.norm(p2 - p1))
        
        self.start_trajectory()

    def start_trajectory(self):
        if self.executing:
            return
            
        self.last_waypoint_idx = 0
        self.executing = True
        # Create a timer that calls the trajectory execution callback at the control rate
        self.execution_timer = rospy.Timer(
            rospy.Duration(1.0 / self.params['control_rate']),
            self.trajectory_execution_cb
        )
        rospy.loginfo("Starting trajectory execution")

    def get_current_state(self):
        if self.joint_states is None:
            return None
            
        # Get base pose from TF
        base_pose = self.get_base_pose_from_tf()
        if base_pose is None:
            return None
            
        joints = [
            self.joint_states.get('torso_lift_joint', 0),
            *[self.joint_states.get(f, 0) for f in [
                'shoulder_pan_joint', 'shoulder_lift_joint',
                'upperarm_roll_joint', 'elbow_flex_joint',
                'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'
            ]]
        ]
        return np.concatenate([base_pose, joints])

    def find_nearest_waypoint(self, current_state):
        """Find the index of the waypoint closest to the current base position and orientation"""
        current_base_pos = current_state[:2]  # x, y coordinates
        current_orientation = current_state[2]  # theta
        
        min_dist = float('inf')
        nearest_idx = self.last_waypoint_idx
        
        # Search from the last waypoint index to the end of the trajectory
        for i in range(self.last_waypoint_idx, len(self.base_traj['points'])):
            waypoint = self.base_traj['points'][i]
            wp_pos = np.array(waypoint[:2])
            wp_orientation = waypoint[2]
            
            # Position distance
            pos_dist = np.linalg.norm(wp_pos - current_base_pos)
            
            # Orientation distance (handle angle wrapping)
            angle_diff = abs(current_orientation - wp_orientation)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            # Combined distance (weighted sum)
            combined_dist = pos_dist + 0.5 * angle_diff
            
            if combined_dist < min_dist:
                min_dist = combined_dist
                nearest_idx = i
        
        return nearest_idx

    def construct_reference_from_waypoint(self, start_idx):
        """Construct reference trajectory starting from the given waypoint index"""
        ref = np.zeros((self.mpc.nx, self.mpc.N+1))
        start_idx += 10
        for i in range(self.mpc.N+1):
            idx = min(start_idx + i, len(self.merged_traj) - 1)
            ref[:,i] = self.merged_traj[idx]
            
        return ref

    def publish_path_as_marker(self, path, publisher, r, g, b, marker_id):
        """Publish a path as a LINE_STRIP marker."""
        if not self.params['debug']:
            return

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "path_viz"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Line width

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0.5)

        # Path is expected to be a list of states (e.g., from X.T)
        for state in path:
            p = Point()
            p.x = state[0]
            p.y = state[1]
            p.z = 0.1  # slightly above ground
            marker.points.append(p)
        
        publisher.publish(marker)

    def is_trajectory_complete(self, current_state, end_state):
        """Check if we've reached the end of the trajectory"""
        # Check base position (x, y)
        base_pos_diff = np.linalg.norm(current_state[:2] - end_state[:2])
        
        # Check orientation separately (handle angle wrapping)
        angle_diff = abs(current_state[2] - end_state[2])
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        # Check arm joints
        arm_diff = np.linalg.norm(current_state[3:] - end_state[3:])
        
        return (base_pos_diff < self.params['trajectory_end_threshold'] and 
                angle_diff < 0.1 and 
                arm_diff < 0.2)
                
    def trajectory_execution_cb(self, event):
        """Execute the trajectory with CBF-based obstacle avoidance"""
        if not self.merged_traj:
            rospy.logwarn_once("Trajectory execution callback called without a valid trajectory. Stopping.")
            self.stop_execution(success=False)
            return
            
        # Get current state
        current_state = self.get_current_state()
        if current_state is None:
            return
        
        # Find nearest waypoint
        nearest_idx = self.find_nearest_waypoint(current_state)
        self.last_waypoint_idx = nearest_idx
        
        # Generate reference trajectory based on nearest waypoint
        ref = self.construct_reference_from_waypoint(nearest_idx)
        
        if self.params['debug']:
            rospy.logdebug(f"Nearest waypoint index: {nearest_idx} of {len(self.merged_traj)}")
            # Visualize reference path (red)
            self.publish_path_as_marker(ref.T, self.ref_path_pub, 1.0, 0.0, 0.0, 0)
            start_time = rospy.Time.now()
            
        # Solve MPC
        X, U = self.mpc.solve(current_state, ref)
        
        if self.params['debug']:
            solve_time = (rospy.Time.now() - start_time).to_sec()
            rospy.logdebug(f"MPC solve time: {solve_time:.4f} seconds")
            if X is not None:
                # Visualize MPC planned path (green)
                self.publish_path_as_marker(X.T, self.mpc_path_pub, 0.0, 1.0, 0.0, 1)
        
        if U is None:
            rospy.logerr("MPC solve failed. Stopping execution.")
            self.stop_execution(success=False)
            return
            
        # Publish commands
        self.publish_commands(U[:,0])
        
        # Check if we've reached the end of the trajectory
        if nearest_idx >= len(self.merged_traj) - 1:
            # Near the final waypoint, check if we're close enough
            if self.is_trajectory_complete(current_state, self.merged_traj[-1]):
                rospy.loginfo("Trajectory completed successfully")
                self.stop_execution(success=True)
                return
    
    def stop_execution(self, success=True):
        """Stop trajectory execution and cleanup timer"""
        if not self.executing:
            return

        self.stop()
        self.executing = False
        
        # Shutdown the timer if it exists
        if self.execution_timer is not None:
            self.execution_timer.shutdown()
            self.execution_timer = None

        # Publish completion signal
        self.execution_finished_pub.publish(Bool(data=success))
        if success:
            rospy.loginfo("Published trajectory completion signal (success).")
        else:
            rospy.loginfo("Published trajectory completion signal (aborted).")

    def publish_commands(self, u):
        """Publish velocity commands to the robot"""
        # Base command
        twist = Twist()
        twist.linear.x = float(u[0])
        twist.angular.z = float(u[1])
        self.cmd_vel_pub.publish(twist)
        
        # Joint command
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = ['torso_lift_joint'] + [
            'shoulder_pan_joint', 'shoulder_lift_joint',
            'upperarm_roll_joint', 'elbow_flex_joint',
            'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'
        ]
        # Convert control input array to list of velocities
        u_np = np.array(u).flatten()
        js.velocity = [float(v) for v in u_np[2:]]
        self.joint_vel_pub.publish(js)

    def stop(self):
        """Send zero velocity commands to stop the robot"""
        zero_control = np.zeros(self.mpc.nu)
        self.publish_commands(zero_control)


if __name__ == '__main__':
    try:
        controller = WholeBodyController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
