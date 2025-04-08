#!/usr/bin/env python3
import rospy
import casadi as ca
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PointStamped
from sensor_msgs.msg import JointState, LaserScan
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tf_trans
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import struct


class WholeBodyMPC:
    def __init__(self, params):
        self.N = params['prediction_horizon']
        self.dt = 1.0 / params['control_rate']
        
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
        
        # Acceleration limits
        self.max_linear_acc = params.get('max_linear_acc', 0.5)
        self.max_angular_acc = params.get('max_angular_acc', 0.8)
        self.max_joint_acc = np.array(params.get('max_joint_accelerations', [0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        
        # State and control weights
        if isinstance(params['Q_state'], list):
            if len(params['Q_state']) == self.nx:
                self.Q_state = np.array(params['Q_state'])
            else:
                rospy.logwarn(f"Q_state in config has wrong dimension: expected {self.nx}, got {len(params['Q_state'])}. Using default.")
                self.Q_state = 10.0 * np.ones(self.nx)
        else:
            self.Q_state = params['Q_state'] * np.ones(self.nx)
        
        if isinstance(params['R_control'], list):
            if len(params['R_control']) == self.nu:
                self.R_control = np.array(params['R_control'])
            else:
                rospy.logwarn(f"R_control in config has wrong dimension: expected {self.nu}, got {len(params['R_control'])}. Using default.")
                self.R_control = 1.0 * np.ones(self.nu)
        else:
            self.R_control = params['R_control'] * np.ones(self.nu)
        
        # Slack weights
        self.slack_dynamics_weight = params.get('slack_dynamics_weight', 1000.0)
        self.slack_cbf_weight = params.get('slack_obstacle_weight', 200.0)
        
        # Obstacle avoidance parameters
        self.safe_distance = params.get('safe_distance', 0.3)
        self.voxel_size = params.get('voxel_size', 0.2)
        self.max_obstacle_points = params.get('max_obstacle_points', 10)
        self.lidar_max_range = params.get('lidar_max_range', 1.0)
        
        # CBF parameters
        self.gamma_k = params.get('gamma_k', 0.1)  # CBF decay rate
        self.M_CBF = min(3, self.N)  # Number of steps to apply CBF (using full horizon)
        
        self.acc_horizon = min(2, self.N)
        
        # Debug flag
        self.debug = params['debug']
        
        if self.debug:
            rospy.loginfo("Initialized MPC with weights:")
            rospy.loginfo(f"Q_state = {self.Q_state}")
            rospy.loginfo(f"R_control = {self.R_control}")
            rospy.loginfo(f"Obstacle params: safe_distance={self.safe_distance}, gamma_k={self.gamma_k}")
            rospy.loginfo(f"Acceleration limits: linear={self.max_linear_acc}, angular={self.max_angular_acc}")
        
        # Initialize obstacle data
        self.obstacle_positions = []
        
        # Set up the optimization problem
        self.setup_optimization()
        
    def setup_optimization(self):
        """Set up the MPC optimization problem with parameterized obstacle constraints"""
        opti = ca.Opti()
        
        # Decision variables
        X = opti.variable(self.nx, self.N+1)
        U = opti.variable(self.nu, self.N)
        
        # Add slack variables for dynamics constraints
        slack_dynamics = opti.variable(self.nx, self.N)
        
        # Parameters for reference trajectory and initial state
        X_ref = opti.parameter(self.nx, self.N+1)
        X0 = opti.parameter(self.nx)
        
        # Parameter for previous control input (for acceleration constraints)
        U_prev = opti.parameter(self.nu)
        
        # Parameters for obstacles: for each obstacle point, store x,y for each time step in horizon
        # We'll limit to max_obstacle_points obstacles, each with N+1 positions (full horizon)
        obstacle_params = opti.parameter(2, self.max_obstacle_points * (self.N+1))
        
        # New parameter: actual number of obstacle points (represented as a binary mask)
        obstacle_mask = opti.parameter(self.max_obstacle_points)  # Binary mask for active obstacles
        
        # Add slack variables for CBF constraints
        cbf_slack = opti.variable(self.max_obstacle_points, self.M_CBF)
        opti.subject_to(opti.bounded(0, cbf_slack, 1.0))
        
        # Cost function
        cost = 0
        for i in range(self.N):
            # Position error
            position_error = X[:2,i] - X_ref[:2,i]
            
            # Orientation error - use angular distance metric
            orientation_error = (1 - ca.cos(X[2,i] - X_ref[2,i])) * 2
            
            # Joint positions error
            joint_error = X[3:,i] - X_ref[3:,i]
            
            # Combined weighted state cost with elementwise weights
            position_cost = self.Q_state[:2].reshape(1, 2) @ (position_error * position_error)
            orientation_cost = self.Q_state[2] * orientation_error
            joint_cost = self.Q_state[3:].reshape(1, self.nx-3) @ (joint_error * joint_error)
            
            state_cost = position_cost + orientation_cost + joint_cost
            cost += state_cost
            
            # Control costs
            control_cost = self.R_control.reshape(1, self.nu) @ (U[:,i] * U[:,i])
            cost += control_cost
            
            # Penalties for slack variables
            cost += self.slack_dynamics_weight * ca.sumsqr(slack_dynamics[:,i])
        
        # Add penalty for CBF slack variables - only for active obstacles
        for i in range(self.M_CBF):
            for j in range(self.max_obstacle_points):
                # Use mask to apply slack cost only for active obstacles
                cost += self.slack_cbf_weight * obstacle_mask[j] * ca.sumsqr(cbf_slack[j,i])
        
        # Dynamics constraints with slack variables
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
            
            # Relax dynamics constraints with slack variables
            opti.subject_to(X[:,i+1] == x + dx * self.dt + slack_dynamics[:,i])
        
        # Input constraints
        for i in range(self.N):
            # Linear velocity
            opti.subject_to(U[0,i] <= self.max_linear_vel)
            opti.subject_to(U[0,i] >= -self.max_linear_vel)
            
            # Angular velocity
            opti.subject_to(U[1,i] <= self.max_angular_vel)
            opti.subject_to(U[1,i] >= -self.max_angular_vel)
            
            # Torso and arm joint velocities
            for j in range(self.n_torso_joints + self.n_arm_joints):
                opti.subject_to(U[2+j,i] <= self.max_joint_vel[j])
                opti.subject_to(U[2+j,i] >= -self.max_joint_vel[j])
        
        # Acceleration constraints - limit rate of change between consecutive control inputs
        # For the first control input, compare with the previous control input (parameter)
        # Linear acceleration (base)
        opti.subject_to(U[0,0] - U_prev[0] <= self.max_linear_acc * self.dt)
        opti.subject_to(U[0,0] - U_prev[0] >= -self.max_linear_acc * self.dt)
        
        # Angular acceleration (base)
        opti.subject_to(U[1,0] - U_prev[1] <= self.max_angular_acc * self.dt)
        opti.subject_to(U[1,0] - U_prev[1] >= -self.max_angular_acc * self.dt)
        
        # Joint accelerations (torso and arm)
        for j in range(self.n_torso_joints + self.n_arm_joints):
            opti.subject_to(U[2+j,0] - U_prev[2+j] <= self.max_joint_acc[j] * self.dt)
            opti.subject_to(U[2+j,0] - U_prev[2+j] >= -self.max_joint_acc[j] * self.dt)
        
        # For the rest of the prediction horizon, compare consecutive control inputs
        for i in range(1, self.acc_horizon):
            # Linear acceleration (base)
            opti.subject_to(U[0,i] - U[0,i-1] <= self.max_linear_acc * self.dt)
            opti.subject_to(U[0,i] - U[0,i-1] >= -self.max_linear_acc * self.dt)
            
            # Angular acceleration (base)
            opti.subject_to(U[1,i] - U[1,i-1] <= self.max_angular_acc * self.dt)
            opti.subject_to(U[1,i] - U[1,i-1] >= -self.max_angular_acc * self.dt)
            
            # Joint accelerations (torso and arm)
            for j in range(self.n_torso_joints + self.n_arm_joints):
                opti.subject_to(U[2+j,i] - U[2+j,i-1] <= self.max_joint_acc[j] * self.dt)
                opti.subject_to(U[2+j,i] - U[2+j,i-1] >= -self.max_joint_acc[j] * self.dt)
        
        # Non-negativity constraints for slack variables
        opti.subject_to(opti.bounded(0, slack_dynamics, 1.0))
        
        # Initial state constraint (no slack)
        opti.subject_to(X[:,0] == X0)
        
        # Define the barrier function
        def h(x_, y_):
            return (x_[0] - y_[0])**2 + (x_[1] - y_[1])**2 - self.safe_distance**2
        
        # Add CBF constraints for each obstacle - using the obstacle_mask to dynamically
        # enable or disable constraints
        for j in range(self.max_obstacle_points):
            # Create the constraint expressions but only apply them conditionally
            for i in range(self.M_CBF):
                # Current and next state positions
                robot_curr = X[:2, i]
                robot_next = X[:2, i+1]
                
                # Get obstacle positions from parameter array
                # Each obstacle has N+1 positions (one for each timestep)
                obs_curr_x = obstacle_params[0, j*(self.N+1) + i]
                obs_curr_y = obstacle_params[1, j*(self.N+1) + i]
                obs_next_x = obstacle_params[0, j*(self.N+1) + i+1]
                obs_next_y = obstacle_params[1, j*(self.N+1) + i+1]
                
                obs_curr = ca.vertcat(obs_curr_x, obs_curr_y)
                obs_next = ca.vertcat(obs_next_x, obs_next_y)
                
                # Create the CBF constraint
                cbf_expression = h(robot_next, obs_next) - (1 - self.gamma_k) * h(robot_curr, obs_curr) + cbf_slack[j, i]
                
                # Only apply the constraint if obstacle_mask[j] == 1 (active obstacle)
                # This multiplies by a very large negative number if mask=0, effectively disabling the constraint
                # If the obstacle is active (mask=1), we just use the original constraint
                opti.subject_to(obstacle_mask[j] * cbf_expression + (1 - obstacle_mask[j]) * 1e6 >= 0)
        
        # Set the objective
        opti.minimize(cost)
        
        # Store the optimization problem and variables/parameters
        self.opti = opti
        self.cost = cost
        self.X = X
        self.U = U
        self.X_ref = X_ref
        self.X0 = X0
        self.U_prev = U_prev
        self.obstacle_params = obstacle_params
        self.obstacle_mask = obstacle_mask  # Store the new parameter
        self.slack_dynamics = slack_dynamics
        self.cbf_slack = cbf_slack
    
    def update_obstacles(self, obstacle_positions):
        """Update obstacle positions"""
        self.obstacle_positions = obstacle_positions
        
        if self.debug and len(self.obstacle_positions) > 0:
            rospy.logdebug(f"Updated {len(self.obstacle_positions)} obstacle positions")

    def solve(self, x0, x_ref, predicted_obstacles=None, u_prev=None):
        """Solve the MPC problem with CBF constraints"""
        try:
            # Set parameter values
            self.opti.set_value(self.X0, x0)
            self.opti.set_value(self.X_ref, x_ref)
            
            # Set previous control input (for acceleration constraints)
            if u_prev is None:
                u_prev = np.zeros(self.nu)
            self.opti.set_value(self.U_prev, u_prev)
            
            # If no predictions provided, assume obstacles are static
            if predicted_obstacles is None:
                predicted_obstacles = []
                for obs_pos in self.obstacle_positions:
                    # Create a trajectory by repeating the current position
                    obs_traj = [obs_pos] * (self.N + 1)
                    predicted_obstacles.append(obs_traj)
            
            # Create obstacle mask (1 for active obstacles, 0 for inactive)
            obstacle_mask = np.zeros(self.max_obstacle_points)
            num_actual_obstacles = min(len(predicted_obstacles), self.max_obstacle_points)
            
            # Set 1s for active obstacles (up to the actual count)
            obstacle_mask[:num_actual_obstacles] = 1.0
            
            # Set the obstacle mask parameter
            self.opti.set_value(self.obstacle_mask, obstacle_mask)
            
            if self.debug:
                rospy.logdebug(f"Setting actual obstacle count to: {num_actual_obstacles}")
                rospy.logdebug(f"Obstacle mask: {obstacle_mask}")
            
            # Pack obstacle trajectories into the parameter array
            obstacle_data = np.zeros((2, self.max_obstacle_points * (self.N+1)))
            
            for j, obs_traj in enumerate(predicted_obstacles):
                if j >= self.max_obstacle_points:
                    break  # Don't exceed maximum number of obstacles
                    
                for i in range(self.N+1):
                    if i < len(obs_traj):
                        # Store x,y for each obstacle at each time step
                        obstacle_data[0, j*(self.N+1) + i] = obs_traj[i][0]  # x
                        obstacle_data[1, j*(self.N+1) + i] = obs_traj[i][1]  # y
                    else:
                        # If trajectory is shorter than horizon, repeat last position
                        obstacle_data[0, j*(self.N+1) + i] = obs_traj[-1][0]
                        obstacle_data[1, j*(self.N+1) + i] = obs_traj[-1][1]
            
            # Set obstacle parameter values
            self.opti.set_value(self.obstacle_params, obstacle_data)
            
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
                cbf_slack = sol.value(self.cbf_slack)
                
                if np.any(slack_dynamics > 1e-6):
                    rospy.logwarn(f"Dynamics slack active, max: {np.max(slack_dynamics)}")
                    
                if np.any(cbf_slack > 1e-6):
                    # Only check slacks for active obstacles
                    active_slacks = cbf_slack[:num_actual_obstacles, :]
                    if np.any(active_slacks > 1e-6):
                        rospy.logwarn(f"CBF slack active, max: {np.max(active_slacks)}")
            
            return X_opt, U_opt
            
        except Exception as e:
            rospy.logerr(f"MPC solve failed: {e}")
            # Print more debug information if in debug mode
            if self.debug:
                import traceback
                rospy.logerr(traceback.format_exc())
            return None, None


class WholeBodyController:
    def __init__(self):
        rospy.init_node('fetch_whole_body_controller')
        
        # State tracking
        self.joint_states = None
        self.base_pose = None
        self.arm_traj = []
        self.base_traj = []
        self.merged_traj = []
        self.executing = False
        self.lidar_data = None
        self.obstacle_positions = []
        
        # Track previous control input for acceleration constraints
        self.prev_control = None
        
        # Create TF buffer with longer cache time for better performance
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Load parameters
        self.params = self.load_parameters()
        self.mpc = WholeBodyMPC(self.params)
        
        # ROS interfaces
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.joint_vel_pub = rospy.Publisher(
            '/arm_with_torso_controller/joint_velocity_controller/command', JointState, queue_size=1)
        
        # Visualization publishers
        self.obstacle_marker_pub = rospy.Publisher('/obstacle_visualization', MarkerArray, queue_size=1)
        self.cloud_pub = rospy.Publisher('/processed_obstacle_cloud', PointCloud2, queue_size=1)
        
        # Subscribe to sensor topics
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_cb)
        rospy.Subscriber('/base_scan', LaserScan, self.lidar_cb)
        
        # Subscribe to trajectory topics
        rospy.Subscriber('/fetch_whole_body_controller/arm_path', JointTrajectory, self.arm_cb)
        rospy.Subscriber('/fetch_whole_body_controller/base_path', JointTrajectory, self.base_cb)
        
        # Create timers for execution and obstacle processing
        self.execution_timer = None
        
        # Process obstacles separately at a potentially different rate
        self.obstacle_timer = rospy.Timer(
            rospy.Duration(1.0 / 5.0),  # Process obstacles at 5 Hz
            self.process_obstacles_cb
        )
        
        rospy.loginfo("Whole Body Controller initialized with CBF-based obstacle avoidance and acceleration constraints")

    def load_parameters(self):
        return {
            'control_rate': rospy.get_param('~control_rate', 10.0),
            'prediction_horizon': rospy.get_param('~prediction_horizon', 10),
            'max_linear_vel': rospy.get_param('~max_linear_vel', 1.5),
            'max_angular_vel': rospy.get_param('~max_angular_vel', 1.0),
            'max_joint_velocities': rospy.get_param('~max_joint_velocities', 
                [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            # Acceleration limits (new parameters)
            'max_linear_acc': rospy.get_param('~max_linear_acc', 0.5),
            'max_angular_acc': rospy.get_param('~max_angular_acc', 0.8),
            'max_joint_accelerations': rospy.get_param('~max_joint_accelerations', 
                [0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            'Q_state': rospy.get_param('~Q_state', 
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            'R_control': rospy.get_param('~R_control', 
                [2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            'slack_dynamics_weight': rospy.get_param('~slack_dynamics_weight', 1000.0),
            'base_pos_threshold': rospy.get_param('~base_pos_threshold', 0.1),
            'min_waypoint_distance': rospy.get_param('~min_waypoint_distance', 0.05),
            'trajectory_end_threshold': rospy.get_param('~trajectory_end_threshold', 0.2),
            'lidar_max_range': rospy.get_param('~lidar_max_range', 1.0),
            'safe_distance': rospy.get_param('~safe_distance', 0.3),
            'voxel_size': rospy.get_param('~voxel_size', 0.2),
            'max_obstacle_points': rospy.get_param('~max_obstacle_points', 10),
            'obstacle_weight': rospy.get_param('~obstacle_weight', 100.0),
            'slack_obstacle_weight': rospy.get_param('~slack_obstacle_weight', 200.0),
            'gamma_k': rospy.get_param('~gamma_k', 0.1),  # CBF decay rate
            'debug': rospy.get_param('~debug', True)
        }

    def joint_cb(self, msg):
        # Only process messages with length > 2 to filter out gripper messages
        if len(msg.name) > 2:
            self.joint_states = dict(zip(msg.name, msg.position))
        else:
            rospy.logdebug(f"Ignoring joint state message with only {len(msg.name)} joints")

    def amcl_pose_cb(self, msg):
        """Callback for AMCL pose messages"""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.base_pose = [pos.x, pos.y, yaw]
        
        if self.params['debug']:
            rospy.logdebug(f"AMCL Pose updated: x={pos.x:.2f}, y={pos.y:.2f}, yaw={yaw:.2f}")

    def lidar_cb(self, msg):
        """Store the latest lidar scan"""
        self.lidar_data = msg

    def process_obstacles_cb(self, event=None):
        """Process lidar data to extract obstacle points - run as a separate timer"""
        if self.lidar_data is None:
            return
            
        self.process_lidar_data()

    def process_lidar_data(self):
        """Process lidar data using proper TF transformations"""
        if self.lidar_data is None:
            return
        
        try:
            # Get the transform from laser frame to map frame
            transform = self.tf_buffer.lookup_transform(
                "map",                      # Target frame
                self.lidar_data.header.frame_id,  # Source frame (laser_link)
                self.lidar_data.header.stamp,     # Time of the scan
                rospy.Duration(0.1)                # Timeout
            )
            
            # Filter valid ranges
            ranges = np.array(self.lidar_data.ranges)
            angles = np.arange(
                self.lidar_data.angle_min,
                self.lidar_data.angle_max + self.lidar_data.angle_increment,
                self.lidar_data.angle_increment
            )
            
            valid_indices = np.where(
                (ranges > self.lidar_data.range_min) & 
                (ranges < self.lidar_data.range_max) & 
                (ranges < self.params['lidar_max_range']) &
                (~np.isnan(ranges)) & 
                (~np.isinf(ranges))
            )[0]
            
            if len(valid_indices) == 0:
                self.obstacle_positions = []
                self.mpc.update_obstacles(self.obstacle_positions)
                return
                
            # Extract valid ranges and angles
            valid_ranges = ranges[valid_indices]
            valid_angles = angles[valid_indices]
            
            # Convert to cartesian coordinates (in lidar frame)
            x_lidar = valid_ranges * np.cos(valid_angles)
            y_lidar = valid_ranges * np.sin(valid_angles)
            
            # Create list of points for transformation
            points_map = []
            
            # Transform each point individually using tf2
            for i in range(len(x_lidar)):
                point_lidar = PointStamped()
                point_lidar.header = self.lidar_data.header
                point_lidar.point.x = x_lidar[i]
                point_lidar.point.y = y_lidar[i]
                point_lidar.point.z = 0.0
                
                # Transform the point to map frame
                point_map = tf2_geometry_msgs.do_transform_point(point_lidar, transform)
                
                # Store the transformed point
                points_map.append((point_map.point.x, point_map.point.y))
            
            # Group points into voxels using a dictionary for efficiency
            voxel_dict = {}
            
            for x, y in points_map:
                voxel_x = int(x / self.params['voxel_size'])
                voxel_y = int(y / self.params['voxel_size'])
                
                voxel_key = (voxel_x, voxel_y)
                if voxel_key not in voxel_dict:
                    voxel_dict[voxel_key] = []
                    
                voxel_dict[voxel_key].append((x, y))
            
            # Calculate centroid of each voxel
            obstacle_positions = []
            
            for voxel_points in voxel_dict.values():
                x_sum = sum(p[0] for p in voxel_points)
                y_sum = sum(p[1] for p in voxel_points)
                count = len(voxel_points)
                
                centroid = (x_sum / count, y_sum / count)
                obstacle_positions.append(centroid)
            
            # Sort by distance to robot (if base_pose is available)
            if self.base_pose is not None:
                robot_x, robot_y, _ = self.base_pose
                obstacle_positions.sort(key=lambda p: (p[0] - robot_x)**2 + (p[1] - robot_y)**2)
            
            # Keep only the closest obstacles
            obstacle_positions = obstacle_positions[:self.params['max_obstacle_points']]
            
            # Update MPC with new obstacle positions
            self.obstacle_positions = obstacle_positions
            self.mpc.update_obstacles(obstacle_positions)
            
            # Publish visualization
            self.publish_obstacle_markers()
            self.publish_point_cloud(points_map)
            
            if self.params['debug']:
                rospy.logdebug(f"Detected {len(obstacle_positions)} obstacle voxels")
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing lidar data: {e}")
            if self.params['debug']:
                import traceback
                rospy.logerr(traceback.format_exc())

    def publish_obstacle_markers(self):
        """Publish visualization markers for detected obstacles"""
        marker_array = MarkerArray()
        
        for i, (x, y) in enumerate(self.obstacle_positions):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1  # Slightly above ground
            
            marker.pose.orientation.w = 1.0
            
            # Size
            marker.scale.x = self.params['safe_distance'] * 2  # Diameter
            marker.scale.y = self.params['safe_distance'] * 2  # Diameter
            marker.scale.z = 0.1  # Height
            
            # Color (red, semi-transparent)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            marker.lifetime = rospy.Duration(0.5)  # half-second lifetime
            
            marker_array.markers.append(marker)
        
        self.obstacle_marker_pub.publish(marker_array)

    def publish_point_cloud(self, points):
        """Publish a point cloud for visualization"""
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        
        # Create point cloud fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
        ]
        
        # Create the point cloud data
        cloud_data = []
        for x, y in points:
            # RGBA value (blue color)
            rgba = struct.unpack('I', struct.pack('BBBB', 0, 0, 255, 255))[0]
            cloud_data.append([x, y, 0.1, rgba])  # z slightly above ground
            
        # Create and publish the point cloud
        cloud = pc2.create_cloud(header, fields, cloud_data)
        self.cloud_pub.publish(cloud)

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

    def check_trajectories(self):
        if not self.arm_traj or not self.base_traj:
            return
            
        if len(self.arm_traj['points']) != len(self.base_traj['points']):
            rospy.logerr("Trajectory mismatch: different number of points!")
            return
        
        # Create merged trajectory
        self.merged_traj = [
            np.concatenate([base, arm])
            for base, arm in zip(self.base_traj['points'], self.arm_traj['points'])
        ]
        
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
            
        self.executing = True
        # Create a timer that calls the trajectory execution callback at the control rate
        self.execution_timer = rospy.Timer(
            rospy.Duration(1.0 / self.params['control_rate']),
            self.trajectory_execution_cb
        )
        rospy.loginfo("Starting trajectory execution with CBF-based obstacle avoidance")

    def get_current_state(self):
        if self.joint_states is None or self.base_pose is None:
            return None
            
        joints = [
            self.joint_states.get('torso_lift_joint', 0),
            *[self.joint_states.get(f, 0) for f in [
                'shoulder_pan_joint', 'shoulder_lift_joint',
                'upperarm_roll_joint', 'elbow_flex_joint',
                'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'
            ]]
        ]
        return np.concatenate([self.base_pose, joints])

    def find_nearest_waypoint(self, current_state):
        """Find the index of the waypoint closest to the current base position and orientation"""
        current_base_pos = current_state[:2]  # x, y coordinates
        current_orientation = current_state[2]  # theta
        
        min_dist = float('inf')
        nearest_idx = 0
        
        # Check distances to all waypoints, including orientation
        for i, waypoint in enumerate(self.base_traj['points']):
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
        
        for i in range(self.mpc.N+1):
            idx = min(start_idx + i, len(self.merged_traj) - 1)
            ref[:,i] = self.merged_traj[idx]
            
        return ref

    def predict_obstacle_trajectories(self):
        """Generate predicted obstacle trajectories for the CBF constraints
        
        For now, this assumes static obstacles, but you could implement more advanced
        prediction methods that consider obstacle velocity or use learned models.
        """
        predicted_obstacles = []
        
        for obs_pos in self.obstacle_positions:
            # For static obstacles, just repeat the current position over the horizon
            obs_traj = [obs_pos] * (self.mpc.N + 1)
            predicted_obstacles.append(obs_traj)
            
        return predicted_obstacles

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
        # Get current state
        current_state = self.get_current_state()
        if current_state is None:
            rospy.logwarn("Cannot get current state, skipping control iteration")
            return
        
        # Find nearest waypoint
        nearest_idx = self.find_nearest_waypoint(current_state)
        
        # Generate reference trajectory based on nearest waypoint
        ref = self.construct_reference_from_waypoint(nearest_idx)
        
        # Generate predicted obstacle trajectories
        predicted_obstacles = self.predict_obstacle_trajectories()
        
        if self.params['debug']:
            rospy.logdebug(f"Nearest waypoint index: {nearest_idx} of {len(self.merged_traj)}")
            rospy.logdebug(f"Number of obstacles: {len(predicted_obstacles)}")
            start_time = rospy.Time.now()
            
        # Solve MPC with CBF constraints and acceleration constraints
        X, U = self.mpc.solve(current_state, ref, predicted_obstacles, self.prev_control)
        
        if self.params['debug']:
            solve_time = (rospy.Time.now() - start_time).to_sec()
            rospy.logdebug(f"MPC solve time: {solve_time:.4f} seconds")
        
        if U is None:
            rospy.logerr("MPC solve failed. Stopping execution.")
            self.stop_execution()
            return
            
        # Store current control input as previous for next iteration
        self.prev_control = U[:,0]
            
        # Publish commands
        self.publish_commands(U[:,0])
        
        # Check if we've reached the end of the trajectory
        if nearest_idx >= len(self.merged_traj) - 1:
            # Near the final waypoint, check if we're close enough
            if self.is_trajectory_complete(current_state, self.merged_traj[-1]):
                rospy.loginfo("Trajectory completed successfully")
                self.stop_execution()
                return
    
    def stop_execution(self):
        """Stop trajectory execution and cleanup timer"""
        self.stop()
        self.executing = False
        
        # Shutdown the timer if it exists
        if self.execution_timer is not None:
            self.execution_timer.shutdown()
            self.execution_timer = None

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
        # Update prev_control to zeros for smooth restart if needed
        self.prev_control = zero_control


if __name__ == '__main__':
    try:
        controller = WholeBodyController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass