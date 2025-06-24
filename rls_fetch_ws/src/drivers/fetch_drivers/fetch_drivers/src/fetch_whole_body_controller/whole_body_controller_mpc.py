#!/usr/bin/env python3
import rospy
import casadi as ca
import numpy as np
from geometry_msgs.msg import Twist, PointStamped, Point
from sensor_msgs.msg import JointState, LaserScan
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tf_trans
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import struct
from std_msgs.msg import Bool


class WholeBodyMPC:
    def __init__(self, params):
        self.N = params['prediction_horizon']
        self.dt = 0.1
        
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
        self.slack_cbf_weight = params.get('slack_obstacle_weight', 200.0)
        
        # Terminal cost weights
        self.P_state = np.array(params['P_state'])
        
        # Obstacle avoidance parameters
        self.safe_distance = params.get('safe_distance', 0.3)
        self.voxel_size = params.get('voxel_size', 0.2)
        self.max_obstacle_points = params.get('max_obstacle_points', 10)
        self.lidar_max_range = params.get('lidar_max_range', 1.0)
        
        # CBF parameters
        self.gamma_k = params.get('gamma_k', 0.1)  # CBF decay rate
        self.M_CBF = min(3, self.N)  # Number of steps to apply CBF (using full horizon)
        
        # Debug flag
        self.debug = params['debug']
        
        if self.debug:
            rospy.loginfo("Initialized MPC with weights:")
            rospy.loginfo(f"Q_state = {self.Q_state}")
            rospy.loginfo(f"R_control = {self.R_control}")
            rospy.loginfo(f"Obstacle params: safe_distance={self.safe_distance}, gamma_k={self.gamma_k}")
        
        # Initialize obstacle data
        self.obstacle_positions = []
        
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
        
        # Parameters for obstacles
        obstacle_params = opti.parameter(2, self.max_obstacle_points * (self.N+1))
        obstacle_mask = opti.parameter(self.max_obstacle_points)
        
        # Add slack variables for CBF constraints
        cbf_slack = opti.variable(self.max_obstacle_points, self.M_CBF)
        opti.subject_to(opti.bounded(0, cbf_slack, 1.0))
        
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
        
        # Add penalty for CBF slack variables - only for active obstacles
        # for i in range(self.M_CBF):
        #     for j in range(self.max_obstacle_points):
        #         cost += self.slack_cbf_weight * obstacle_mask[j] * ca.sumsqr(cbf_slack[j,i])
        
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
        
        # Define the barrier function (unchanged)
        def h(x_, y_):
            return (x_[0] - y_[0])**2 + (x_[1] - y_[1])**2 - self.safe_distance**2
        
        # Add CBF constraints for each obstacle (unchanged)
        # for j in range(self.max_obstacle_points):
        #     for i in range(self.M_CBF):
        #         robot_curr = X[:2, i]
        #         robot_next = X[:2, i+1]
        #         obs_curr_x = obstacle_params[0, j*(self.N+1) + i]
        #         obs_curr_y = obstacle_params[1, j*(self.N+1) + i]
        #         obs_next_x = obstacle_params[0, j*(self.N+1) + i+1]
        #         obs_next_y = obstacle_params[1, j*(self.N+1) + i+1]
        #         obs_curr = ca.vertcat(obs_curr_x, obs_curr_y)
        #         obs_next = ca.vertcat(obs_next_x, obs_next_y)
        #         cbf_expression = h(robot_next, obs_next) - (1 - self.gamma_k) * h(robot_curr, obs_curr) + cbf_slack[j, i]
        #         opti.subject_to(obstacle_mask[j] * cbf_expression + (1 - obstacle_mask[j]) * 1e6 >= 0)
        
        # Set the objective (unchanged)
        opti.minimize(cost)
        
        # Store the optimization problem and variables/parameters (unchanged)
        self.opti = opti
        self.cost = cost
        self.X = X
        self.U = U
        self.X_ref = X_ref
        self.X0 = X0
        self.obstacle_params = obstacle_params
        self.obstacle_mask = obstacle_mask
        self.slack_dynamics = slack_dynamics
        # self.cbf_slack = cbf_slack
    
    def update_obstacles(self, obstacle_positions):
        """Update obstacle positions"""
        self.obstacle_positions = obstacle_positions
        
        if self.debug and len(self.obstacle_positions) > 0:
            rospy.logdebug(f"Updated {len(self.obstacle_positions)} obstacle positions")

    def solve(self, x0, x_ref, predicted_obstacles=None):
        """Solve the MPC problem with CBF constraints"""
        # Set parameter values
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.X_ref, x_ref)
        
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
            # cbf_slack = sol.value(self.cbf_slack)
            
            if np.any(slack_dynamics > 1e-6):
                rospy.logwarn(f"Dynamics slack active, max: {np.max(slack_dynamics)}")
                
            # if np.any(cbf_slack > 1e-6):
            #     # Only check slacks for active obstacles
            #     active_slacks = cbf_slack[:num_actual_obstacles, :]
            #     if np.any(active_slacks > 1e-6):
            #         rospy.logwarn(f"CBF slack active, max: {np.max(active_slacks)}")
        
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
        self.lidar_data = None
        self.obstacle_positions = []
        
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
        
        # Visualization publishers
        self.obstacle_marker_pub = rospy.Publisher('/obstacle_visualization', MarkerArray, queue_size=1)
        self.cloud_pub = rospy.Publisher('/processed_obstacle_cloud', PointCloud2, queue_size=1)
        
        # Debug publishers for paths
        if self.params['debug']:
            self.ref_path_pub = rospy.Publisher('/ref_path_viz', Marker, queue_size=1)
            self.mpc_path_pub = rospy.Publisher('/mpc_path_viz', Marker, queue_size=1)
            self.global_path_pub = rospy.Publisher('/global_path_viz', Marker, queue_size=1)
        
        # Subscribe to sensor topics
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        rospy.Subscriber('/base_scan', LaserScan, self.lidar_cb)
        
        # Subscribe to trajectory topics
        rospy.Subscriber('/fetch_whole_body_controller/arm_path', JointTrajectory, self.arm_cb)
        rospy.Subscriber('/fetch_whole_body_controller/base_path', JointTrajectory, self.base_cb)
        
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
            'Q_state': rospy.get_param('~Q_state', 
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            'R_control': rospy.get_param('~R_control', 
                [2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            'slack_dynamics_weight': rospy.get_param('~slack_dynamics_weight', 1000.0),
            'base_pos_threshold': rospy.get_param('~base_pos_threshold', 0.05),
            'min_waypoint_distance': rospy.get_param('~min_waypoint_distance', 0.05),
            'trajectory_end_threshold': rospy.get_param('~trajectory_end_threshold', 0.05),
            'lidar_max_range': rospy.get_param('~lidar_max_range', 1.0),
            'safe_distance': rospy.get_param('~safe_distance', 0.3),
            'voxel_size': rospy.get_param('~voxel_size', 0.2),
            'max_obstacle_points': rospy.get_param('~max_obstacle_points', 10),
            'obstacle_weight': rospy.get_param('~obstacle_weight', 100.0),
            'slack_obstacle_weight': rospy.get_param('~slack_obstacle_weight', 200.0),
            'gamma_k': rospy.get_param('~gamma_k', 0.1),  # CBF decay rate
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
            
        # Get the transform from laser frame to map frame at the latest available time.
        # Using rospy.Time(0) requests the latest transform, avoiding the extrapolation error
        # which occurs when the scan's timestamp is too old for the tf buffer.
        transform = self.tf_buffer.lookup_transform(
            "map",                                # Target frame
            self.lidar_data.header.frame_id,      # Source frame (e.g., laser_link)
            rospy.Time(0),                        # FIX: Ask for the latest available transform
            rospy.Duration(0.1)                   # Timeout
        )
        
        # Filter valid ranges
        ranges = np.array(self.lidar_data.ranges)
        # Correctly create the angles array to match the length of ranges
        angles = np.arange(
            self.lidar_data.angle_min,
            self.lidar_data.angle_max,
            self.lidar_data.angle_increment
        )[:len(ranges)]
        
        valid_indices = np.where(
            (ranges > self.lidar_data.range_min) & 
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
        
        # Sort by distance to robot
        current_base_pose = self.get_base_pose_from_tf()
        if current_base_pose is not None:
            robot_x, robot_y, _ = current_base_pose
            obstacle_positions.sort(key=lambda p: (p[0] - robot_x)**2 + (p[1] - robot_y)**2)
        
        # Keep only the closest obstacles
        obstacle_positions = obstacle_positions[:self.params['max_obstacle_points']]
        
        # Update MPC with new obstacle positions
        self.obstacle_positions = obstacle_positions
        self.mpc.update_obstacles(obstacle_positions)
        
        # Publish visualization
        self.publish_obstacle_markers()
        self.publish_point_cloud(points_map)

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
            
        self.executing = True
        # Create a timer that calls the trajectory execution callback at the control rate
        self.execution_timer = rospy.Timer(
            rospy.Duration(1.0 / self.params['control_rate']),
            self.trajectory_execution_cb
        )
        rospy.loginfo("Starting trajectory execution with CBF-based obstacle avoidance")

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
        start_idx += 10
        for i in range(self.mpc.N+1):
            idx = min(start_idx + i, len(self.merged_traj) - 1)
            ref[:,i] = self.merged_traj[idx]
            
        return ref

    def predict_obstacle_trajectories(self):
        """Generate predicted obstacle trajectories for the CBF constraints"""
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
            # Visualize reference path (red)
            self.publish_path_as_marker(ref.T, self.ref_path_pub, 1.0, 0.0, 0.0, 0)
            start_time = rospy.Time.now()
            
        # Solve MPC with CBF constraints and acceleration constraints
        X, U = self.mpc.solve(current_state, ref, predicted_obstacles)
        
        if self.params['debug']:
            solve_time = (rospy.Time.now() - start_time).to_sec()
            rospy.logdebug(f"MPC solve time: {solve_time:.4f} seconds")
            if X is not None:
                # Visualize MPC planned path (green)
                self.publish_path_as_marker(X.T, self.mpc_path_pub, 0.0, 1.0, 0.0, 1)
        
        if U is None:
            rospy.logerr("MPC solve failed. Stopping execution.")
            self.stop_execution()
            return
            
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

        # Publish completion signal
        self.execution_finished_pub.publish(Bool(data=True))
        rospy.loginfo("Published trajectory completion signal.")

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
