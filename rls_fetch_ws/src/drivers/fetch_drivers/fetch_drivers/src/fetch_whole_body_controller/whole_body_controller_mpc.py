#!/usr/bin/env python3
import rospy
import casadi as ca
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory
import tf2_ros
import tf.transformations as tf_trans


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
        
        # Weights
        self.Q_state = params['Q_state']
        self.R_control = params['R_control']
        self.R_smooth = params['R_smooth']
        self.P_terminal = params['P_terminal']
        
        # Slack weights
        self.slack_dynamics_weight = params.get('slack_dynamics_weight', 1000.0)
        self.slack_input_weight = params.get('slack_input_weight', 100.0)
        
        # Debug flag
        self.debug = params['debug']
        
        self.solver = self.create_solver()

    def create_solver(self):
        opti = ca.Opti()
        
        # Decision variables
        X = opti.variable(self.nx, self.N+1)
        U = opti.variable(self.nu, self.N)
        
        # Add slack variables for constraints relaxation
        slack_dynamics = opti.variable(self.nx, self.N)
        slack_inputs = opti.variable(self.nu, self.N)
        
        # Parameters
        X_ref = opti.parameter(self.nx, self.N+1)
        X0 = opti.parameter(self.nx)
        
        # Slack variable weights (penalties)
        slack_dynamics_weight = self.slack_dynamics_weight
        slack_input_weight = self.slack_input_weight
        
        # Cost function
        cost = 0
        for i in range(self.N):
            # Split the state vector into components for proper handling
            # Position (x, y)
            position_error = ca.sumsqr(X[:2,i] - X_ref[:2,i])
            
            # Orientation (theta) - use angular distance metric
            orientation_error = 1 - ca.cos(X[2,i] - X_ref[2,i])
            
            # Joint positions
            joint_error = ca.sumsqr(X[3:,i] - X_ref[3:,i])
            
            # Combined cost with appropriate weights
            state_cost = self.Q_state * (position_error + 2 * orientation_error + joint_error)
            cost += state_cost
            
            # Control and smoothness costs remain unchanged
            cost += self.R_control * ca.sumsqr(U[:,i])
            if i > 0:
                cost += self.R_smooth * ca.sumsqr(U[:,i] - U[:,i-1])
            
            # Add penalties for slack variables
            cost += slack_dynamics_weight * ca.sumsqr(slack_dynamics[:,i])
            cost += slack_input_weight * ca.sumsqr(slack_inputs[:,i])
        
        # Terminal cost - also using proper angle handling
        position_error_terminal = ca.sumsqr(X[:2,-1] - X_ref[:2,-1])
        orientation_error_terminal = 1 - ca.cos(X[2,-1] - X_ref[2,-1])
        joint_error_terminal = ca.sumsqr(X[3:,-1] - X_ref[3:,-1])
        terminal_cost = self.P_terminal * (position_error_terminal + 2 * orientation_error_terminal + joint_error_terminal)
        cost += terminal_cost
        
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
        
        # Input constraints with slack variables
        for i in range(self.N):
            # Linear velocity
            opti.subject_to(U[0,i] <= self.max_linear_vel + slack_inputs[0,i])
            opti.subject_to(U[0,i] >= -self.max_linear_vel - slack_inputs[0,i])
            
            # Angular velocity
            opti.subject_to(U[1,i] <= self.max_angular_vel + slack_inputs[1,i])
            opti.subject_to(U[1,i] >= -self.max_angular_vel - slack_inputs[1,i])
            
            # Joint velocities
            for j in range(self.nu-2):
                opti.subject_to(U[2+j,i] <= self.max_joint_vel[j] + slack_inputs[2+j,i])
                opti.subject_to(U[2+j,i] >= -self.max_joint_vel[j] - slack_inputs[2+j,i])
        
        # Non-negativity constraints for slack variables
        opti.subject_to(opti.bounded(0, slack_dynamics, 1.0))
        opti.subject_to(opti.bounded(0, slack_inputs, 0.5))
        
        # Initial state constraint (no slack)
        opti.subject_to(X[:,0] == X0)
        
        # Define objective
        opti.minimize(cost)
        
        # Set solver options
        opts = {
            'ipopt.print_level': 0 if not self.debug else 3,
            'print_time': 0 if not self.debug else 1,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_obj_change_tol': 1e-4
        }
        opti.solver('ipopt', opts)
        
        return opti.to_function('solver', [X0, X_ref], [X, U, slack_dynamics, slack_inputs], 
                             ['x0', 'x_ref'], ['X', 'U', 'slack_dynamics', 'slack_inputs'])

    def solve(self, x0, x_ref):
        try:
            # Print input information if in debug mode
            if self.debug:
                rospy.loginfo("MPC Solver Input:")
                rospy.loginfo(f"Current state (x0): {x0}")
                rospy.loginfo(f"Reference trajectory shape: {x_ref.shape}")
            
            # Call the solver
            res = self.solver(x0=x0, x_ref=x_ref)
            
            # Get results and convert to numpy arrays
            X = np.array(res['X'])
            U = np.array(res['U'])
            slack_dynamics = np.array(res['slack_dynamics'])
            slack_inputs = np.array(res['slack_inputs'])
            
            # Print detailed results for validation if in debug mode
            if self.debug:
                rospy.loginfo("MPC Solver Results:")
                rospy.loginfo(f"Solved state trajectory shape: {X.shape}")
                rospy.loginfo(f"Solved control trajectory shape: {U.shape}")
                rospy.loginfo(f"First control action: {U[:,0]}")
                rospy.loginfo(f"Current state: {X[:,0]}")
                rospy.loginfo(f"Predicted next state: {X[:,1]}")
                
                # Log slack variable usage to diagnose constraint violations
                if np.any(slack_dynamics > 1e-6) or np.any(slack_inputs > 1e-6):
                    rospy.logwarn("Slack variables active in solution:")
                    rospy.logwarn(f"Max dynamics slack: {np.max(slack_dynamics)}")
                    rospy.logwarn(f"Max input slack: {np.max(slack_inputs)}")
                    
                    # Log which constraints are being violated the most
                    max_dyn_idx = np.unravel_index(np.argmax(slack_dynamics), slack_dynamics.shape)
                    max_inp_idx = np.unravel_index(np.argmax(slack_inputs), slack_inputs.shape)
                    
                    rospy.logwarn(f"Largest dynamics slack at state dimension {max_dyn_idx[0]}, time step {max_dyn_idx[1]}")
                    rospy.logwarn(f"Largest input slack at control dimension {max_inp_idx[0]}, time step {max_inp_idx[1]}")
            
            # Always check for invalid solutions, regardless of debug mode
            if np.any(np.isnan(U)) or np.any(np.isinf(U)):
                rospy.logerr("Invalid solution: NaN or inf values in control")
                return None, None
                
            # Warn if slack variables are significantly active, indicating infeasibility
            if np.max(slack_dynamics) > 0.1 or np.max(slack_inputs) > 0.1:
                rospy.logwarn("Solution required significant constraint relaxation")
            
            return X, U
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
        
        # Load parameters
        self.params = self.load_parameters()
        self.mpc = WholeBodyMPC(self.params)
        
        # ROS interfaces
        self.cmd_vel_pub = rospy.Publisher('/base_controller/command', Twist, queue_size=1)
        self.joint_vel_pub = rospy.Publisher(
            '/arm_with_torso_controller/joint_velocity_controller/command', JointState, queue_size=1)
        
        # Create a timer for trajectory execution, but don't start it yet
        self.execution_timer = None
        
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        
        # Replace odometry subscription with AMCL pose
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_cb)
        
        rospy.Subscriber('/fetch_whole_body_controller/arm_path', JointTrajectory, self.arm_cb)
        rospy.Subscriber('/fetch_whole_body_controller/base_path', JointTrajectory, self.base_cb)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo("Whole Body Controller initialized with AMCL pose tracking")

    def load_parameters(self):
        return {
            'control_rate': rospy.get_param('~control_rate', 20.0),
            'prediction_horizon': rospy.get_param('~prediction_horizon', 10),
            'max_linear_vel': rospy.get_param('~max_linear_vel', 0.5),
            'max_angular_vel': rospy.get_param('~max_angular_vel', 1.0),
            'max_joint_velocities': rospy.get_param('~max_joint_velocities', 
                [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            'Q_state': rospy.get_param('~Q_state', 10.0),
            'R_control': rospy.get_param('~R_control', 1.0),
            'R_smooth': rospy.get_param('~R_smooth', 5.0),
            'P_terminal': rospy.get_param('~P_terminal', 20.0),
            'slack_dynamics_weight': rospy.get_param('~slack_dynamics_weight', 1000.0),
            'slack_input_weight': rospy.get_param('~slack_input_weight', 100.0),
            'base_pos_threshold': rospy.get_param('~base_pos_threshold', 0.1),
            'min_waypoint_distance': rospy.get_param('~min_waypoint_distance', 0.05),
            'trajectory_end_threshold': rospy.get_param('~trajectory_end_threshold', 0.2),
            'debug': rospy.get_param('~debug', False)
        }

    def joint_cb(self, msg):
        self.joint_states = dict(zip(msg.name, msg.position))

    def amcl_pose_cb(self, msg):
        """Callback for AMCL pose messages instead of odometry"""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.base_pose = [pos.x, pos.y, yaw]
        
        if self.params['debug']:
            rospy.loginfo(f"AMCL Pose updated: x={pos.x:.2f}, y={pos.y:.2f}, yaw={yaw:.2f}")
            
            # Optionally, log covariance information if needed for diagnostics
            cov = msg.pose.covariance
            pos_cov = [cov[0], cov[7], cov[35]]  # x, y, yaw variance
            rospy.logdebug(f"AMCL Pose covariance: {pos_cov}")

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
        # Replace threading with a ROS timer
        # Create a timer that calls the trajectory execution callback at the control rate
        self.execution_timer = rospy.Timer(
            rospy.Duration(1.0 / self.params['control_rate']),
            self.trajectory_execution_cb
        )
        rospy.loginfo("Starting trajectory execution with AMCL-based localization")

    def get_current_state(self):
        if self.joint_states is None or self.base_pose is None:
            return None
            
        joints = [
            self.joint_states['torso_lift_joint'],
            *[self.joint_states[f] for f in [
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
            # Weight of 0.5 for orientation component (can be adjusted)
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
        # Get current state
        current_state = self.get_current_state()
        if current_state is None:
            rospy.logwarn("Cannot get current state, skipping control iteration")
            return
        
        # Find nearest waypoint
        nearest_idx = self.find_nearest_waypoint(current_state)
        
        # Debug logging
        if self.params['debug']:
            rospy.loginfo(f"Nearest waypoint index: {nearest_idx} of {len(self.merged_traj)}")
        
        # Generate reference trajectory based on nearest waypoint
        ref = self.construct_reference_from_waypoint(nearest_idx)
        
        if self.params['debug']:
            rospy.loginfo(f"Generated reference trajectory with shape: {ref.shape}")
            rospy.loginfo("Solving MPC problem...")
            start_time = rospy.Time.now()
            
        # Solve MPC
        X, U = self.mpc.solve(current_state, ref)
        
        if self.params['debug']:
            solve_time = (rospy.Time.now() - start_time).to_sec()
            rospy.loginfo(f"MPC solve time: {solve_time:.4f} seconds")
        
        if U is None:
            rospy.logerr("MPC solve failed. Stopping execution.")
            self.stop_execution()
            return
            
        # Print summary of control actions if in debug mode
        if self.params['debug']:
            rospy.loginfo(f"MPC solution: base linear vel={U[0,0]:.3f}, angular vel={U[1,0]:.3f}")
        
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
        # Convert DM to numpy array first
        u_np = np.array(u).flatten()
        js.velocity = [float(v) for v in u_np[2:]]
        self.joint_vel_pub.publish(js)

    def stop(self):
        self.publish_commands(np.zeros(self.mpc.nu))


if __name__ == '__main__':
    controller = WholeBodyController()
    rospy.spin()