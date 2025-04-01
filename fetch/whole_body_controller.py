import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState


class WholeBodyController:
    """
    Controller class for whole body motion of the Fetch robot.
    Provides coordinated control of the base and arm using PID control.
    """

    def __init__(self):
        """Initialize the whole body controller with default parameters."""
        # Publisher for base movement commands
        self.cmd_vel_pub = rospy.Publisher('/base_controller/command', Twist, queue_size=10)
        
        # Publisher for joint velocity commands
        self.joint_vel_pub = rospy.Publisher(
            '/arm_with_torso_controller/joint_velocity_controller/command', 
            JointState, 
            queue_size=10
        )

        # Control loop rate (Hz)
        self.control_rate = 20.0
        self.rate = rospy.Rate(self.control_rate)

        # Set velocity limits
        self.MAX_LINEAR_VEL = 0.5  # m/s
        self.MAX_ANGULAR_VEL = 1.0  # rad/s
        self.POSITION_TOLERANCE = 0.03  # m
        self.ANGLE_TOLERANCE = 0.05  # rad
        
        # PID controller gains for base velocity control
        self.KP_LINEAR = 2.0
        self.KP_ANGULAR = 3.0
        self.KD_LINEAR = 0.5
        self.KD_ANGULAR = 0.8
        
        # PID controller gains for joint velocity control
        self.KP_JOINT = 5.0
        self.KD_JOINT = 0.5
        self.KI_JOINT = 0.1
        
        # For derivative term in base control
        self.last_pos_error = 0
        self.last_ang_error = 0
        
        # For integral and derivative terms in joint control
        self.joint_integral_errors = [0.0] * 8
        self.joint_last_errors = [0.0] * 8
        
        # Maximum joint velocities (torso is slower)
        self.MAX_JOINT_VEL = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Joint names for the arm and torso (8-DOF)
        self.joint_names = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

    def reset_errors(self):
        """Reset all error terms for a new trajectory execution."""
        self.last_pos_error = 0
        self.last_ang_error = 0
        self.joint_integral_errors = [0.0] * 8
        self.joint_last_errors = [0.0] * 8

    def calculate_base_velocity(self, current_position, target_position, distance_error, angle_to_target, final_angle_error, base_position_reached):
        """
        Calculate base velocity commands using PID control.
        
        Args:
            current_position: Current [x, y, theta] of the base
            target_position: Target [x, y, theta] for the base
            distance_error: Euclidean distance to the target position
            angle_to_target: Angle error to face the target position
            final_angle_error: Angle error for the final orientation
            base_position_reached: Whether the base has reached the target position
            
        Returns:
            tuple: (linear_vel, angular_vel) commands for the base
        """
        # Compute derivative terms for smoother control
        d_pos_error = distance_error - self.last_pos_error
        d_ang_error = angle_to_target - self.last_ang_error if not base_position_reached else final_angle_error - self.last_ang_error
        
        # Update last errors for next iteration
        self.last_pos_error = distance_error
        self.last_ang_error = angle_to_target if not base_position_reached else final_angle_error
        
        # PID control for linear velocity
        linear_vel = self.KP_LINEAR * distance_error + self.KD_LINEAR * d_pos_error
        
        # When close to target, slow down and focus on orientation
        if distance_error < 0.1:
            linear_vel *= (distance_error / 0.1)
            # Start considering final orientation
            angular_vel = self.KP_ANGULAR * (0.5 * angle_to_target + 0.5 * final_angle_error) + self.KD_ANGULAR * d_ang_error
        else:
            # Far from target: focus on getting to the right position first
            angular_vel = self.KP_ANGULAR * angle_to_target + self.KD_ANGULAR * d_ang_error
        
        # Limit base velocities
        linear_vel = np.clip(linear_vel, -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
        angular_vel = np.clip(angular_vel, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        
        return linear_vel, angular_vel

    def calculate_joint_velocities(self, current_joint_positions, target_joint_positions):
        """
        Calculate joint velocity commands using PID control.
        
        Args:
            current_joint_positions: Current joint positions [8-DOF]
            target_joint_positions: Target joint positions [8-DOF]
            
        Returns:
            list: Joint velocity commands for each joint
            list: Joint errors
            bool: Whether all joints have reached their targets
        """
        joint_velocities = []
        joint_errors = []
        joints_reached = True
        
        for j in range(len(current_joint_positions)):
            # Calculate joint error
            joint_error = target_joint_positions[j] - current_joint_positions[j]
            joint_errors.append(joint_error)
            
            # Check if joint has reached target
            if abs(joint_error) > 0.02:  # Joint position tolerance
                joints_reached = False
            
            # Calculate integral term (with anti-windup)
            if abs(joint_error) < 0.1:  # Only integrate when close to target
                self.joint_integral_errors[j] += joint_error / self.control_rate
            else:
                self.joint_integral_errors[j] = 0  # Reset when far from target
            
            # Limit integral term to prevent windup
            self.joint_integral_errors[j] = np.clip(self.joint_integral_errors[j], -1.0, 1.0)
            
            # Calculate derivative term
            d_joint_error = (joint_error - self.joint_last_errors[j]) * self.control_rate
            self.joint_last_errors[j] = joint_error
            
            # PID control for joint velocity
            joint_vel = self.KP_JOINT * joint_error + self.KD_JOINT * d_joint_error + self.KI_JOINT * self.joint_integral_errors[j]
            
            # Apply velocity limit for this joint
            joint_vel = np.clip(joint_vel, -self.MAX_JOINT_VEL[j], self.MAX_JOINT_VEL[j])
            
            joint_velocities.append(joint_vel)
        
        return joint_velocities, joint_errors, joints_reached

    def send_velocity_commands(self, linear_vel, angular_vel, joint_velocities):
        """
        Send velocity commands to both the base and joints.
        
        Args:
            linear_vel: Linear velocity for the base
            angular_vel: Angular velocity for the base
            joint_velocities: List of velocities for each joint
            
        Returns:
            None
        """
        # Create and publish velocity command for base
        base_cmd = Twist()
        base_cmd.linear.x = linear_vel
        base_cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(base_cmd)
        
        # Create and publish velocity command for joints
        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.name = self.joint_names
        joint_cmd.velocity = joint_velocities
        self.joint_vel_pub.publish(joint_cmd)

    def stop_all_motion(self):
        """
        Stop all robot motion by sending zero velocity commands.
        
        Returns:
            None
        """
        # Send zero velocity to base
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        
        # Send zero velocity to joints
        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.name = self.joint_names
        joint_cmd.velocity = [0.0] * len(self.joint_names)
        self.joint_vel_pub.publish(joint_cmd) 

    def follow_whole_body_trajectory(self, arm_path, base_configs, tf_buffer, get_current_planning_joints, duration=10.0):
        """
        Follow a coordinated whole body trajectory using direct velocity control.
        
        This implements coordinated velocity control for both the base and arm/torso,
        moving through waypoints sequentially.
        
        Args:
            arm_path: List of arm joint configurations (8-DOF including torso)
            base_configs: List of base configurations [x, y, theta]
            tf_buffer: TF2 buffer for position tracking
            get_current_planning_joints: Function to get current joint positions
            duration: Total duration for the trajectory execution in seconds
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # Validate input
            if arm_path is None or base_configs is None:
                rospy.logerr("Cannot execute motion: arm_path or base_configs is None")
                return False
                
            # Process arm path
            arm_points = []
            for i in range(len(arm_path)):
                point = arm_path[i]
                if isinstance(point, list):
                    arm_points.append(point)
                elif isinstance(point, np.ndarray):
                    arm_points.append(point.tolist())
                else:
                    # Assume it has a to_list method
                    arm_points.append(point.to_list())
            
            # Ensure equal length
            if len(arm_points) != len(base_configs):
                min_len = min(len(arm_points), len(base_configs))
                rospy.logwarn(f"Length mismatch: arm_points({len(arm_points)}) != base_configs({len(base_configs)}). Truncating to {min_len}.")
                arm_points = arm_points[:min_len]
                base_configs = base_configs[:min_len]
            
            # Import necessary modules
            import tf.transformations
            from nav_msgs.msg import Odometry
            
            # Reset controller errors before starting
            self.reset_errors()
            
            # Keep track of odom data
            odom_data = None
            def odom_callback(msg):
                nonlocal odom_data
                odom_data = msg
            
            # Subscribe to odom topic
            odom_subscriber = rospy.Subscriber('/odom', Odometry, odom_callback)
            
            # Wait briefly for first odom message
            wait_start = rospy.Time.now()
            while odom_data is None and not rospy.is_shutdown():
                if (rospy.Time.now() - wait_start).to_sec() > 2.0:
                    rospy.logwarn("No odometry data received. Will use TF for position tracking instead.")
                    break
                rospy.sleep(0.1)
            
            # Calculate time per waypoint
            waypoint_duration = duration / len(arm_points)
            
            rospy.loginfo(f"Executing {len(arm_points)} waypoints over {duration:.2f} seconds")
            
            # Execute trajectory point by point
            for i in range(len(arm_points)):
                if rospy.is_shutdown():
                    rospy.logwarn("ROS shutdown detected, stopping trajectory execution")
                    odom_subscriber.unregister()
                    return False
                
                # Get target position for this waypoint
                target_joint_positions = arm_points[i]
                target_base_config = base_configs[i]
                
                rospy.loginfo(f"Processing waypoint {i+1}/{len(arm_points)}: "
                            f"base [{target_base_config[0]:.2f}, {target_base_config[1]:.2f}, "
                            f"{target_base_config[2]:.2f}], joints [{', '.join([f'{v:.2f}' for v in target_joint_positions[:3]])}...]")
                
                # Control loop for this waypoint
                waypoint_start_time = rospy.Time.now()
                
                # Continue until we reach the position or timeout
                while not rospy.is_shutdown():
                    # Check if we should timeout
                    elapsed = (rospy.Time.now() - waypoint_start_time).to_sec()
                    if elapsed >= waypoint_duration:
                        rospy.logwarn(f"Timeout reached for waypoint {i+1}")
                        break
                    
                    # Get current position from TF
                    try:
                        # First try to get position from map to base_link transform
                        transform = tf_buffer.lookup_transform(
                            "map", "base_link", rospy.Time(0), rospy.Duration(0.1)
                        )
                        
                        # Extract current position and orientation
                        current_x = transform.transform.translation.x
                        current_y = transform.transform.translation.y
                        
                        # Get current yaw from quaternion
                        q = transform.transform.rotation
                        quaternion = [q.x, q.y, q.z, q.w]
                        _, _, current_theta = tf.transformations.euler_from_quaternion(quaternion)
                        
                    except Exception as e:
                        # Fall back to odometry if TF fails
                        if odom_data is None:
                            rospy.logwarn(f"No position data available: {e}")
                            continue
                        
                        # Get position from odom
                        current_x = odom_data.pose.pose.position.x
                        current_y = odom_data.pose.pose.position.y
                        
                        # Get orientation as yaw from quaternion
                        q = odom_data.pose.pose.orientation
                        quaternion = [q.x, q.y, q.z, q.w]
                        _, _, current_theta = tf.transformations.euler_from_quaternion(quaternion)
                        
                        # Transform from odom to map frame if possible
                        try:
                            transform = tf_buffer.lookup_transform(
                                "map", "odom", rospy.Time(0), rospy.Duration(0.1)
                            )
                            
                            # Apply transform
                            cos_yaw = np.cos(transform.transform.rotation.z)
                            sin_yaw = np.sin(transform.transform.rotation.z)
                            
                            # Rotate position
                            tx = transform.transform.translation.x
                            ty = transform.transform.translation.y
                            rx = current_x * cos_yaw - current_y * sin_yaw + tx
                            ry = current_x * sin_yaw + current_y * cos_yaw + ty
                            
                            # Update with transformed values
                            current_x = rx
                            current_y = ry
                            current_theta += transform.transform.rotation.z
                            
                            # Normalize angle
                            while current_theta > np.pi:
                                current_theta -= 2*np.pi
                            while current_theta < -np.pi:
                                current_theta += 2*np.pi
                                
                        except Exception:
                            # Continue with odom frame if transform fails
                            pass
                    
                    # Get current joint positions
                    current_joint_positions = get_current_planning_joints()
                    if current_joint_positions is None:
                        rospy.logwarn("Could not get current joint positions")
                        continue
                    
                    # Calculate position error in map frame for the base
                    dx = target_base_config[0] - current_x
                    dy = target_base_config[1] - current_y
                    distance_error = np.sqrt(dx*dx + dy*dy)
                    
                    # Calculate heading to target
                    target_heading = np.arctan2(dy, dx)
                    
                    # Calculate angle error (handle wraparound)
                    angle_to_target = target_heading - current_theta
                    while angle_to_target > np.pi:
                        angle_to_target -= 2*np.pi
                    while angle_to_target < -np.pi:
                        angle_to_target += 2*np.pi
                    
                    # Final orientation error (only consider when close to target position)
                    final_angle_error = target_base_config[2] - current_theta
                    while final_angle_error > np.pi:
                        final_angle_error -= 2*np.pi
                    while final_angle_error < -np.pi:
                        final_angle_error += 2*np.pi
                    
                    # Determine if we've reached the target for base
                    base_position_reached = distance_error < self.POSITION_TOLERANCE
                    base_angle_reached = abs(final_angle_error) < self.ANGLE_TOLERANCE
                    
                    # Use the controller to calculate base velocity
                    current_position = [current_x, current_y, current_theta]
                    linear_vel, angular_vel = self.calculate_base_velocity(
                        current_position, 
                        target_base_config, 
                        distance_error, 
                        angle_to_target, 
                        final_angle_error, 
                        base_position_reached
                    )
                    
                    # Use the controller to calculate joint velocities
                    joint_velocities, joint_errors, joints_reached = self.calculate_joint_velocities(
                        current_joint_positions, target_joint_positions
                    )
                    
                    # Check if we've reached target for both base and joints
                    target_reached = base_position_reached and base_angle_reached and joints_reached
                    
                    if target_reached:
                        rospy.loginfo(f"Waypoint {i+1} reached completely")
                        break
                    
                    # Send velocity commands
                    self.send_velocity_commands(linear_vel, angular_vel, joint_velocities)
                    
                    # Log progress every second
                    if int(elapsed * 2) % 2 == 0:  # Every ~0.5 seconds
                        progress = int(100 * elapsed / waypoint_duration)
                        rospy.loginfo(f"Waypoint {i+1} progress: {progress}%, " 
                                    f"base error: {distance_error:.3f}m, {np.degrees(final_angle_error):.1f}°, "
                                    f"joint errors: {np.mean(np.abs(joint_errors)):.3f}")
                    
                    # Control loop rate
                    self.rate.sleep()
                
                # Stop robot at the end of each waypoint
                self.stop_all_motion()
                
                # Give robot a moment to settle
                rospy.sleep(0.2)
            
            # Final verification that we reached the last point
            try:
                # Get final joint positions
                final_joint_positions = get_current_planning_joints()
                if final_joint_positions is None:
                    rospy.logwarn("Could not get final joint positions")
                    return False
                
                # Calculate final joint errors
                final_joint_errors = [abs(final_joint_positions[j] - arm_points[-1][j]) for j in range(len(final_joint_positions))]
                avg_joint_error = sum(final_joint_errors) / len(final_joint_errors)
                
                # Get final base position
                transform = tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
                final_x = transform.transform.translation.x
                final_y = transform.transform.translation.y
                
                q = transform.transform.rotation
                quaternion = [q.x, q.y, q.z, q.w]
                _, _, final_theta = tf.transformations.euler_from_quaternion(quaternion)
                
                final_base_pos_error = np.sqrt((final_x - base_configs[-1][0])**2 + (final_y - base_configs[-1][1])**2)
                
                final_base_angle_error = abs(final_theta - base_configs[-1][2])
                while final_base_angle_error > np.pi:
                    final_base_angle_error -= 2*np.pi
                final_base_angle_error = abs(final_base_angle_error)
                
                rospy.loginfo(f"Trajectory complete - Final errors: joint avg={avg_joint_error:.4f}, "
                            f"base position={final_base_pos_error:.4f}m, "
                            f"base angle={final_base_angle_error:.4f}rad")
                
                # Return success if errors are acceptable
                return (avg_joint_error < 0.05 and 
                        final_base_pos_error < 0.1 and 
                        final_base_angle_error < 0.1)
                
            except Exception as e:
                rospy.logwarn(f"Could not verify final position: {e}")
                # If we can't verify, assume success
                return True
                
        except Exception as e:
            rospy.logerr(f"Error in whole body trajectory following: {e}")
            
            # Ensure the robot stops moving
            try:
                # Stop all motion
                self.stop_all_motion()
            except:  # noqa: E722
                pass
                
            import traceback
            rospy.logerr(traceback.format_exc())
            return False 