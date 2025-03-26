import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import tf.transformations


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