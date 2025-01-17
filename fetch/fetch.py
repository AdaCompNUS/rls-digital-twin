import rospy
import numpy as np
import actionlib
from geometry_msgs.msg import Twist, PoseStamped
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, GripperCommandAction, GripperCommandGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class Fetch:
    """
    Core class for controlling the Fetch robot.
    """
    def __init__(self):
        """Initialize the Fetch robot interface."""
        try:
            rospy.init_node('fetch_controller', anonymous=True)
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")
        
        # Publisher for base movement commands
        self._base_publisher = rospy.Publisher(
            "/base_controller/command", 
            Twist, 
            queue_size=2
        )
        
        # Control parameters
        self.control_rate = 10
        self.rate = rospy.Rate(self.control_rate)
        
        # Movement limits
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s

        # Initialize action clients
        self.arm_traj_client = actionlib.SimpleActionClient(
            "arm_controller/follow_joint_trajectory", 
            FollowJointTrajectoryAction
        )
        self.arm_traj_client.wait_for_server()

        self.gripper_client = actionlib.SimpleActionClient(
            "gripper_controller/gripper_action", 
            GripperCommandAction
        )
        self.gripper_client.wait_for_server()

        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", 
            MoveBaseAction
        )
        self.move_base_client.wait_for_server()

        # Initialize torso action client
        self.torso_client = actionlib.SimpleActionClient(
            "torso_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction
        )
        self.torso_client.wait_for_server()

        # End effector pose publisher
        self.ee_pose_publisher = rospy.Publisher(
            "/arm_controller/cartesian_pose_vel_controller/command",
            PoseStamped,
            queue_size=10
        )

        # Define joint names
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint"
        ]
        
        self.torso_joint_names = ["torso_lift_joint"]

    def move_base(self, linear_x, angular_z):
        """
        Move the robot base with specified linear and angular velocities.
        
        Args:
            linear_x (float): Forward/backward velocity (-1.0 to 1.0)
            angular_z (float): Rotational velocity (-1.0 to 1.0)
        """
        # Clip velocities to safe ranges
        linear_x = np.clip(linear_x, -1.0, 1.0) * self.max_linear_speed
        angular_z = np.clip(angular_z, -1.0, 1.0) * self.max_angular_speed
        
        # Create and publish movement command
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        
        self._base_publisher.publish(twist)
    
    def stop_base(self):
        """Stop all base movement."""
        twist = Twist()
        self._base_publisher.publish(twist)

    def send_target_position(self, position, orientation):
        """
        Send the robot base to a target position in the map frame.
        
        Args:
            position (list): [x, y, z] target position
            orientation (list): [x, y, z, w] target orientation as quaternion
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # Set position
        goal.target_pose.pose.position.x = position[0]
        goal.target_pose.pose.position.y = position[1]
        goal.target_pose.pose.position.z = position[2]
        
        # Set orientation
        goal.target_pose.pose.orientation.x = orientation[0]
        goal.target_pose.pose.orientation.y = orientation[1]
        goal.target_pose.pose.orientation.z = orientation[2]
        goal.target_pose.pose.orientation.w = orientation[3]
        
        # Send goal and wait for result
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        
        return self.move_base_client.get_result()

    def send_joint_values(self, joint_positions, duration=5.0):
        """
        Move the arm to specified joint positions.
        
        Args:
            joint_positions (list): List of 7 joint angles in radians
            duration (float): Time to execute the movement in seconds
        """
        if len(joint_positions) != 7:
            raise ValueError("Expected 7 joint positions")
            
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.arm_joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points.append(point)
        
        self.arm_traj_client.send_goal(goal)
        self.arm_traj_client.wait_for_result()
        
        return self.arm_traj_client.get_result()

    def set_torso_height(self, height, duration=5.0):
        """
        Set the torso height.
        
        Args:
            height (float): Target height in meters (0.0 to 0.4)
            duration (float): Time to execute the movement in seconds
        """
        # Clip height to safe range
        height = np.clip(height, 0.0, 0.4)
        
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.torso_joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [height]
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points.append(point)
        
        self.torso_client.send_goal(goal)
        self.torso_client.wait_for_result()
        
        return self.torso_client.get_result()

    def send_end_effector_pose(self, position, orientation):
        """
        Move the end effector to a target pose in the base_link frame.
        
        Args:
            position (list): [x, y, z] target position
            orientation (list): [x, y, z, w] target orientation as quaternion
        """
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "base_link"
        pose_msg.header.stamp = rospy.Time.now()
        
        # Set position
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        
        # Set orientation
        pose_msg.pose.orientation.x = orientation[0]
        pose_msg.pose.orientation.y = orientation[1]
        pose_msg.pose.orientation.z = orientation[2]
        pose_msg.pose.orientation.w = orientation[3]
        
        self.ee_pose_publisher.publish(pose_msg)

    def control_gripper(self, position, max_effort=100):
        """
        Control the gripper position.
        
        Args:
            position (float): 0.0 (closed) to 1.0 (open)
            max_effort (float): Maximum effort to apply
        """
        goal = GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result()
        
        return self.gripper_client.get_result()