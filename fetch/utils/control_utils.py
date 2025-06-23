import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class HeadController:
    """
    A class to control the Fetch robot's head.
    """
    def __init__(self):
        """
        Initializes the HeadController.
        """
        self.head_traj_client = actionlib.SimpleActionClient(
            "head_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        rospy.loginfo("Waiting for head controller action server...")
        self.head_traj_client.wait_for_server()
        rospy.loginfo("Head controller action server found.")

        self.head_joint_names = ["head_pan_joint", "head_tilt_joint"]

    def move_head(self, pan, tilt, duration=1.0):
        """
        Moves the robot's head to a given pan and tilt position.

        Args:
            pan (float): The target pan position for the head.
            tilt (float): The target tilt position for the head.
            duration (float): The duration of the movement in seconds.
        """
        goal = FollowJointTrajectoryGoal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.head_joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = rospy.Duration(duration)
        
        trajectory.points.append(point)
        goal.trajectory = trajectory

        self.head_traj_client.send_goal(goal)
        rospy.loginfo(f"Sent head goal: pan={pan}, tilt={tilt}") 