import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
import tf.transformations
import numpy as np
import math

class HeadController:
    """
    A class to control the Fetch robot's head.
    """
    def __init__(self, tf_buffer=None):
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

        if tf_buffer:
            self.tf_buffer = tf_buffer
        else:
            rospy.loginfo("Creating new TF buffer for HeadController")
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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

    def point_head_at(self, target_point, frame_id, duration=1.0):
        """
        Points the robot's head towards a target point.

        This method calculates the necessary pan and tilt angles to align the head
        with a target point specified in any coordinate frame.

        Args:
            target_point (list): [x, y, z] coordinates of the target.
            frame_id (str): The TF frame ID of the target_point.
            duration (float): The duration of the head movement in seconds.
        """
        try:
            # Get the transform from the target frame to the head pan frame
            transform_stamped = self.tf_buffer.lookup_transform(
                "head_pan_link",  # Target frame for the point
                frame_id,  # Source frame of the point
                rospy.Time(0),
                rospy.Duration(1.0),
            )

            # Manually transform the point to avoid tf2_geometry_msgs dependency
            translation = transform_stamped.transform.translation
            rotation = transform_stamped.transform.rotation

            # Create a 4x4 transformation matrix from the transform
            trans_matrix = tf.transformations.quaternion_matrix(
                [rotation.x, rotation.y, rotation.z, rotation.w]
            )
            trans_matrix[0, 3] = translation.x
            trans_matrix[1, 3] = translation.y
            trans_matrix[2, 3] = translation.z

            # Represent the target point as a homogeneous vector
            point_homogeneous = np.array(target_point + [1.0])

            # Apply the transformation
            transformed_point_homogeneous = np.dot(trans_matrix, point_homogeneous)

            # Extract coordinates from the transformed point
            x = transformed_point_homogeneous[0]
            y = transformed_point_homogeneous[1]
            z = transformed_point_homogeneous[2]

            # Calculate pan and tilt
            pan = math.atan2(y, x)
            tilt = math.atan2(-z, math.sqrt(x**2 + y**2))

            # Command the head to the new position
            self.move_head(pan, tilt, duration)
            rospy.loginfo(
                f"Calculated and sent head goal: pan={pan:.3f}, tilt={tilt:.3f}"
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"TF transform failed in point_head_at: {e}") 