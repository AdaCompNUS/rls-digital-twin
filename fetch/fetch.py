import vamp._core
import rospy
import numpy as np
import vamp
import actionlib
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class Fetch:
    """
    Core class for controlling the Fetch robot.
    """

    def __init__(self):
        """Initialize the Fetch robot interface."""
        try:
            rospy.init_node("fetch_controller", anonymous=True)
        except rospy.exceptions.ROSException:
            print("Node has already been initialized, do nothing")

        # Add joint states subscriber
        self.joint_states = None
        self.joint_state_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self._joint_states_callback
        )

        # Wait for first joint states message
        rospy.loginfo("Waiting for joint states...")
        while self.joint_states is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Received joint states")

        # Publisher for base movement commands
        self._base_publisher = rospy.Publisher(
            "/base_controller/command", Twist, queue_size=2
        )

        # Control parameters
        self.control_rate = 10
        self.rate = rospy.Rate(self.control_rate)

        # Movement limits
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s

        # Initialize action clients
        self.arm_traj_client = actionlib.SimpleActionClient(
            "arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.arm_traj_client.wait_for_server()

        self.gripper_client = actionlib.SimpleActionClient(
            "gripper_controller/gripper_action", GripperCommandAction
        )
        self.gripper_client.wait_for_server()

        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction
        )
        self.move_base_client.wait_for_server()

        # End effector pose publisher
        self.ee_pose_publisher = rospy.Publisher(
            "/arm_controller/cartesian_pose_vel_controller/command",
            PoseStamped,
            queue_size=10,
        )

        # Initialize torso action client
        self.torso_client = actionlib.SimpleActionClient(
            "torso_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.torso_client.wait_for_server()

        # Define joint names (8-DOF for planning)
        self.planning_joint_names = [
            "torso_lift_joint",  # First joint is torso
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]

        # Initialize VAMP planner
        self._init_vamp_planner()

    def _joint_states_callback(self, msg):
        """Callback for joint states messages."""
        self.joint_states = msg

    def _init_vamp_planner(self):
        """Initialize VAMP motion planner with 8-DOF configuration."""
        try:
            self.env = vamp.Environment()
            (
                self.vamp_module,
                self.planner_func,
                self.plan_settings,
                self.simp_settings,
            ) = vamp.configure_robot_and_planner_with_kwargs(
                "fetch", "rrtc", sampler_name="halton", radius=0.0001
            )

            self.sampler = self.vamp_module.halton()
            self.sampler.skip(0)

            rospy.loginfo("VAMP planner initialized for 8-DOF planning")

        except Exception as e:
            rospy.logerr(f"Failed to initialize VAMP planner: {str(e)}")
            raise

    def get_current_planning_joints(self):
        """Get current joint positions for planning (8-DOF including torso)."""
        if self.joint_states is None:
            return None

        joint_dict = dict(zip(self.joint_states.name, self.joint_states.position))

        positions = []
        for joint_name in self.planning_joint_names:
            if joint_name in joint_dict:
                positions.append(joint_dict[joint_name])
            else:
                rospy.logerr(f"Joint {joint_name} not found in joint states")
                return None

        return positions

    def _plan_path_with_vamp(self, current_joints, target_joints):
        """Plan a path using VAMP motion planner for 8-DOF configuration."""
        try:
            current_joints = np.array(current_joints, dtype=np.float64)
            target_joints = np.array(target_joints, dtype=np.float64)

            rospy.loginfo("Planning with VAMP (8-DOF):")
            rospy.loginfo(f"Start config values: {current_joints}")
            rospy.loginfo(f"Goal config values: {target_joints}")

            if len(current_joints) != 8 or len(target_joints) != 8:
                rospy.logerr(
                    f"Invalid joint dimensions. Expected 8, got {len(current_joints)} and {len(target_joints)}"
                )
                return None

            result = self.planner_func(
                current_joints,
                target_joints,
                self.env,
                self.plan_settings,
                self.sampler,
            )

            if result.solved:
                rospy.loginfo("Path planning succeeded!")

                # Get planning statistics
                simple = self.vamp_module.simplify(
                    result.path, self.env, self.simp_settings, self.sampler
                )

                stats = vamp.results_to_dict(result, simple)
                rospy.loginfo(
                    f"""Planning statistics:
                    Planning Time: {stats['planning_time'].microseconds:8d}μs
                    Simplify Time: {stats['simplification_time'].microseconds:8d}μs
                    Total Time: {stats['total_time'].microseconds:8d}μs
                    Planning Iters: {stats['planning_iterations']}
                    Graph States: {stats['planning_graph_size']}
                    Path Length:
                        Initial: {stats['initial_path_cost']:5.3f}
                        Simplified: {stats['simplified_path_cost']:5.3f}"""
                )

                # Interpolate path
                simple.path.interpolate(self.vamp_module.resolution())

                # Convert path to trajectory points
                trajectory_points = []
                for i in range(len(simple.path)):
                    point = simple.path[i]
                    if isinstance(point, list):
                        trajectory_points.append(point)
                    elif isinstance(point, np.ndarray):
                        trajectory_points.append(point.tolist())
                    else:
                        trajectory_points.append(point.to_list())

                rospy.loginfo(f"Generated {len(trajectory_points)} trajectory points")
                return trajectory_points

            else:
                rospy.logwarn(f"Path planning failed. Iterations: {result.iterations}")
                return None

        except Exception as e:
            rospy.logerr(f"VAMP planning error: {str(e)}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return None

    def add_sphere(self, position, radius, name=None):
        """
        Add a sphere constraint to the environment.

        Args:
            position: [x, y, z] center position
            radius: sphere radius
            name: optional name for the sphere
        """
        try:
            position = list(position)  # Convert to list to ensure correct type
            sphere = vamp.Sphere(position, radius)
            if name:
                sphere.name = name
            self.env.add_sphere(sphere)
            rospy.loginfo(f"Added sphere constraint at {position} with radius {radius}")
        except Exception as e:
            rospy.logerr(f"Failed to add sphere constraint: {str(e)}")

    def add_box(self, position, half_extents, orientation_euler_xyz, name=None):
        """
        Add a box constraint to the environment.

        Args:
            position: [x, y, z] center position
            half_extents: [x, y, z] half-lengths in each dimension
            orientation_euler_xyz: [roll, pitch, yaw] orientation in radians
            name: optional name for the box
        """
        try:
            # Convert numpy arrays to lists to ensure correct type
            position = list(position)
            half_extents = list(half_extents)
            orientation_euler_xyz = list(orientation_euler_xyz)

            cuboid = vamp.Cuboid(position, orientation_euler_xyz, half_extents)
            if name:
                cuboid.name = name
            self.env.add_cuboid(cuboid)
            rospy.loginfo(
                f"Added box constraint at {position} with half_extents {half_extents}"
            )
        except Exception as e:
            rospy.logerr(f"Failed to add box constraint: {str(e)}")

    def add_capsule(
        self,
        center=None,
        endpoint1=None,
        endpoint2=None,
        radius=None,
        length=None,
        orientation_euler_xyz=None,
        name=None,
    ):
        """
        Add a capsule constraint to the environment.

        This method supports two ways of defining a capsule:
        1. Using center, orientation, radius, and length
        2. Using two endpoints and radius

        Args:
            center: [x, y, z] center position (for center-based definition)
            endpoint1: [x, y, z] first endpoint (for endpoint-based definition)
            endpoint2: [x, y, z] second endpoint (for endpoint-based definition)
            radius: capsule radius
            length: capsule length (for center-based definition)
            orientation_euler_xyz: [roll, pitch, yaw] orientation in radians (for center-based definition)
            name: optional name for the capsule
        """
        try:
            # Check which capsule definition method to use
            if (
                center is not None
                and orientation_euler_xyz is not None
                and radius is not None
                and length is not None
            ):
                # Method 1: Using center, orientation, radius, and length
                center = list(center)  # Convert to list to ensure correct type
                orientation_euler_xyz = list(orientation_euler_xyz)

                capsule = vamp.Capsule(center, orientation_euler_xyz, radius, length)
                rospy.loginfo(
                    f"Added capsule constraint at {center} with radius {radius} and length {length}"
                )

            elif endpoint1 is not None and endpoint2 is not None and radius is not None:
                # Method 2: Using two endpoints and radius
                endpoint1 = list(endpoint1)  # Convert to list to ensure correct type
                endpoint2 = list(endpoint2)

                capsule = vamp.Capsule(endpoint1, endpoint2, radius)
                rospy.loginfo(
                    f"Added capsule constraint from {endpoint1} to {endpoint2} with radius {radius}"
                )

            else:
                rospy.logerr(
                    "Invalid parameters for add_capsule. Use either center+orientation+radius+length or endpoint1+endpoint2+radius"
                )
                return

            # Set name if provided
            if name:
                capsule.name = name

            # Add capsule to environment
            self.env.add_capsule(capsule)

        except Exception as e:
            rospy.logerr(f"Failed to add capsule constraint: {str(e)}")
            import traceback

            rospy.logerr(traceback.format_exc())

    def add_cylinder(self, position, radius, length, orientation_euler_xyz, name=None):
        """
        Add a cylinder constraint to the environment.

        Args:
            position: [x, y, z] center position
            radius: cylinder radius
            length: cylinder length
            orientation_euler_xyz: [roll, pitch, yaw] orientation in radians
            name: optional name for the cylinder
        """
        try:
            # Convert numpy arrays to lists to ensure correct type
            position = list(position)
            orientation_euler_xyz = list(orientation_euler_xyz)

            cylinder = vamp._core.Cylinder(
                position, orientation_euler_xyz, radius, length
            )
            if name:
                cylinder.name = name
            self.env.add_capsule(cylinder)
            rospy.loginfo(f"Added cylinder constraint at {position}")
        except Exception as e:
            rospy.logerr(f"Failed to add cylinder constraint: {str(e)}")

    def add_pointcloud(
        self, points, samples_per_object=10000, filter_radius=0.02, filter_cull=True
    ):
        """
        Add a pointcloud as collision constraint.

        Args:
            points: Nx3 array of points
            samples_per_object: Number of samples per object for pointcloud
            filter_radius: Filter radius for pointcloud filtering
            filter_cull: Whether to cull pointcloud around robot by maximum distance
        """
        try:
            from vamp import pointcloud as vpc

            # Process pointcloud
            (
                new_env,
                original_pc,
                filtered_pc,
                filter_time,
                build_time,
            ) = vpc.points_to_pointcloud(
                "fetch",  # Robot name
                points,
                samples_per_object,
                filter_radius,
                filter_cull,
            )

            rospy.loginfo(
                f"""Pointcloud processing stats:
                Original size: {len(original_pc)}
                Filtered size: {len(filtered_pc)}
                Filter time: {filter_time * 1e-6:.3f}ms
                Build time: {build_time * 1e-6:.3f}ms"""
            )

            # Add pointcloud environment to existing environment
            for obj in new_env.objects():
                self.env.add_object(obj)

        except Exception as e:
            rospy.logerr(f"Failed to add pointcloud constraint: {str(e)}")

    def send_joint_values(self, target_joints, duration=5.0):
        """
        Move the arm and torso to specified joint positions using VAMP planning.

        Args:
            target_joints: List of 8 joint positions [torso + 7 arm joints]
            duration: Time to execute trajectory
        """
        try:
            if len(target_joints) != 8:
                raise ValueError("Expected 8 joint positions [torso + 7 arm joints]")

            # Get current joints
            current_joints = self.get_current_planning_joints()
            if current_joints is None:
                rospy.logerr("Failed to get current joint positions")
                return None

            rospy.loginfo(f"Planning motion to target configuration: {target_joints}")

            # Plan path using VAMP
            trajectory_points = self._plan_path_with_vamp(current_joints, target_joints)

            if trajectory_points is None:
                rospy.logerr("Failed to plan path with VAMP")
                return None

            # Split trajectory points for torso and arm
            torso_points = []  # First joint
            arm_points = []  # Last 7 joints

            # Extract points for each controller
            for point in trajectory_points:
                torso_points.append([point[0]])  # First joint (torso)
                arm_points.append(point[1:])  # Remaining joints (arm)

            # Create torso trajectory
            torso_goal = FollowJointTrajectoryGoal()
            torso_goal.trajectory = JointTrajectory()
            torso_goal.trajectory.joint_names = ["torso_lift_joint"]

            # Create arm trajectory
            arm_goal = FollowJointTrajectoryGoal()
            arm_goal.trajectory = JointTrajectory()
            arm_goal.trajectory.joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",
                "elbow_flex_joint",
                "forearm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ]

            # Add trajectory points
            point_duration = duration / len(trajectory_points)
            for i in range(len(trajectory_points)):
                # Torso trajectory point
                torso_point = JointTrajectoryPoint()
                torso_point.positions = torso_points[i]
                torso_point.velocities = [0.0]
                torso_point.accelerations = [0.0]
                torso_point.time_from_start = rospy.Duration(point_duration * (i + 1))
                torso_goal.trajectory.points.append(torso_point)

                # Arm trajectory point
                arm_point = JointTrajectoryPoint()
                arm_point.positions = arm_points[i]
                arm_point.velocities = [0.0] * 7
                arm_point.accelerations = [0.0] * 7
                arm_point.time_from_start = rospy.Duration(point_duration * (i + 1))
                arm_goal.trajectory.points.append(arm_point)

            # Execute trajectories
            rospy.loginfo(
                f"Executing trajectory with {len(trajectory_points)} points..."
            )

            # Send goals to both controllers
            self.torso_client.send_goal(torso_goal)
            self.arm_traj_client.send_goal(arm_goal)

            # Wait for completion
            torso_success = self.torso_client.wait_for_result(
                rospy.Duration(duration + 5.0)
            )
            arm_success = self.arm_traj_client.wait_for_result(
                rospy.Duration(duration + 5.0)
            )

            if torso_success and arm_success:
                rospy.loginfo("Trajectory execution completed successfully")
                return self.arm_traj_client.get_result()
            else:
                rospy.logwarn("Trajectory execution failed or timed out")
                return None

        except Exception as e:
            rospy.logerr(f"Error in send_joint_values: {str(e)}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return None

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
