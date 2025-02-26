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
import tf2_ros
import tf2_geometry_msgs
import time


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

        # Add TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
        """
        Initialize VAMP motion planner with 8-DOF configuration and collision settings.

        This function sets up the planning environment and configures the motion planner
        with appropriate parameters for collision avoidance.
        """
        try:
            # Create a new environment
            self.env = vamp.Environment()

            # Configure robot and planner with custom settings
            (
                self.vamp_module,
                self.planner_func,
                self.plan_settings,
                self.simp_settings,
            ) = vamp.configure_robot_and_planner_with_kwargs(
                "fetch",  # Robot name
                "rrtc",  # Planner algorithm (Rapidly-exploring Random Tree Connect)
                sampler_name="halton",  # Use Halton sampler for better coverage
                radius=0.0001,  # Connection radius
                max_iters=5000,  # Maximum planning iterations
                timeout_us=3000000,  # Timeout in microseconds (3 seconds)
                collision_check_res=0.01,  # Collision checking resolution
            )

            # Initialize the sampler
            self.sampler = self.vamp_module.halton()
            self.sampler.skip(0)  # Skip initial samples if needed

            rospy.loginfo("VAMP planner initialized with collision avoidance settings")

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
        self, points, frame_id="map", filter_radius=0.02, filter_cull=True
    ):
        """
        Add a pointcloud as collision constraint from already loaded point data.

        Args:
            points: List of [x,y,z] points or numpy array of shape (N,3)
            frame_id: The frame ID that the point cloud is defined in
            filter_radius: Filter radius for pointcloud filtering
            filter_cull: Whether to cull pointcloud around robot

        Returns:
            float: Time taken to process and add the point cloud or -1 if error
        """
        start_time = time.time()
        rospy.loginfo(f"Processing point cloud with {len(points)} points...")

        try:
            # Check if transformation is needed
            if frame_id != "base_link":
                rospy.loginfo(f"Transforming points from {frame_id} to base_link...")

                # Wait for transform to be available
                try:
                    self.tf_buffer.lookup_transform(
                        "base_link", frame_id, rospy.Time(0), rospy.Duration(5.0)
                    )
                    points = self._transform_points(points, frame_id, "base_link")

                    # Check if transformation was successful
                    if len(points) == 0:
                        rospy.logerr("Transformation resulted in zero points")
                        return -1

                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logerr(f"Transform lookup failed: {e}")
                    return -1

            # Process point cloud for collision avoidance
            rospy.loginfo(f"Processing {len(points)} points for collision checking...")

            # Get robot-specific parameters
            filter_origin = [0.0, 0.0, 0.0]  # Default filter origin (robot base)
            filter_cull_radius = 1.4  # Default max reach radius for Fetch

            # Define bounding box for filtering
            bbox_lo = np.array(filter_origin) - filter_cull_radius
            bbox_hi = np.array(filter_origin) + filter_cull_radius

            # Filter the point cloud
            filtered_pc, filter_time = self._filter_pointcloud(
                points.tolist() if isinstance(points, np.ndarray) else points,
                filter_radius,
                filter_cull_radius,
                filter_origin,
                bbox_lo,
                bbox_hi,
                filter_cull,
            )

            # Check if filtering was successful
            if len(filtered_pc) == 0:
                rospy.logwarn(
                    "Filtering resulted in zero points, using simple sphere obstacle instead"
                )
                # Add a simple sphere as a fallback
                sphere = vamp.Sphere([0.5, 0.0, 0.5], 0.2)
                sphere.name = "fallback_obstacle"
                self.env.add_sphere(sphere)
                processing_time = time.time() - start_time
                return processing_time

            # Define robot-specific radius parameters
            r_min, r_max = 0.03, 0.08  # Min/max sphere radius for Fetch robot
            point_radius = 0.01  # Default point radius for collision checking

            # Add the filtered point cloud to the environment
            try:
                build_time = self.env.add_pointcloud(
                    filtered_pc, r_min, r_max, point_radius
                )

                processing_time = time.time() - start_time

                rospy.loginfo(
                    f"""Pointcloud processing stats:
                    Original size: {len(points)}
                    Filtered size: {len(filtered_pc)}
                    Filter time: {filter_time * 1e-6:.3f}ms
                    Build time: {build_time * 1e-6:.3f}ms
                    Total processing time: {processing_time:.3f}s"""
                )

                return processing_time

            except Exception as e:
                rospy.logerr(f"Failed to add point cloud to environment: {e}")
                # Add a simple obstacle as fallback
                rospy.logwarn("Adding fallback obstacles instead")
                self.add_box(
                    position=[0.5, 0.0, 0.5],
                    half_extents=[0.2, 0.2, 0.2],
                    orientation_euler_xyz=[0, 0, 0],
                    name="fallback_obstacle",
                )
                return -1

        except Exception as e:
            rospy.logerr(f"Failed to add pointcloud: {str(e)}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return -1

    def _filter_pointcloud(
        self,
        points,
        filter_radius,
        filter_cull_radius,
        filter_origin,
        bbox_lo,
        bbox_hi,
        filter_cull=True,
    ):
        """
        Filter point cloud to remove redundant points and apply culling if needed.

        Args:
            points: List of 3D points
            filter_radius: Radius to filter nearby points
            filter_cull_radius: Maximum distance from origin to keep points
            filter_origin: Origin point for distance calculations
            bbox_lo: Lower corner of bounding box
            bbox_hi: Upper corner of bounding box
            filter_cull: Whether to apply distance-based culling

        Returns:
            tuple: (filtered_points, filter_time_microseconds)
        """
        start_time = time.time()

        try:
            # Convert to numpy array if it's not already
            points_np = (
                np.array(points, dtype=np.float64)
                if not isinstance(points, np.ndarray)
                else points.astype(np.float64)
            )

            # Make sure points are valid
            if points_np.size == 0 or points_np.shape[1] != 3:
                rospy.logwarn(f"Invalid point cloud shape: {points_np.shape}")
                return [], int((time.time() - start_time) * 1e6)

            # Handle potential NaN or infinite values
            mask = np.isfinite(points_np).all(axis=1)
            points_np = points_np[mask]

            # Manually downsample instead of using Open3D's voxel_down_sample
            # Create a voxel grid
            voxel_indices = np.floor(points_np / filter_radius).astype(int)

            # Use a dictionary to keep track of points per voxel
            voxel_dict = {}
            for i, idx in enumerate(voxel_indices):
                idx_tuple = tuple(idx)
                if idx_tuple not in voxel_dict:
                    voxel_dict[idx_tuple] = i

            # Get indices of downsampled points
            downsampled_indices = list(voxel_dict.values())
            points_filtered = points_np[downsampled_indices]

            # If culling is enabled, remove points outside the robot's reach
            if filter_cull and len(points_filtered) > 0:
                # Keep points within bounding box
                origin = np.array(filter_origin)

                mask_x = (points_filtered[:, 0] >= bbox_lo[0]) & (
                    points_filtered[:, 0] <= bbox_hi[0]
                )
                mask_y = (points_filtered[:, 1] >= bbox_lo[1]) & (
                    points_filtered[:, 1] <= bbox_hi[1]
                )
                mask_z = (points_filtered[:, 2] >= bbox_lo[2]) & (
                    points_filtered[:, 2] <= bbox_hi[2]
                )
                mask_bbox = mask_x & mask_y & mask_z

                # Apply distance-based filtering
                distances = np.linalg.norm(points_filtered - origin, axis=1)
                mask_distance = distances <= filter_cull_radius

                # Combine masks
                mask = mask_bbox & mask_distance
                points_filtered = points_filtered[mask]

            rospy.loginfo(
                f"Filtered point cloud from {len(points_np)} to {len(points_filtered)} points"
            )

            filter_time = int(
                (time.time() - start_time) * 1e6
            )  # Convert to microseconds
            return points_filtered.tolist(), filter_time

        except Exception as e:
            rospy.logerr(f"Error in point cloud filtering: {str(e)}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return [], int((time.time() - start_time) * 1e6)

    def _transform_points(self, points, source_frame, target_frame):
        """
        Transform a set of points from source frame to target frame using TF2.

        This function handles transformations between different coordinate frames:
        - world: The global coordinate system (used by point clouds)
        - map: The SLAM-generated map coordinate system
        - base_link: The robot's base coordinate system (used for motion planning)

        Args:
            points: Nx3 array of points
            source_frame: Source frame ID (e.g., "world", "map")
            target_frame: Target frame ID (e.g., "base_link")

        Returns:
            Nx3 array of transformed points
        """
        transformed_points = []

        # Log transformation details
        rospy.loginfo(
            f"Transforming {len(points)} points from '{source_frame}' to '{target_frame}'"
        )

        # Get the latest transform
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(5.0)
            )
            rospy.loginfo(f"Found transform from '{source_frame}' to '{target_frame}'")

            # Log transform information for debugging
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            rospy.loginfo(
                f"Translation: [{translation.x}, {translation.y}, {translation.z}]"
            )
            rospy.loginfo(
                f"Rotation: [{rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}]"
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Failed to lookup transform: {e}")
            return points  # Return original points if transform fails

        # Batch processing to improve performance
        batch_size = 1000  # Process points in batches
        num_batches = (len(points) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(points))
            batch_points = points[start_idx:end_idx]

            batch_transformed = []
            for point in batch_points:
                p = tf2_geometry_msgs.PointStamped()
                p.header.frame_id = source_frame
                p.header.stamp = rospy.Time(0)
                p.point.x = float(point[0])
                p.point.y = float(point[1])
                p.point.z = float(point[2])

                try:
                    # Apply the transformation
                    transformed_p = self.tf_buffer.transform(
                        p, target_frame, rospy.Duration(0.1)
                    )
                    batch_transformed.append(
                        [
                            transformed_p.point.x,
                            transformed_p.point.y,
                            transformed_p.point.z,
                        ]
                    )
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logwarn(f"Transform failed for point {point}: {e}")
                    # Skip this point
                    continue

            transformed_points.extend(batch_transformed)

            # Log progress for large point clouds
            if num_batches > 1:
                rospy.loginfo(
                    f"Transformed batch {batch_idx+1}/{num_batches} ({len(batch_transformed)} points)"
                )

        rospy.loginfo(
            f"Successfully transformed {len(transformed_points)} out of {len(points)} points"
        )

        return np.array(transformed_points)

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
