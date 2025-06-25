import vamp._core
import rospy
import numpy as np
import vamp
import actionlib
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped, Pose
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf2_ros
import time
import fetch.utils.transform_utils as transform_utils
from fetch.utils.ik_solver_utils import WholeBodyIKSolver
import tf.transformations
from fetch.utils.control_utils import HeadController
from std_msgs.msg import Bool


class Fetch:
    """
    Core class for controlling the Fetch robot.
    """

    def __init__(self, urdf_path="resources/fetch_ext/fetch ext.urdf", costmap_path="resources/costmap.npz"):
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

        # Initialize head controller
        self.head_controller = HeadController(self.tf_buffer)

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

        self.planner = "rrtc"  # ["rrtc", "fcit", "prm"]

        # Initialize VAMP planner
        self._init_vamp_planner()

        # Publisher for whole body trajectory execution
        # This will communicate with the C++ WholeBodyController node
        self.arm_path_pub = rospy.Publisher(
            "/fetch_whole_body_controller/arm_path", JointTrajectory, queue_size=1
        )
        self.base_path_pub = rospy.Publisher(
            "/fetch_whole_body_controller/base_path", JointTrajectory, queue_size=1
        )

        # Subscriber for whole body execution completion
        self.execution_finished = False
        self.execution_finished_sub = rospy.Subscriber(
            "/fetch_whole_body_controller/execution_finished",
            Bool,
            self._execution_finished_callback,
        )

        # Initialize the whole-body IK solver
        self.ik_solver = WholeBodyIKSolver(urdf_path=urdf_path, costmap_path=costmap_path)

    def solve_whole_body_ik(
        self, target_pose, max_attempts=100, manipulation_radius=1.0
    ):
        """
        Solve whole-body inverse kinematics for a given target pose.

        This method attempts to find a valid (collision-free) configuration for the whole robot
        (base + arm) that places the end effector at the desired pose.

        Args:
            target_pose: Target end effector pose (geometry_msgs/Pose)
            max_attempts: Maximum number of sampling attempts
            manipulation_radius: Radius for base position sampling

        Returns:
            dict: Solution containing base and arm configuration, or None if no solution found
        """
        return self.ik_solver.solve(
            vamp_module=self.vamp_module,
            env=self.env,
            target_pose=target_pose,
            max_attempts=max_attempts,
            manipulation_radius=manipulation_radius,
            use_fixed_base=True,
        )

    def solve_ik(self, target_pose, frame_id='map', arm_seed=None, max_attempts=10):
        """
        Solve inverse kinematics for the arm, from base_link to the end-effector.
        The target pose can be provided in any frame, and will be transformed to base_link.

        Args:
            target_pose: Target end effector pose (geometry_msgs/Pose).
            frame_id: The TF frame of the target_pose. Defaults to 'map'.
            arm_seed (list, optional): An initial guess for the arm joint angles (8-DOF).
            max_attempts: Maximum number of sampling attempts.

        Returns:
            list: A list of 8 joint values for the arm if a solution is found, otherwise None.
        """
        rospy.loginfo(f"Solving arm-only IK for pose in '{frame_id}' frame.")

        target_pose_in_base_link = target_pose

        if frame_id != 'base_link':
            try:
                # Get the transform from the source frame to the base_link frame
                transform_stamped = self.tf_buffer.lookup_transform(
                    'base_link',      # Target frame
                    frame_id,         # Source frame
                    rospy.Time(),    # Get the latest transform
                    rospy.Duration(1.0) # Wait for up to 1 second
                )
                
                # Convert transform to a 4x4 matrix
                r = transform_stamped.transform.rotation
                t = transform_stamped.transform.translation
                T_base_link__frame = transform_utils.quaternion_matrix([r.x, r.y, r.z, r.w])
                T_base_link__frame[:3, 3] = [t.x, t.y, t.z]

                # Convert target_pose to a 4x4 matrix
                p = target_pose.position
                o = target_pose.orientation
                T_frame__ee = transform_utils.quaternion_matrix([o.x, o.y, o.z, o.w])
                T_frame__ee[:3, 3] = [p.x, p.y, p.z]
                
                # Transform the pose: T_base_link__ee = T_base_link__frame * T_frame__ee
                T_base_link__ee = np.dot(T_base_link__frame, T_frame__ee)

                # Convert the resulting matrix back to a Pose message
                pos_final = transform_utils.translation_from_matrix(T_base_link__ee)
                quat_final = transform_utils.quaternion_from_matrix(T_base_link__ee)
                
                target_pose_in_base_link = Pose()
                target_pose_in_base_link.position.x = pos_final[0]
                target_pose_in_base_link.position.y = pos_final[1]
                target_pose_in_base_link.position.z = pos_final[2]
                target_pose_in_base_link.orientation.x = quat_final[0]
                target_pose_in_base_link.orientation.y = quat_final[1]
                target_pose_in_base_link.orientation.z = quat_final[2]
                target_pose_in_base_link.orientation.w = quat_final[3]
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"Failed to transform pose from '{frame_id}' to 'base_link': {e}")
                return None
        
        return self.ik_solver.solve_arm_ik(
            target_pose=target_pose_in_base_link,
            arm_seed=arm_seed,
            max_attempts=max_attempts
        )

    def move_to_pose(self, target_pose, max_attempts=100, manipulation_radius=1.0):
        """
        Move the robot to place its end effector at the target pose.

        This method:
        1. Solves whole-body IK to find a valid goal configuration
        2. Plans a whole-body motion from current to goal configuration
        3. Executes the planned motion

        Args:
            target_pose: Target end effector pose (geometry_msgs/Pose)
            execution_time: Time for trajectory execution in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        # Get current configuration
        current_joints = self.get_current_planning_joints()
        if current_joints is None:
            rospy.logerr("Failed to get current joint positions")
            return False

        # Get current base position
        current_base = self.get_base_params()

        # Step 1: Solve whole-body IK
        rospy.loginfo("Step 1: Solving whole-body IK...")
        ik_solution = self.solve_whole_body_ik(target_pose, max_attempts=max_attempts, manipulation_radius=manipulation_radius)

        if ik_solution is None:
            rospy.logerr("Failed to find IK solution")
            return False

        goal_base = ik_solution["base_config"]
        goal_joints = ik_solution["arm_config"]

        rospy.loginfo("Found valid IK solution:")
        rospy.loginfo(f"Goal base: {[round(v, 3) for v in goal_base]}")
        rospy.loginfo(f"Goal joints: {[round(v, 3) for v in goal_joints]}")

        # Step 2: Plan whole-body motion
        rospy.loginfo("Step 2: Planning whole-body motion...")
        plan_result = self.plan_whole_body_motion(
            current_joints, goal_joints, list(current_base), goal_base
        )

        if not plan_result or not plan_result["success"]:
            rospy.logerr("Failed to plan whole-body motion")
            return False

        # Step 3: Execute the planned motion
        rospy.loginfo("Step 3: Executing whole-body motion...")
        execution_success = self.execute_whole_body_motion(
            plan_result["arm_path"], plan_result["base_configs"]
        )

        if not execution_success:
            rospy.logerr("Failed to execute whole-body motion")
            return False

        rospy.loginfo("Successfully moved to target pose")
        return True

    def get_end_effector_pose(self, joint_values=None, base_config=None):
        """
        Get the end effector pose for a given configuration.

        If joint_values and base_config are not provided, it uses the current robot state.

        Args:
            joint_values: Optional list of 8 joint values (torso + 7 arm joints)
            base_config: Optional [x, y, theta] base configuration

        Returns:
            pose: End effector pose (geometry_msgs/Pose) in world frame
        """
        from geometry_msgs.msg import Pose

        # Use current configuration if not provided
        if joint_values is None:
            joint_values = self.get_current_planning_joints()
            if joint_values is None:
                rospy.logerr("Failed to get current joint positions")
                return None

        if base_config is None:
            base_config = self.get_base_params()

        # Call the eefk function to get the end effector pose in robot frame
        ee_pos, ee_quat = self.vamp_module.eefk(joint_values)

        # Transform to world frame using the utility function
        world_pos, world_quat = transform_utils.transform_pose_to_world(
            [base_config[0], base_config[1], 0], base_config[2], ee_pos, ee_quat
        )

        # Create pose message
        pose = Pose()
        pose.position.x = world_pos[0]
        pose.position.y = world_pos[1]
        pose.position.z = world_pos[2]
        pose.orientation.x = world_quat[0]
        pose.orientation.y = world_quat[1]
        pose.orientation.z = world_quat[2]
        pose.orientation.w = world_quat[3]

        return pose

    def _joint_states_callback(self, msg):
        """Callback for joint states messages."""
        # Only save messages with length > 2 to filter out gripper messages
        if msg is not None and len(msg.name) > 3:
            self.joint_states = msg
        # else:
        #     self.joint_states = None
        #     rospy.logdebug(
        #         f"Ignoring joint state message with only {len(msg.name)} joints"
        #     )

    def _execution_finished_callback(self, msg):
        """Callback for the whole body execution finished signal."""
        if msg.data:
            self.execution_finished = True

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
                self.planner,  # Planner algorithm (Rapidly-exploring Random Tree Connect)
                sampler_name="halton",  # Use Halton sampler for better coverage
            )

            # Initialize the sampler
            self.sampler = self.vamp_module.halton()
            self.sampler.skip(0)  # Skip initial samples if needed

            rospy.loginfo("VAMP planner initialized with collision avoidance settings")

        except Exception as e:
            rospy.logerr(f"Failed to initialize VAMP planner: {str(e)}")
            raise

    def set_base_params(self, theta, x, y):
        """
        Set the base parameters for the Fetch robot.

        Args:
            theta (float): Base rotation around z-axis in radians
            x (float): Base x position in meters
            y (float): Base y position in meters
        """
        self.base_theta = theta
        self.base_x = x
        self.base_y = y

        # Update the base parameters in the VAMP module
        self.vamp_module.set_base_params(theta, x, y)

        rospy.loginfo(
            f"Set base parameters: theta={theta:.6f}, x={x:.6f}, y={y:.6f}"
        )
        return True

    def get_base_params(self, world_frame='map', robot_base_frame='base_link'):
        """
        Get the current base parameters from the ROS TF tree.
        This is now the single source of truth for the robot's current pose.
        Args:
            world_frame (str): The name of the fixed world frame (e.g., 'map').
            robot_base_frame (str): The name of the robot's base frame (e.g., 'base_link').
        Returns:
            tuple: (x, y, theta) current base parameters.
        """
        # Lookup the transform from the world frame to the robot's base
        transform = self.tf_buffer.lookup_transform(
            world_frame, robot_base_frame, rospy.Time(0), rospy.Duration(1.0)
        )
        trans = transform.transform.translation
        rot = transform.transform.rotation
        # Convert quaternion to Euler angles to get the yaw
        quaternion = [rot.x, rot.y, rot.z, rot.w]
        # Using numpy to get the yaw from a quaternion
        x, y, z, w = quaternion
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        
        # Return in the (x, y) order expected by set_base_params and VAMP
        return (trans.x, trans.y, yaw)

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
                rospy.logerr(
                    f"Joint {joint_name} not found in joint states, The current joints are: {joint_dict}"
                )
                return None

        return positions
    
    def get_camera_pose(self, camera_frame='head_camera_rgb_optical_frame', world_frame='map'):
        """
        Get the current pose of a specified camera directly in the world frame using TF.
        Args:
            camera_frame (str): The name of the camera's TF frame.
            world_frame (str): The name of the world's TF frame (e.g., 'map').
        Returns:
            numpy.ndarray: 4x4 transformation matrix representing the pose of the camera 
                        in the world frame.
        """
        from tf.transformations import quaternion_matrix
        
        rospy.loginfo(f"Attempting to get pose of '{camera_frame}' in frame '{world_frame}'")
        
        # Directly look up the transform from the world to the camera.
        # TF2 handles the entire chain: world -> base -> ... -> camera
        transform_stamped = self.tf_buffer.lookup_transform(
            world_frame,     # Target frame
            camera_frame,    # Source frame
            rospy.Time(),   # Get the latest transform
            rospy.Duration(1.0)  # Wait for up to 1 second
        )
        
        # Extract translation and rotation from the transform
        translation = transform_stamped.transform.translation
        rotation = transform_stamped.transform.rotation
        
        # Convert quaternion to rotation matrix
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = quaternion_matrix(quaternion)
        
        # Set the translation part of the matrix
        rotation_matrix[0, 3] = translation.x
        rotation_matrix[1, 3] = translation.y
        rotation_matrix[2, 3] = translation.z
        
        rospy.loginfo("Successfully retrieved camera pose as transformation matrix.")
        return rotation_matrix

    def _plan_path_with_vamp(self, current_joints, target_joints):
        """Plan a path using VAMP motion planner for 8-DOF configuration."""
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
            interpolate = 64
            simple.path.interpolate(interpolate)

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

    def plan_whole_body_motion(self, start_joints, goal_joints, start_base, goal_base):
        """`
        Plan a whole body motion using the multilayer RRTC planner.

        Args:
            start_joints: List of 8 joint positions for start configuration
            goal_joints: List of 8 joint positions for goal configuration
            start_base: List of 3 values [x, y, theta] for start base configuration
            goal_base: List of 3 values [x, y, theta] for goal base configuration

        Returns:
            dict: Planning results or None if planning failed
        """
        # Validate input dimensions
        assert len(start_joints) == 8 and len(goal_joints) == 8, "Invalid joint dimensions. Expected 8, got {len(start_joints)} and {len(goal_joints)}"
        assert len(start_base) == 3 and len(goal_base) == 3, "Invalid base dimensions. Expected 3, got {len(start_base)} and {len(goal_base)}"

        start_joints = [round(val, 3) for val in start_joints]
        goal_joints = [round(val, 3) for val in goal_joints]
        start_base = [round(val, 3) for val in start_base]
        goal_base = [round(val, 3) for val in goal_base]

        rospy.loginfo("Planning whole body motion:")
        rospy.loginfo(f"Start arm config: {start_joints}")
        rospy.loginfo(f"Goal arm config: {goal_joints}")
        rospy.loginfo(f"Start base config: {start_base}")
        rospy.loginfo(f"Goal base config: {goal_base}")

        # Use multilayer_rrtc for planning
        result = self.vamp_module.multilayer_rrtc(
            start_joints,  # arm start config array
            goal_joints,  # arm goal config array
            start_base,  # base start config array
            goal_base,  # base goal config array
            self.env,
            self.plan_settings,
            self.sampler,
        )

        if result.is_successful():
            rospy.loginfo("Whole body motion planning succeeded!")

            # Get the arm path from the multilayer planning result
            arm_path = result.arm_result.path

            # Get the base path from the result
            base_configs = []
            try:
                # Extract the base path
                base_path = result.base_path
                for config in base_path:
                    base_configs.append(config.config)
            except Exception as e:
                rospy.logwarn(
                    f"Error extracting base_path: {e}, using alternative method"
                )
                # Alternative method to get base path
                if (
                    hasattr(result.base_result, "path")
                    and result.base_result.path is not None
                ):
                    base_path = result.base_result.path
                    for i in range(len(base_path)):
                        config = base_path[i]
                        if hasattr(config, "to_list"):
                            base_configs.append(config.to_list())
                        elif hasattr(config, "config"):
                            base_configs.append(config.config)
                        else:
                            base_configs.append(list(config))

            # Convert base_configs to list of lists for whole_body_simplify
            base_path_list = []
            for config in base_configs:
                if isinstance(config, list):
                    base_path_list.append(config)
                else:
                    base_path_list.append(list(config))

            # Convert arm path to a list of lists for whole_body_simplify
            arm_path_list = []
            for config in arm_path:
                if isinstance(config, list):
                    arm_path_list.append(config)
                elif isinstance(config, np.ndarray):
                    arm_path_list.append(config.tolist())
                else:
                    arm_path_list.append(config.to_list())

            rospy.loginfo(
                f"Using whole_body_simplify with {len(arm_path_list)} arm configurations and {len(base_path_list)} base configurations"
            )

            # Use whole_body_simplify instead of regular simplify
            whole_body_result = self.vamp_module.whole_body_simplify(
                arm_path_list,
                base_path_list,
                self.env,
                self.simp_settings,
                self.sampler,
            )

            # Verify that arm and base paths have the same length after simplification
            if not whole_body_result.validate_paths():
                rospy.logwarn(
                    "Simplified arm and base paths have different lengths!"
                )
                rospy.logwarn(
                    f"Arm path: {len(whole_body_result.arm_result.path)}, Base path: {len(whole_body_result.base_path)}"
                )
            else:
                rospy.loginfo(
                    f"Verified: Arm and base paths both have {len(whole_body_result.arm_result.path)} waypoints"
                )

            # Interpolate both arm and base paths together
            rospy.loginfo("Interpolating whole-body path...")
            interpolation_resolution = 64
            # interpolation_resolution = self.vamp_module.resolution()
            whole_body_result.interpolate(interpolation_resolution)
            rospy.loginfo(
                f"Interpolated with resolution {interpolation_resolution}"
            )

            # Get the interpolated paths
            arm_path = whole_body_result.arm_result.path
            base_path = whole_body_result.base_path

            # Extract base configurations as lists
            base_configs = []
            for config in base_path:
                if hasattr(config, "config"):
                    base_configs.append(config.config)
                elif hasattr(config, "to_list"):
                    base_configs.append(config.to_list())
                else:
                    base_configs.append(list(config))

            # Create stats dictionary manually to avoid pandas dependency
            stats = {
                "arm_planning_time_ms": result.arm_result.nanoseconds / 1e6,
                "base_planning_time_ms": result.base_result.nanoseconds / 1e6,
                "total_planning_time_ms": result.nanoseconds / 1e6,
                "planning_iterations": result.arm_result.iterations,
                "base_planning_iterations": result.base_result.iterations,
                "planning_graph_size": (
                    sum(result.arm_result.size) if result.arm_result.size else 0
                ),
                "simplification_time_ms": whole_body_result.arm_result.nanoseconds
                / 1e6,
            }

            # Log statistics
            rospy.loginfo(
                f"""Whole body planning statistics:
                Total Planning Time: {stats['total_planning_time_ms']:.3f}ms
                Arm Planning Time: {stats['arm_planning_time_ms']:.3f}ms
                Base Planning Time: {stats['base_planning_time_ms']:.3f}ms
                Simplification Time: {stats['simplification_time_ms']:.3f}ms
                Planning Iters: {stats['planning_iterations']}
                Base Planning Iters: {stats['base_planning_iterations']}
                Graph States: {stats['planning_graph_size']}"""
            )

            rospy.loginfo(f"Base path length: {len(base_configs)} waypoints")
            rospy.loginfo(f"Arm path length: {len(arm_path)} waypoints")

            # Return planning results
            return {
                "success": True,
                "stats": stats,
                "arm_path": arm_path,
                "base_configs": base_configs,
            }
        else:
            rospy.logwarn("Whole body motion planning failed.")

            # Create stats dictionary for failure case without pandas dependency
            stats = {
                "arm_planning_time_ms": result.arm_result.nanoseconds / 1e6,
                "base_planning_time_ms": result.base_result.nanoseconds / 1e6,
                "total_planning_time_ms": result.nanoseconds / 1e6,
                "planning_iterations": result.arm_result.iterations,
                "base_planning_iterations": result.base_result.iterations,
                "planning_graph_size": (
                    sum(result.arm_result.size) if result.arm_result.size else 0
                ),
            }

            return {
                "success": False,
                "stats": stats,
                "arm_path": None,
                "base_configs": None,
            }

    def execute_whole_body_motion(self, arm_path, base_configs):
        """
        Execute a whole body motion plan with the Fetch robot.

        This method sends the planned trajectories to the C++ whole body controller
        for coordinated whole-body motion and waits for a completion signal.

        Args:
            arm_path: List of arm joint configurations (8-DOF including torso)
            base_configs: List of base configurations [x, y, theta]

        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # Validate input
            assert arm_path is not None and base_configs is not None, "Cannot execute motion: arm_path or base_configs is None"

            # Reset completion flag before starting
            self.execution_finished = False

            # Process arm path
            processed_arm_path = []
            for point in arm_path:
                if isinstance(point, list):
                    processed_arm_path.append(point)
                elif isinstance(point, np.ndarray):
                    processed_arm_path.append(point.tolist())
                else:
                    # Assume it has a to_list method
                    processed_arm_path.append(point.to_list())

            # Send trajectories to the C++ controller

            # Create the arm trajectory message
            arm_traj_msg = JointTrajectory()
            arm_traj_msg.header.stamp = rospy.Time.now()
            arm_traj_msg.joint_names = self.planning_joint_names

            # Create the base trajectory message (using joint_names as a placeholder)
            base_traj_msg = JointTrajectory()
            base_traj_msg.header.stamp = rospy.Time.now()
            base_traj_msg.joint_names = ["x", "y", "theta"]  # Base has 3 DOF

            # Calculate time spacing for points based on a reasonable duration estimate
            # This is only for the JointTrajectoryPoint `time_from_start` which some controllers use for timing.
            # The actual execution is handled by the MPC controller's rate.
            estimated_duration = len(processed_arm_path) * 0.1  # Heuristic
            time_step = estimated_duration / len(processed_arm_path) if len(processed_arm_path) > 0 else 0

            # Add points to both trajectories
            for i in range(len(processed_arm_path)):
                # Arm point
                arm_point = JointTrajectoryPoint()
                arm_point.positions = processed_arm_path[i]
                arm_point.time_from_start = rospy.Duration(i * time_step)
                arm_traj_msg.points.append(arm_point)

                # Base point
                base_point = JointTrajectoryPoint()
                base_point.positions = base_configs[i]
                base_point.time_from_start = rospy.Duration(i * time_step)
                base_traj_msg.points.append(base_point)

            # Publish trajectories
            self.arm_path_pub.publish(arm_traj_msg)
            self.base_path_pub.publish(base_traj_msg)

            # Give time for controller to receive and process the message
            rospy.sleep(0.5)

            # Wait for completion signal from the controller
            rospy.loginfo(
                "Whole body motion execution started, waiting for completion signal..."
            )
            while not self.execution_finished and not rospy.is_shutdown():
                rospy.sleep(0.1)

            if self.execution_finished:
                rospy.loginfo("Whole body motion execution completed successfully.")
                return True
            else:
                rospy.logwarn("ROS shutdown while waiting for motion completion.")
                return False

        except Exception as e:
            rospy.logerr(f"Error in whole body motion execution: {e}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return False

    def add_sphere(self, position, radius, name=None):
        """
        Add a sphere constraint to the environment.

        Args:
            position: [x, y, z] center position
            radius: sphere radius
            name: optional name for the sphere
        """
        position = list(position)  # Convert to list to ensure correct type
        sphere = vamp.Sphere(position, radius)
        if name:
            sphere.name = name
        self.env.add_sphere(sphere)
        rospy.loginfo(f"Added sphere constraint at {position} with radius {radius}")

    def add_box(self, position, half_extents, orientation_euler_xyz, name=None):
        """
        Add a box constraint to the environment.

        Args:
            position: [x, y, z] center position
            half_extents: [x, y, z] half-lengths in each dimension
            orientation_euler_xyz: [roll, pitch, yaw] orientation in radians
            name: optional name for the box
        """
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
        is_center_based = (
            center is not None
            and orientation_euler_xyz is not None
            and radius is not None
            and length is not None
        )
        is_endpoint_based = (
            endpoint1 is not None and endpoint2 is not None and radius is not None
        )

        assert is_center_based or is_endpoint_based, "Invalid parameters for add_capsule. Use either (center, orientation, radius, length) or (endpoint1, endpoint2, radius)."

        # Check which capsule definition method to use
        if is_center_based:
            # Method 1: Using center, orientation, radius, and length
            center = list(center)  # Convert to list to ensure correct type
            orientation_euler_xyz = list(orientation_euler_xyz)

            capsule = vamp.Capsule(center, orientation_euler_xyz, radius, length)
            rospy.loginfo(
                f"Added capsule constraint at {center} with radius {radius} and length {length}"
            )

        elif is_endpoint_based:
            # Method 2: Using two endpoints and radius
            endpoint1 = list(endpoint1)  # Convert to list to ensure correct type
            endpoint2 = list(endpoint2)

            capsule = vamp.Capsule(endpoint1, endpoint2, radius)
            rospy.loginfo(
                f"Added capsule constraint from {endpoint1} to {endpoint2} with radius {radius}"
            )

        # Set name if provided
        if name:
            capsule.name = name

        # Add capsule to environment
        self.env.add_capsule(capsule)

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

    def add_pointcloud(
        self,
        points,
        filter_radius=0.02,
        filter_cull=False,
        enable_filtering=False,
    ):
        """
        Add a pointcloud as collision constraint from already loaded point data.

        Args:
            points: List of [x,y,z] points or numpy array of shape (N,3)
            filter_radius: Filter radius for pointcloud filtering (used only if enable_filtering=True)
            filter_cull: Whether to cull pointcloud around robot (used only if enable_filtering=True)
            enable_filtering: Whether to enable point cloud filtering (default: False)

        Returns:
            float: Time taken to process and add the point cloud or -1 if error
        """
        import numpy as np
        from time import time

        start_time = time()
        rospy.loginfo(f"Processing point cloud with {len(points)} points...")

        if not enable_filtering:
            rospy.loginfo("Point cloud filtering is DISABLED")

        # Convert to numpy array if not already
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)

        # Determine whether to apply filtering
        if enable_filtering:
            rospy.loginfo(
                f"Processing {len(points)} points for collision checking..."
            )

            # Get robot-specific parameters
            filter_origin = [0.0, 0.0, 0.0]  # Default filter origin (robot base)
            filter_cull_radius = 1.4  # Default max reach radius for Fetch

            # Define bounding box for filtering
            bbox_lo = np.array(filter_origin) - filter_cull_radius
            bbox_hi = np.array(filter_origin) + filter_cull_radius

            # Filter the point cloud
            filtered_pc, filter_time_us = self._filter_pointcloud(
                points,
                filter_radius,
                filter_cull_radius,
                filter_origin,
                bbox_lo,
                bbox_hi,
                filter_cull,
            )

            filter_time = filter_time_us / 1e6  # Convert to seconds

            # Check if filtering was successful
            if len(filtered_pc) == 0:
                rospy.logwarn(
                    "Filtering resulted in zero points, using simple sphere obstacle instead"
                )
                # Add a simple sphere as a fallback
                sphere = vamp.Sphere([0.5, 0.0, 0.5], 0.2)
                sphere.name = "fallback_obstacle"
                self.env.add_sphere(sphere)
                processing_time = time() - start_time
                return processing_time

            points_to_use = filtered_pc
            rospy.loginfo(f"Using {len(points_to_use)} filtered points")
        else:
            # Skip filtering
            points_to_use = points
            filter_time = 0
            rospy.loginfo(f"Using all {len(points_to_use)} points (no filtering)")

        # Define robot-specific radius parameters
        r_min, r_max = 0.03, 0.08  # Min/max sphere radius for Fetch robot
        point_radius = 0.03  # Default point radius for collision checking

        # Add the filtered point cloud to the environment
        # Ensure points are in list format for vamp
        if isinstance(points_to_use, np.ndarray):
            points_to_use = points_to_use.tolist()

        add_start_time = time()
        build_time = self.env.add_pointcloud(
            points_to_use, r_min, r_max, point_radius
        )
        add_time = time() - add_start_time

        processing_time = time() - start_time

        if enable_filtering:
            rospy.loginfo(
                f"""Pointcloud processing stats:
                Original size: {len(points)}
                Filtered size: {len(points_to_use)}
                Filter time: {filter_time:.3f}s
                Build time: {build_time * 1e-6:.3f}s
                Add time: {add_time:.3f}s
                Total processing time: {processing_time:.3f}s"""
            )
        else:
            rospy.loginfo(
                f"""Pointcloud processing stats (NO FILTERING):
                Points processed: {len(points)}
                Build time: {build_time * 1e-6:.3f}s
                Add time: {add_time:.3f}s
                Total processing time: {processing_time:.3f}s
                Processing rate: {len(points) / processing_time:.1f} points/sec"""
            )

        return processing_time

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

    def send_joint_values(self, target_joints, duration=5.0):
        """
        Move the arm and torso to specified joint positions using VAMP planning.

        Args:
            target_joints: List of 8 joint positions [torso + 7 arm joints]
            duration: Time to execute trajectory
        """
        assert len(target_joints) == 8, "Expected 8 joint positions [torso + 7 arm joints]"

        # Get current joints
        current_joints = self.get_current_planning_joints()

        max_attempts = 5  # You can adjust this number based on your needs
        attempt_count = 0

        while current_joints is None:
            current_joints = self.get_current_planning_joints()
            if current_joints is None:
                attempt_count += 1
                rospy.logwarn(
                    f"Failed to get current joint positions (attempt {attempt_count}/{max_attempts})"
                )
                assert attempt_count < max_attempts, "Max attempts reached. Failed to get current joint positions"
                rospy.sleep(0.5)  # Add a small delay between attempts

        rospy.loginfo(f"Planning motion to target configuration: {target_joints}")

        # Plan path using VAMP
        trajectory_points = self._plan_path_with_vamp(current_joints, target_joints)
        assert trajectory_points is not None, "Failed to plan path with VAMP"

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

    def move_head(self, pan, tilt, duration=1.0):
        """
        Moves the robot's head to a given pan and tilt position.

        Args:
            pan (float): The target pan position for the head.
            tilt (float): The target tilt position for the head.
            duration (float): The duration of the movement in seconds.
        """
        self.head_controller.move_head(pan, tilt, duration)

    def point_head_at(self, target_point, frame_id="map", duration=1.0):
        """
        Points the robot's head towards a target point in the specified frame.

        Args:
            target_point (list): [x, y, z] coordinates of the target.
            frame_id (str): The TF frame ID of the target point (default: 'map').
            duration (float): The duration of the head movement in seconds.
        """
        self.head_controller.point_head_at(target_point, frame_id, duration)
