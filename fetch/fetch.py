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
        
        self.planner = "rrtc" # ["rrtc", "fcit", "prm"]

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

    def _joint_states_callback(self, msg):
        """Callback for joint states messages."""
        # Only save messages with length > 2 to filter out gripper messages
        if len(msg.name) > 2:
            self.joint_states = msg
        else:
            rospy.logdebug(f"Ignoring joint state message with only {len(msg.name)} joints")

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
        try:
            self.base_theta = theta
            self.base_x = x
            self.base_y = y
            
            # Update the base parameters in the VAMP module
            self.vamp_module.set_base_params(theta, x, y)
            
            rospy.loginfo(f"Set base parameters: theta={theta:.6f}, x={x:.6f}, y={y:.6f}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to set base parameters: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False
        
    def get_base_params(self):
        """
        Get the current base parameters for the Fetch robot.
        
        Returns:
            tuple: (theta, x, y) current base parameters
        """
        try:
            theta = self.vamp_module.get_base_theta()
            x = self.vamp_module.get_base_x()
            y = self.vamp_module.get_base_y()
            return (theta, x, y)
        except Exception as e:
            rospy.logerr(f"Failed to get base parameters: {str(e)}")
            return (self.base_theta, self.base_x, self.base_y)

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
        try:
            # Validate input dimensions
            if len(start_joints) != 8 or len(goal_joints) != 8:
                rospy.logerr(f"Invalid joint dimensions. Expected 8, got {len(start_joints)} and {len(goal_joints)}")
                return None
                
            if len(start_base) != 3 or len(goal_base) != 3:
                rospy.logerr(f"Invalid base dimensions. Expected 3, got {len(start_base)} and {len(goal_base)}")
                return None
            
            start_joints = [round(val, 3) for val in start_joints]
            goal_joints = [round(val, 3) for val in goal_joints]
            start_base = [round(val, 3) for val in start_base]
            goal_base = [round(val, 3) for val in goal_base]
            
            # Validate input values
            self.set_base_params(*start_base)
            if not self.vamp_module.validate(start_joints, self.env):
                rospy.loginfo("Start configuration valid")
            else:
                rospy.logerr("Start configuration not valid")
                return
            self.set_base_params(*goal_base)
            if not self.vamp_module.validate(goal_joints, self.env):
                rospy.loginfo("Target configuration valid")
            else:
                rospy.logerr("Target configuration not valid")
                return
            
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
                    rospy.logwarn(f"Error extracting base_path: {e}, using alternative method")
                    # Alternative method to get base path
                    if hasattr(result.base_result, "path") and result.base_result.path is not None:
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
                
                rospy.loginfo(f"Using whole_body_simplify with {len(arm_path_list)} arm configurations and {len(base_path_list)} base configurations")
                
                # Use whole_body_simplify instead of regular simplify
                whole_body_result = self.vamp_module.whole_body_simplify(
                    arm_path_list, 
                    base_path_list, 
                    self.env, 
                    self.simp_settings, 
                    self.sampler
                )
                
                # Verify that arm and base paths have the same length after simplification
                if not whole_body_result.validate_paths():
                    rospy.logwarn("Simplified arm and base paths have different lengths!")
                    rospy.logwarn(f"Arm path: {len(whole_body_result.arm_result.path)}, Base path: {len(whole_body_result.base_path)}")
                else:
                    rospy.loginfo(f"Verified: Arm and base paths both have {len(whole_body_result.arm_result.path)} waypoints")
                
                # Interpolate both arm and base paths together
                rospy.loginfo("Interpolating whole-body path...")
                interpolation_resolution = 32
                # interpolation_resolution = self.vamp_module.resolution()
                whole_body_result.interpolate(interpolation_resolution)
                rospy.loginfo(f"Interpolated with resolution {interpolation_resolution}")
                
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
                    "simplification_time_ms": whole_body_result.arm_result.nanoseconds / 1e6
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
                    "base_configs": base_configs
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
                    "base_configs": None
                }
        except Exception as e:
            rospy.logerr(f"Error in whole body motion planning: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None

    def execute_whole_body_motion(self, arm_path, base_configs, duration=10.0):
        """
        Execute a whole body motion plan with the Fetch robot.
        
        This method sends the planned trajectories to the C++ whole body controller
        for coordinated whole-body motion.
        
        Args:
            arm_path: List of arm joint configurations (8-DOF including torso)
            base_configs: List of base configurations [x, y, theta]
            duration: Total duration of the motion execution in seconds
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            # Validate input
            if arm_path is None or base_configs is None:
                rospy.logerr("Cannot execute motion: arm_path or base_configs is None")
                return False
            
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
            
            # Calculate time spacing for points
            time_step = duration / len(processed_arm_path)
            
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
            
            # The C++ controller is handling execution, so we just wait for completion
            rospy.loginfo(f"Whole body motion execution started, waiting {duration + 2.0} seconds...")
            rospy.sleep(duration + 2.0)  # Add a small buffer time for completion
            
            rospy.loginfo("Whole body motion execution completed")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error in whole body motion execution: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False

    def send_whole_body_motion(self, target_joints, target_base, start_joints=None, start_base=None, duration=10.0):
        """
        Plan and execute a whole body motion to target joint and base configurations.
        
        Args:
            target_joints: List of 8 joint positions [torso + 7 arm joints]
            target_base: List of 3 base values [x, y, theta]
            start_joints: List of 8 joint positions for start, or None to use current
            start_base: List of 3 base values for start, or None to use current
            duration: Time to execute trajectory
            
        Returns:
            bool: True if motion succeeded, False otherwise
        """
        try:
            # Get current joint positions if not provided
            if start_joints is None:
                start_joints = self.get_current_planning_joints()
                if start_joints is None:
                    rospy.logerr("Failed to get current joint positions")
                    return False
                    
            # Get current base position if not provided
            if start_base is None:
                try:
                    # Get transform from map to base_link
                    transform = self.tf_buffer.lookup_transform(
                        "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
                    )
                    
                    # Extract current base position and orientation
                    current_x = transform.transform.translation.x
                    current_y = transform.transform.translation.y
                    
                    # Extract orientation as quaternion
                    quat = [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w
                    ]
                    
                    # Convert quaternion to Euler angles
                    import tf.transformations
                    euler = tf.transformations.euler_from_quaternion(quat)
                    current_theta = euler[2]  # Yaw angle
                    
                    start_base = [current_x, current_y, current_theta]
                    
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                        tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"Could not get current base pose: {e}")
                    # Use default values
                    start_base = [0.0, 0.0, 0.0]
                    
            # Validate input dimensions
            if len(target_joints) != 8:
                rospy.logerr(f"Invalid target joints: expected 8 values, got {len(target_joints)}")
                return False
                
            if len(target_base) != 3:
                rospy.logerr(f"Invalid target base: expected 3 values, got {len(target_base)}")
                return False
                
            if len(start_joints) != 8:
                rospy.logerr(f"Invalid start joints: expected 8 values, got {len(start_joints)}")
                return False
                
            if len(start_base) != 3:
                rospy.logerr(f"Invalid start base: expected 3 values, got {len(start_base)}")
                return False
                
            # Log motion parameters
            rospy.loginfo(f"Planning whole body motion from {start_base} to {target_base}")
            
            # Plan whole body motion
            plan_result = self.plan_whole_body_motion(
                start_joints, target_joints, start_base, target_base
            )
            
            if plan_result is None or not plan_result["success"]:
                rospy.logerr("Failed to plan whole body motion")
                return False
                
            # Execute the planned motion
            execution_result = self.execute_whole_body_motion(
                plan_result["arm_path"], plan_result["base_configs"], duration
            )
            
            return execution_result
            
        except Exception as e:
            rospy.logerr(f"Error in send_whole_body_motion: {e}")
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

        try:
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
            try:
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
        Transform a set of points from source frame to target frame using NumPy vectorization.

        This optimized version replaces the point-by-point TF2 transformation with a much faster
        vectorized matrix operation that processes all points at once.

        Args:
            points: Nx3 array or list of points
            source_frame: Source frame ID (e.g., "world", "map")
            target_frame: Target frame ID (e.g., "base_link")

        Returns:
            Nx3 array of transformed points
        """
        import numpy as np
        from time import time

        start_time = time()

        # Convert points to numpy array if not already
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)

        # Handle empty input
        if points.size == 0:
            return points

        # Log transformation details
        rospy.loginfo(
            f"Fast transforming {len(points)} points from '{source_frame}' to '{target_frame}'"
        )

        try:
            # Get the latest transform
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(5.0)
            )

            # Extract translation and rotation from transform
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Log transform information for debugging
            rospy.loginfo(
                f"Translation: [{translation.x}, {translation.y}, {translation.z}]"
            )
            rospy.loginfo(
                f"Rotation: [{rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}]"
            )

            # Convert quaternion to rotation matrix (more efficient than individual transforms)
            x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w

            # Precompute common terms to optimize the calculation
            xx, xy, xz = x * x, x * y, x * z
            yy, yz, zz = y * y, y * z, z * z
            wx, wy, wz = w * x, w * y, w * z

            # Build rotation matrix
            rot_matrix = np.array(
                [
                    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
                ],
                dtype=np.float64,
            )

            # Create translation vector
            trans_vector = np.array(
                [translation.x, translation.y, translation.z], dtype=np.float64
            )

            # Process points in chunks to avoid memory issues with very large point clouds
            chunk_size = 500000  # Adjust based on available memory
            n_points = len(points)
            num_chunks = int(np.ceil(n_points / chunk_size))
            result = np.zeros_like(points)

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_points)

                # Get current chunk
                chunk = points[start_idx:end_idx]

                # Apply transformation to chunk: R * points + t
                result[start_idx:end_idx] = np.dot(chunk, rot_matrix.T) + trans_vector

                # Log progress for large point clouds
                if num_chunks > 1 and (i % 10 == 0 or i == num_chunks - 1):
                    points_processed = end_idx
                    percentage = (points_processed / n_points) * 100
                    elapsed = time() - start_time
                    rate = points_processed / elapsed if elapsed > 0 else 0
                    rospy.loginfo(
                        f"Transformed {points_processed}/{n_points} points ({percentage:.1f}%), "
                        f"Rate: {rate:.1f} points/sec"
                    )

            total_time = time() - start_time
            rospy.loginfo(
                f"Transformation complete: {n_points} points in {total_time:.3f} seconds "
                f"({n_points/total_time:.1f} points/sec)"
            )

            return result

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Failed to lookup transform: {e}")
            return points  # Return original points if transform fails
        except Exception as e:
            rospy.logerr(f"Error in point transformation: {str(e)}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return points  # Return original points on error

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
