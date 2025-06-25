import rospy
import time
from geometry_msgs.msg import Pose
import fetch.utils.whole_body_ik_utils as ik_utils
from trac_ik_python.trac_ik import IK
import traceback
import numpy as np
import tf.transformations as transformations

class WholeBodyIKSolver:
    """
    A class to handle whole-body inverse kinematics for the Fetch robot,
    encapsulating the IK solver, costmap, and other related utilities.
    """

    def __init__(self, urdf_path="resources/fetch_ext/fetch ext.urdf", costmap_path="resources/costmap.npz"):
        """
        Initialize the WholeBodyIKSolver.

        Args:
            urdf_path (str): Path to the URDF file for the robot.
            costmap_path (str): Path to the costmap file.
        """
        self.BASE_LINK = "world_link"
        self.ARM_BASE_LINK = "base_link"
        self.EE_LINK = "gripper_link"
        self.urdf_path = urdf_path
        self.costmap_path = costmap_path
        
        self.full_ik_solver = None
        self.arm_ik_solver = None
        
        self.lower_limits = None
        self.upper_limits = None
        self.arm_lower_limits = None
        self.arm_upper_limits = None
        
        self.costmap = None
        self.costmap_metadata = None

        self._initialize_ik_solvers()
        self._load_costmap()

    def _initialize_ik_solvers(self):
        """Load URDF and initialize the TRAC-IK solvers."""
        try:
            rospy.loginfo(f"Loading URDF from {self.urdf_path} for IK solvers.")
            with open(self.urdf_path, "r") as f:
                self.urdf_str = f.read()

            # Full body solver (floating base from world_link)
            self.full_ik_solver = IK(
                self.BASE_LINK,
                self.EE_LINK,
                urdf_string=self.urdf_str,
                timeout=0.5,
                epsilon=1e-6,
            )
            self.lower_limits, self.upper_limits = self.full_ik_solver.get_joint_limits()
            rospy.loginfo(f"Initialized full-body IK solver ('{self.BASE_LINK}' to '{self.EE_LINK}') with {len(self.lower_limits)} joints.")

            # Arm only solver (fixed base from base_link)
            self.arm_ik_solver = IK(
                self.ARM_BASE_LINK,
                self.EE_LINK,
                urdf_string=self.urdf_str,
                timeout=0.5,
                epsilon=1e-6,
            )
            self.arm_lower_limits, self.arm_upper_limits = self.arm_ik_solver.get_joint_limits()
            rospy.loginfo(f"Initialized arm-only IK solver ('{self.ARM_BASE_LINK}' to '{self.EE_LINK}') with {len(self.arm_lower_limits)} joints.")

        except Exception as e:
            rospy.logerr(f"Failed to initialize IK solvers: {str(e)}")
            rospy.logerr(traceback.format_exc())

    def _load_costmap(self):
        """Load the costmap and its metadata."""
        if self.costmap_path:
            self.costmap, self.costmap_metadata = ik_utils.load_costmap(self.costmap_path)
            if self.costmap is None:
                rospy.logwarn(f"Failed to load costmap from {self.costmap_path}. Proceeding without costmap features.")
        else:
            rospy.logwarn("No costmap path provided. Proceeding without costmap features.")

    def solve(
        self,
        vamp_module,
        env,
        target_pose,
        max_attempts=100,
        manipulation_radius=1.0,
        use_fixed_base=False,
    ):
        """
        Solve whole-body inverse kinematics for a given target pose.

        This method attempts to find a valid (collision-free) configuration for the whole robot
        (base + arm) that places the end effector at the desired pose.

        Args:
            vamp_module: VAMP planning module for collision checking.
            env: VAMP environment.
            target_pose: Target end effector pose (geometry_msgs/Pose).
            max_attempts: Maximum number of sampling attempts.
            manipulation_radius: Radius for base position sampling.
            use_fixed_base: If True, uses the fixed-base IK strategy.

        Returns:
            dict: Solution containing base and arm configuration, or None if no solution found.
        """
        if not self.full_ik_solver or (use_fixed_base and not self.arm_ik_solver):
            rospy.logerr("IK solver not initialized, cannot solve.")
            return None

        if use_fixed_base:
            rospy.loginfo("Solving whole-body IK with a fixed base strategy.")
            return self._solve_with_fixed_base(
                vamp_module, env, target_pose, max_attempts, manipulation_radius
            )
        else:
            rospy.loginfo("Solving whole-body IK with a floating base strategy.")
            return self._solve_with_floating_base(
                vamp_module, env, target_pose, max_attempts, manipulation_radius
            )

    def _solve_with_floating_base(
        self, vamp_module, env, target_pose, max_attempts, manipulation_radius
    ):
        """IK solving with a floating base (original method)."""
        pose = Pose()
        pose.position.x = target_pose.position.x
        pose.position.y = target_pose.position.y
        pose.position.z = target_pose.position.z
        pose.orientation.x = target_pose.orientation.x
        pose.orientation.y = target_pose.orientation.y
        pose.orientation.z = target_pose.orientation.z
        pose.orientation.w = target_pose.orientation.w

        rospy.loginfo(
            "Solving whole-body IK for pose: "
            + f"position [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}], "
            + f"orientation [{pose.orientation.x:.3f}, {pose.orientation.y:.3f}, "
            + f"{pose.orientation.z:.3f}, {pose.orientation.w:.3f}]"
        )

        solution = None
        is_valid = False
        sample_count = 0
        all_time = time.time()

        while not is_valid and sample_count < max_attempts:
            sample_count += 1
            rospy.loginfo(f"IK attempt {sample_count}/{max_attempts}")

            seed = ik_utils.generate_ik_seed(
                pose,
                self.costmap,
                self.costmap_metadata,
                self.lower_limits,
                self.upper_limits,
                manipulation_radius=manipulation_radius,
            )
            if seed is None:
                rospy.logerr("Failed to generate IK seed. Aborting IK attempt.")
                continue

            rospy.loginfo(
                f"Initial configuration (seed): {[round(val, 3) for val in seed]}"
            )

            start_time = time.time()
            ik_solution = self.full_ik_solver.get_ik(
                seed,
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
            )
            solve_time = time.time() - start_time
            rospy.loginfo(f"IK solving time: {solve_time:.4f} seconds")

            if not ik_solution:
                rospy.logwarn("IK failed. Trying again...")
                continue

            rospy.loginfo("IK solution found, performing collision check...")

            vamp_solution = list(ik_solution)
            base_config = vamp_solution[:3]

            if len(vamp_solution) > 8:
                arm_config_vamp = vamp_solution[3:11]
                vamp_module.set_base_params(base_config[2], base_config[0], base_config[1])
                is_valid = vamp_module.validate(arm_config_vamp, env)

            if is_valid:
                rospy.loginfo("IK solution is valid (collision-free)")
                solution = ik_solution
            else:
                rospy.logwarn("IK solution is not valid (has collisions). Trying again...")

        if is_valid and solution:
            rospy.loginfo(f"Found valid solution after {sample_count} attempts in {time.time() - all_time:.4f} seconds")
            base_position = [solution[0], solution[1]]
            base_orientation = solution[2]
            goal_base = [base_position[0], base_position[1], base_orientation]

            if len(solution) >= 11:
                arm_config = solution[3:11]
                return {
                    "base_config": goal_base,
                    "arm_config": list(arm_config),
                    "full_solution": list(solution),
                    "attempts": sample_count,
                }
            else:
                rospy.logerr("Solution doesn't have enough values for arm configuration")
                return None
        else:
            rospy.logerr(f"Failed to find valid solution after {max_attempts} attempts")
            return None

    def _solve_with_fixed_base(
        self, vamp_module, env, target_pose, max_attempts, manipulation_radius
    ):
        """IK solving with a fixed base, solving only for the arm."""
        solution = None
        is_valid = False
        sample_count = 0
        all_time = time.time()

        while not is_valid and sample_count < max_attempts:
            sample_count += 1
            rospy.loginfo(f"IK attempt {sample_count}/{max_attempts} with fixed base")

            # 1. Sample a base pose and arm seed. The full seed is for the floating base solver,
            # but we use it to get a consistent starting point.
            seed = ik_utils.generate_ik_seed(
                target_pose,
                self.costmap,
                self.costmap_metadata,
                self.lower_limits,  # Use full limits for seeding
                self.upper_limits,
                manipulation_radius=manipulation_radius,
            )
            if seed is None:
                rospy.logerr("Failed to generate IK seed. Aborting IK attempt.")
                continue

            base_x, base_y, base_theta = seed[0], seed[1], seed[2]
            arm_seed = seed[3:11]  # 8-DOF arm seed

            # 2. Transform the world-frame target pose to the base_link frame
            T_world_base = transformations.euler_matrix(0, 0, base_theta)
            T_world_base[0:3, 3] = [base_x, base_y, 0]
            
            pos = target_pose.position
            ori = target_pose.orientation
            T_world_ee = transformations.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
            T_world_ee[0:3, 3] = [pos.x, pos.y, pos.z]

            T_base_world = transformations.inverse_matrix(T_world_base)
            T_base_ee = np.dot(T_base_world, T_world_ee)

            pos_base_ee = transformations.translation_from_matrix(T_base_ee)
            quat_base_ee = transformations.quaternion_from_matrix(T_base_ee)

            # 3. Solve IK for the arm
            start_time = time.time()
            arm_solution = self.arm_ik_solver.get_ik(
                arm_seed,
                pos_base_ee[0], pos_base_ee[1], pos_base_ee[2],
                quat_base_ee[0], quat_base_ee[1], quat_base_ee[2], quat_base_ee[3]
            )
            rospy.loginfo(f"Arm-only IK solving time: {time.time() - start_time:.4f} seconds")

            if not arm_solution:
                rospy.logwarn("Arm-only IK failed. Trying new base sample...")
                continue

            # 4. If solved, construct the full solution and validate
            rospy.loginfo("Arm-only IK solution found, performing collision check...")
            full_solution = list(seed[:3]) + list(arm_solution)
            
            vamp_module.set_base_params(full_solution[2], full_solution[0], full_solution[1])
            is_valid = vamp_module.validate(arm_solution, env)

            if is_valid:
                rospy.loginfo("Full solution is valid (collision-free)")
                solution = full_solution
            else:
                rospy.logwarn("Full solution is not valid (has collisions). Trying new base sample...")
        
        if is_valid and solution:
            rospy.loginfo(f"Found valid solution after {sample_count} attempts in {time.time() - all_time:.4f} seconds")
            goal_base = solution[:3]
            arm_config = solution[3:11]
            return {
                "base_config": goal_base,
                "arm_config": list(arm_config),
                "full_solution": list(solution),
                "attempts": sample_count,
            }
        else:
            rospy.logerr(f"Failed to find valid solution after {max_attempts} attempts")
            return None

    def solve_arm_ik(
        self,
        target_pose,
        arm_seed=None,
        max_attempts=10
    ):
        """
        Solves IK for the arm only, from base_link to the end-effector, without collision checking.
        The target pose is assumed to be in the base_link frame.

        Args:
            target_pose (geometry_msgs.msg.Pose): The target pose for the end-effector in the base_link frame.
            arm_seed (list, optional): An initial guess for the arm joint angles (8-DOF).
                                    If None, a random seed will be generated. Defaults to None.
            max_attempts (int): Number of attempts with different random seeds if no seed is provided.

        Returns:
            list: A list of 8 joint values for the arm if a solution is found, otherwise None.
        """
        if not self.arm_ik_solver:
            rospy.logerr("Arm IK solver not initialized, cannot solve.")
            return None

        rospy.loginfo(
            "Solving arm-only IK for pose in base_link frame: "
            + f"position [{target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f}], "
            + f"orientation [{target_pose.orientation.x:.3f}, {target_pose.orientation.y:.3f}, "
            + f"{target_pose.orientation.z:.3f}, {target_pose.orientation.w:.3f}]"
        )

        for attempt in range(max_attempts):
            current_seed = arm_seed
            if current_seed is None:
                # Generate a random seed within joint limits.
                current_seed = [np.random.uniform(L, U) for L, U in zip(self.arm_lower_limits, self.arm_upper_limits)]
                rospy.loginfo(f"Generated random seed for attempt {attempt+1}/{max_attempts}")

            if not current_seed:
                rospy.logerr("Could not generate seed.")
                return None

            arm_solution = self.arm_ik_solver.get_ik(
                current_seed,
                target_pose.position.x, target_pose.position.y, target_pose.position.z,
                target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w
            )

            if arm_solution:
                rospy.loginfo(f"Found arm-only IK solution on attempt {attempt+1}")
                return list(arm_solution)

            if arm_seed is not None:
                # if a seed was provided and it failed, don't try again.
                break

        rospy.logerr(f"Failed to find arm-only IK solution after {max_attempts} attempts.")
        return None 