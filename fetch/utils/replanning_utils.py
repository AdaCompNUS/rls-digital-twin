import rospy
import numpy as np
from typing import List, Any


def _process_arm_path(arm_path: List) -> List[List[float]]:
    """
    Process arm path configurations to ensure they're in the correct format.
    
    This follows the same processing pattern used in plan_whole_body_motion
    to ensure consistency between planning and collision checking.
    
    Args:
        arm_path: List of arm configurations in various formats
        
    Returns:
        List of arm configurations as lists of floats
    """
    processed_arm_path = []
    for config in arm_path:
        if isinstance(config, list):
            processed_arm_path.append(config)
        elif isinstance(config, np.ndarray):
            processed_arm_path.append(config.tolist())
        else:
            # Assume it has a to_list method
            processed_arm_path.append(config.to_list())
    return processed_arm_path


def _process_base_path(base_path: List) -> List[List[float]]:
    """
    Process base path configurations to ensure they're in the correct format.
    
    This follows the same processing pattern used in plan_whole_body_motion
    to ensure consistency between planning and collision checking.
    
    Args:
        base_path: List of base configurations in various formats
        
    Returns:
        List of base configurations as lists of floats
    """
    processed_base_path = []
    for config in base_path:
        if hasattr(config, "config"):
            processed_base_path.append(config.config)
        elif hasattr(config, "to_list"):
            processed_base_path.append(config.to_list())
        elif isinstance(config, list):
            processed_base_path.append(config)
        else:
            processed_base_path.append(list(config))
    return processed_base_path


def check_trajectory_for_collisions(
    vamp_module: Any,
    env: Any,
    arm_path: List[List[float]],
    base_path: List[List[float]],
    current_waypoint_index: int,
) -> bool:
    """
    Checks the remainder of a trajectory for collisions with the current environment.

    Args:
        vamp_module: The VAMP planner module.
        env: The VAMP collision environment.
        arm_path: The full arm trajectory.
        base_path: The full base trajectory.
        current_waypoint_index: The index of the waypoint that has been most recently
                                passed. The check will start from the next one.

    Returns:
        bool: True if a collision is detected in the remaining path, False otherwise.
    """
    # Process arm and base configurations to convert them to lists
    processed_arm_path = _process_arm_path(arm_path)
    processed_base_path = _process_base_path(base_path)

    if current_waypoint_index >= len(processed_arm_path) - 1:
        rospy.loginfo(
            "Trajectory execution finished or almost finished, no need to check for collision."
        )
        return False

    remaining_arm_path = processed_arm_path[current_waypoint_index + 1 :]
    remaining_base_path = processed_base_path[current_waypoint_index + 1 :]

    if not remaining_arm_path:
        rospy.loginfo("No remaining waypoints to check.")
        return False

    rospy.loginfo(
        f"Checking for collisions in the remaining {len(remaining_arm_path)} waypoints."
    )

    # The paths are already processed, so we can directly use them.
    assert len(remaining_arm_path) == len(
        remaining_base_path
    ), "The number of arm and base configs should be the same."
    assert all(
        len(config) == 8 for config in remaining_arm_path
    ), "The dimension of each arm configuration should be 8."
    assert all(
        len(config) == 3 for config in remaining_base_path
    ), "The dimension of each base configuration should be 3."

    collision_results = vamp_module.check_whole_body_collisions(
        env, remaining_arm_path, remaining_base_path
    )

    if collision_results:
        rospy.loginfo(
            "Collision detected in the planned path. Replanning needed."
        )
        return True
    else:
        rospy.loginfo("No collisions detected in the remaining path.")
        return False
