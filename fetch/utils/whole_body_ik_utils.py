import numpy as np
import rospy
import math
import os
import traceback

def load_costmap_file(costmap_path):
    """
    Load a previously generated costmap from file.

    Args:
        costmap_path: Path to the costmap .npz file

    Returns:
        tuple: (costmap, metadata) or (None, None) if loading fails
    """
    try:
        data = np.load(costmap_path)

        # Extract costmap and metadata
        costmap = data["costmap"]

        # Create metadata dictionary
        metadata = {
            "resolution": float(data["resolution"]),
            "origin_x": float(data["origin_x"]),
            "origin_y": float(data["origin_y"]),
            "width": int(data["width"]),
            "height": int(data["height"]),
            "max_distance": float(data["max_distance"]),
        }

        # Check if valid_area_mask exists in the data
        if "valid_area_mask" in data:
            metadata["valid_area_mask"] = data["valid_area_mask"]

        return costmap, metadata

    except Exception as e:
        rospy.logerr(f"Error loading costmap file {costmap_path}: {e}")
        rospy.logerr(traceback.format_exc())
        return None, None

def load_costmap(costmap_path):
    """
    Loads the costmap and its metadata from the specified path.

    Args:
        costmap_path (str): The path to the costmap file (.npz).

    Returns:
        tuple: A tuple containing (costmap, costmap_metadata).
               Returns (None, None) if loading fails or the file doesn't exist.
    """
    try:
        if not os.path.exists(costmap_path):
            rospy.logerr(f"Costmap file not found: {costmap_path}")
            return None, None

        rospy.loginfo(f"Loading costmap from {costmap_path}")
        costmap, metadata = load_costmap_file(costmap_path)

        if costmap is not None and metadata is not None:
            # Explicitly log costmap dimensions to verify loading
            rospy.loginfo(
                f"Successfully loaded costmap: {costmap.shape}, min={np.min(costmap)}, max={np.max(costmap)}"
            )
            return costmap, metadata
        else:
            rospy.logerr(f"Failed to load costmap data from {costmap_path}")
            return None, None
    except Exception as e:
        rospy.logerr(f"Error loading costmap: {e}")
        rospy.logerr(traceback.format_exc())
        return None, None

def world_to_grid(world_x, world_y, costmap_metadata):
    """
    Convert world coordinates to grid coordinates.

    Args:
        world_x, world_y: World coordinates
        costmap_metadata: Dictionary containing costmap metadata (resolution, origin_x, origin_y)

    Returns:
        grid_x, grid_y: Grid coordinates or (None, None) if metadata is missing.
    """
    if costmap_metadata is None:
        rospy.logwarn("Costmap metadata not available for world_to_grid conversion.")
        return None, None

    try:
        grid_x = int(
            (world_x - costmap_metadata["origin_x"])
            / costmap_metadata["resolution"]
        )
        grid_y = int(
            (world_y - costmap_metadata["origin_y"])
            / costmap_metadata["resolution"]
        )
        return grid_x, grid_y
    except KeyError as e:
        rospy.logerr(f"Missing key in costmap_metadata for world_to_grid: {e}")
        return None, None
    except Exception as e:
        rospy.logerr(f"Error in world_to_grid conversion: {e}")
        return None, None

def grid_to_world(grid_x, grid_y, costmap_metadata):
    """
    Convert grid coordinates to world coordinates.

    Args:
        grid_x, grid_y: Grid coordinates
        costmap_metadata: Dictionary containing costmap metadata (resolution, origin_x, origin_y)

    Returns:
        world_x, world_y: World coordinates or (None, None) if metadata is missing.
    """
    if costmap_metadata is None:
        rospy.logwarn("Costmap metadata not available for grid_to_world conversion.")
        return None, None

    try:
        world_x = (
            costmap_metadata["origin_x"]
            + grid_x * costmap_metadata["resolution"]
        )
        world_y = (
            costmap_metadata["origin_y"]
            + grid_y * costmap_metadata["resolution"]
        )
        return world_x, world_y
    except KeyError as e:
        rospy.logerr(f"Missing key in costmap_metadata for grid_to_world: {e}")
        return None, None
    except Exception as e:
        rospy.logerr(f"Error in grid_to_world conversion: {e}")
        return None, None

def find_valid_base_positions(ee_pose, costmap, costmap_metadata, manipulation_radius=1.0, cost_threshold=0.3):
    """
    Find valid base positions for the robot using the costmap.

    Args:
        ee_pose: End effector pose (geometry_msgs/Pose)
        costmap: The costmap numpy array.
        costmap_metadata: Dictionary containing costmap metadata.
        manipulation_radius: Radius of the manipulation range in meters
        cost_threshold: Minimum cost value to consider a cell valid (0-1)

    Returns:
        valid_positions: List of valid base positions (world_x, world_y, cost_value, theta)
                         or None if costmap is not available or errors occur.
    """
    target_point = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
    return find_valid_base_positions_from_point(
        target_point, costmap, costmap_metadata, manipulation_radius, cost_threshold
    )

def find_valid_base_positions_from_point(
    target_point, costmap, costmap_metadata, manipulation_radius=1.0, cost_threshold=0.3
):
    """
    Find valid base positions for the robot using the costmap from a 3D point.

    Args:
        target_point: A 3D point [x, y, z]
        costmap: The costmap numpy array.
        costmap_metadata: Dictionary containing costmap metadata.
        manipulation_radius: Radius of the manipulation range in meters
        cost_threshold: Minimum cost value to consider a cell valid (0-1)

    Returns:
        valid_positions: List of valid base positions (world_x, world_y, cost_value, theta)
                         or None if costmap is not available or errors occur.
    """
    if costmap is None or costmap_metadata is None:
        rospy.logwarn("Costmap not available for finding valid base positions")
        return None

    pos_x, pos_y = target_point[0], target_point[1]

    # Convert end effector position to grid coordinates
    ee_grid_x, ee_grid_y = world_to_grid(pos_x, pos_y, costmap_metadata)

    if ee_grid_x is None or ee_grid_y is None:
        rospy.logwarn("Failed to convert end effector position to grid coordinates")
        return None

    # Make sure the end effector is within the costmap bounds
    if (
        ee_grid_x < 0
        or ee_grid_x >= costmap_metadata["width"]
        or ee_grid_y < 0
        or ee_grid_y >= costmap_metadata["height"]
    ):
        rospy.logwarn(
            f"End effector position ({pos_x}, {pos_y}) is outside costmap bounds"
        )
        return None

    # Create 2D grid coordinates
    y_grid, x_grid = np.mgrid[
        0 : costmap_metadata["height"], 0 : costmap_metadata["width"]
    ]

    # Calculate distances from each grid cell to end effector position
    distances = (
        np.sqrt((x_grid - ee_grid_x) ** 2 + (y_grid - ee_grid_y) ** 2)
        * costmap_metadata["resolution"]
    )

    # Create circle mask (cells within manipulation range)
    circle_mask = distances <= manipulation_radius

    # Create valid cells mask (cells with cost >= threshold and not NaN)
    valid_cells = (costmap >= cost_threshold) & ~np.isnan(costmap)

    # Combine circle mask and valid cells mask to get final valid positions
    valid_mask = circle_mask & valid_cells

    # Convert valid grid positions to world coordinates
    valid_positions = []
    for y in range(costmap_metadata["height"]):
        for x in range(costmap_metadata["width"]):
            if valid_mask[y, x]:
                world_x, world_y = grid_to_world(x, y, costmap_metadata)
                if world_x is None or world_y is None:
                    continue  # Skip if conversion failed

                # Calculate orientation towards the end effector
                dx = pos_x - world_x
                dy = pos_y - world_y
                theta = math.atan2(dy, dx)

                # Get the cost value (lower is better for sampling)
                cost_value = costmap[y, x]

                valid_positions.append((world_x, world_y, cost_value, theta))

    if not valid_positions:
        rospy.logwarn("No valid base positions found")
    else:
        rospy.loginfo(f"Found {len(valid_positions)} valid base positions")

    return valid_positions

def _generate_base_seed(
    pose, costmap, costmap_metadata, manipulation_radius, cost_threshold=0.3
):
    """Generates a base seed [x, y, theta] by finding a valid base position."""
    valid_positions = find_valid_base_positions(
        pose, costmap, costmap_metadata, manipulation_radius, cost_threshold
    )

    if not valid_positions:
        rospy.logwarn("No valid base positions found for seed generation")
        return None

    # Sort positions by cost (lower is better) and pick the best one
    sorted_positions = sorted(valid_positions, key=lambda pos: pos[2])
    best_position = sorted_positions[0]

    # Return x, y, theta
    return [best_position[0], best_position[1], best_position[3]]

def _generate_arm_seed(lower_limits, upper_limits, normalized_arm_seed=None):
    """
    Generates a random arm seed within the provided joint limits.
    If normalized_arm_seed is provided, it will be used to generate the seed.
    """
    lower = np.array(lower_limits[3:11])
    upper = np.array(upper_limits[3:11])
    
    if normalized_arm_seed is not None:
        return (lower + np.array(normalized_arm_seed) * (upper - lower)).tolist()
    else:
        return np.random.uniform(lower, upper).tolist()

def generate_ik_seed(
    pose, costmap, costmap_metadata, lower_limits, upper_limits, manipulation_radius=1.0, normalized_arm_seed=None
):
    """
    Generate a seed configuration for whole-body IK.

    This seed is a starting point for the IK solver, combining a strategically chosen
    base position with a randomized arm configuration.

    Args:
        pose: Target end effector pose (geometry_msgs/Pose)
        costmap: The costmap numpy array.
        costmap_metadata: Dictionary containing costmap metadata.
        lower_limits: Lower joint limits for the full robot configuration
        upper_limits: Upper joint limits for the full robot configuration
        manipulation_radius: Radius for base position sampling
        normalized_arm_seed: Optional normalized arm seed for specific arm configuration

    Returns:
        list: A full 11-DOF seed configuration [x, y, theta, j1, ..., j8] or None
    """
    if costmap is None or costmap_metadata is None:
        rospy.logwarn("Costmap not available, cannot generate a valid IK seed.")
        return None

    # 1. Generate a base seed
    base_seed = _generate_base_seed(
        pose, costmap, costmap_metadata, manipulation_radius
    )
    if base_seed is None:
        rospy.logerr("Failed to generate base seed.")
        return None

    # 2. Generate an arm seed
    arm_seed = _generate_arm_seed(lower_limits, upper_limits, normalized_arm_seed=normalized_arm_seed)

    # 3. Combine them into a full seed
    full_seed = base_seed + arm_seed
    rospy.loginfo(f"Generated IK seed: {[f'{x:.3f}' for x in full_seed]}")

    return full_seed

def generate_random_ik_seed(lower_limits, upper_limits):
    """
    Generate a random seed for IK by sampling from the entire configuration space.

    Args:
        lower_limits: Lower joint limits for the full robot configuration
        upper_limits: Upper joint limits for the full robot configuration

    Returns:
        list: A full 11-DOF seed configuration [x, y, theta, j1, ..., j8] or None
    """
    if lower_limits is None or upper_limits is None:
        rospy.logwarn("Joint limits not provided, cannot generate a random IK seed.")
        return None

    # Generate a random configuration within the joint limits
    random_config = np.random.uniform(lower_limits, upper_limits).tolist()

    # Ensure the configuration is valid (within joint limits)
    for i in range(len(random_config)):
        if random_config[i] < lower_limits[i] or random_config[i] > upper_limits[i]:
            rospy.logwarn(f"Generated configuration out of joint limits for joint {i}")
            return None

    rospy.loginfo(f"Generated random IK seed: {[f'{x:.3f}' for x in random_config]}")
    return random_config 