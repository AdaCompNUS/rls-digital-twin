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
    if costmap is None or costmap_metadata is None:
        rospy.logwarn("Costmap not available for finding valid base positions")
        return None

    # Convert end effector position to grid coordinates
    ee_grid_x, ee_grid_y = world_to_grid(
        ee_pose.position.x, ee_pose.position.y, costmap_metadata
    )

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
            f"End effector position ({ee_pose.position.x}, {ee_pose.position.y}) is outside costmap bounds"
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
                    continue # Skip if conversion failed

                # Calculate orientation towards the end effector
                dx = ee_pose.position.x - world_x
                dy = ee_pose.position.y - world_y
                theta = math.atan2(dy, dx)

                # Get the cost value (lower is better for sampling)
                cost_value = costmap[y, x]

                valid_positions.append((world_x, world_y, cost_value, theta))

    if not valid_positions:
        rospy.logwarn("No valid base positions found")
    else:
        rospy.loginfo(f"Found {len(valid_positions)} valid base positions")

    return valid_positions

def generate_ik_seed(pose, costmap, costmap_metadata, lower_limits, upper_limits, manipulation_radius=1.0):
    """
    Generate a deterministic seed for IK using costmap for base position.
    Selects the best (lowest cost) base position from the costmap if available.

    Args:
        pose: Target end effector pose (geometry_msgs/Pose)
        costmap: The costmap numpy array.
        costmap_metadata: Dictionary containing costmap metadata.
        lower_limits: List of lower joint limits (including base).
        upper_limits: List of upper joint limits (including base).
        manipulation_radius: Radius for base position sampling.

    Returns:
        seed: List representing the generated IK seed (base + arm joints).
    """
    rospy.loginfo("Generating deterministic seed for IK...")

    # Ensure limits are provided and have the expected length (at least 11 for Fetch base+arm)
    if lower_limits is None or upper_limits is None or len(lower_limits) < 11 or len(upper_limits) < 11:
         rospy.logerr(f"Invalid or missing joint limits provided for IK seed generation. Got lengths: {len(lower_limits) if lower_limits else 'None'}, {len(upper_limits) if upper_limits else 'None'}")
         # Fallback to a default seed structure if limits are invalid
         num_dof = 11 # Assume 11 DoF if limits are missing
         seed = [0.0] * num_dof
         rospy.logwarn("Falling back to default zero seed due to missing/invalid limits.")
         return seed

    # Initialize the seed array with midpoints as fallback
    seed = [
        (l + u) / 2.0 for l, u in zip(lower_limits, upper_limits)  # noqa: E741
    ]

    # Try to use costmap for base position sampling
    if costmap is not None and costmap_metadata is not None:
        valid_positions = find_valid_base_positions(
            pose, costmap, costmap_metadata, manipulation_radius, cost_threshold=0.3
        )

        if valid_positions and len(valid_positions) > 0:
            # Sort positions by cost (ascending order - lower cost is better)
            sorted_positions = sorted(valid_positions, key=lambda pos: pos[2])

            # Take the position with lowest cost
            best_position = sorted_positions[0]
            base_x, base_y, _, base_theta = best_position

            # Update the first 3 elements of the seed (base position and orientation)
            # Clamp base position/orientation to limits if necessary
            seed[0] = np.clip(base_x, lower_limits[0], upper_limits[0])
            seed[1] = np.clip(base_y, lower_limits[1], upper_limits[1])
            seed[2] = np.clip(base_theta, lower_limits[2], upper_limits[2])


            rospy.loginfo(
                f"Using best base position from costmap: [{seed[0]:.4f}, {seed[1]:.4f}, {seed[2]:.4f}]"
            )
        else:
            rospy.logwarn("No valid base positions found in costmap or costmap not used. Using default mid-point base seed.")
            # Base seed remains the midpoint calculated initially
            rospy.loginfo(
                f"Using mid-point base seed: [{seed[0]:.4f}, {seed[1]:.4f}, {seed[2]:.4f}]"
            )

    else:
         rospy.logwarn("Costmap not available. Using default mid-point base seed.")
         rospy.loginfo(
             f"Using mid-point base seed: [{seed[0]:.4f}, {seed[1]:.4f}, {seed[2]:.4f}]"
         )

    # Use midpoint values for arm joints (remaining DOFs) - indices 3 through 10 for 8-DOF arm
    # The initial seed calculation already set these to midpoints.
    # Optionally, use random uniform sampling within limits for arm joints:
    # for i in range(3, len(seed)): # Iterate through arm joints
    #    if i < len(lower_limits): # Check index bounds
    #        seed[i] = np.random.uniform(lower_limits[i], upper_limits[i])

    rospy.loginfo(
        f"Deterministic seed generated: {[round(val, 3) for val in seed]}"
    )
    return seed 