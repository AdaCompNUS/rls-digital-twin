import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_pose_to_world(base_pos, base_yaw, ee_pos, ee_quat):
    """
    Transform end effector pose from robot base frame to world frame.

    Args:
        base_pos: [x, y, z] position of the robot base in world frame
        base_yaw: Yaw angle of the robot base in world frame
        ee_pos: [x, y, z] position of the end effector in robot base frame
        ee_quat: [x, y, z, w] quaternion of the end effector in robot base frame

    Returns:
        world_pos: [x, y, z] position of the end effector in world frame
        world_quat: [x, y, z, w] quaternion of the end effector in world frame
    """
    # Create base rotation matrix from yaw angle
    base_rot = R.from_euler("z", base_yaw)
    base_rot_matrix = base_rot.as_matrix()

    # Transform end effector position to world frame
    ee_pos_rotated = base_rot_matrix @ np.array(ee_pos)
    world_pos = np.array(
        [
            base_pos[0] + ee_pos_rotated[0],
            base_pos[1] + ee_pos_rotated[1],
            # base_pos[2] + ee_pos_rotated[2], # Original assumption base_pos[2] = 0 might not hold
            ee_pos_rotated[2] # Assuming Z relative to base_pos is what matters
        ]
    )

    # Transform end effector orientation to world frame
    ee_rot = R.from_quat([ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]])
    # base_rot is already calculated
    world_rot = base_rot * ee_rot
    world_quat = world_rot.as_quat()  # xyzw format

    return world_pos.tolist(), world_quat.tolist()

def transform_pose_to_base(world_pos, world_quat, base_pos, base_yaw):
    """
    Transform end effector pose from world frame to robot base frame.

    Args:
        world_pos: [x, y, z] position of the end effector in world frame
        world_quat: [x, y, z, w] quaternion of the end effector in world frame
        base_pos: [x, y, z] position of the robot base in world frame
        base_yaw: Yaw angle of the robot base in world frame

    Returns:
        ee_pos: [x, y, z] position of the end effector in robot base frame
        ee_quat: [x, y, z, w] quaternion of the end effector in robot base frame
    """
    # Create inverse base rotation
    base_rot = R.from_euler("z", base_yaw)
    base_rot_inv = base_rot.inv()

    # Translate world position to base origin
    pos_rel_to_base = np.array(
        [
            world_pos[0] - base_pos[0],
            world_pos[1] - base_pos[1],
            world_pos[2] - base_pos[2] # Account for base Z
        ]
    )

    # Rotate to base frame
    ee_pos = base_rot_inv.apply(pos_rel_to_base)

    # Transform orientation
    world_rot = R.from_quat(world_quat)
    ee_rot = base_rot_inv * world_rot
    ee_quat = ee_rot.as_quat()

    return ee_pos.tolist(), ee_quat.tolist()

def quaternion_matrix(quaternion):
    """
    Converts a quaternion [x, y, z, w] to a 4x4 transformation matrix.
    """
    x, y, z, w = quaternion
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0
    
    wx, wy, wz = w * s, w * s, w * s
    xx, xy, xz = x * s, x * s, x * s
    yy, yz, zz = y * s, y * s, z * s
    
    matrix = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
        [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
        [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return matrix

def translation_from_matrix(matrix):
    """
    Returns the translation part of a 4x4 transformation matrix.
    """
    return matrix[:3, 3].tolist()

def quaternion_from_matrix(matrix):
    """
    Converts the rotation part of a 4x4 matrix to a quaternion [x, y, z, w].
    """
    M = matrix[:3, :3]
    t = np.trace(M)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (M[2, 1] - M[1, 2]) * s
        y = (M[0, 2] - M[2, 0]) * s
        z = (M[1, 0] - M[0, 1]) * s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = 2.0 * np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2])
        w = (M[2, 1] - M[1, 2]) / s
        x = 0.25 * s
        y = (M[0, 1] + M[1, 0]) / s
        z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = 2.0 * np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2])
        w = (M[0, 2] - M[2, 0]) / s
        x = (M[0, 1] + M[1, 0]) / s
        y = 0.25 * s
        z = (M[1, 2] + M[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1])
        w = (M[1, 0] - M[0, 1]) / s
        x = (M[0, 2] + M[2, 0]) / s
        y = (M[1, 2] + M[2, 1]) / s
        z = 0.25 * s
        
    return [x, y, z, w] 