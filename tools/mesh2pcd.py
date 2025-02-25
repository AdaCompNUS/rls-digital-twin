#!/usr/bin/env python3
"""
DAE Mesh to Uniform Point Cloud Converter for Motion Planning

This script converts DAE (COLLADA) files to uniform point clouds
optimized for indoor environments in motion planning.
"""

import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import open3d as o3d
import time


def convert_mesh_to_pointcloud(
    mesh_path,
    output_path=None,
    voxel_size=0.02,
    poisson_sampling=True,
    blue_noise=True,
    uniform_density=True,
    visualize=False,
):
    """
    Convert a mesh file to a uniform point cloud with advanced sampling.

    Args:
        mesh_path (str): Path to the mesh file (DAE, OBJ, STL, etc.)
        output_path (str): Path to save the output point cloud (in PLY format)
        voxel_size (float): Base size for sampling resolution (smaller = more detailed)
        poisson_sampling (bool): Use Poisson disk sampling for initial uniformity
        blue_noise (bool): Apply blue noise sampling for better distribution
        uniform_density (bool): Apply density equalization
        visualize (bool): Whether to visualize the resulting point cloud

    Returns:
        numpy.ndarray: Uniform point cloud as Nx3 array
    """
    print(f"Loading mesh from {mesh_path}")
    start_time = time.time()

    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # Handle scene with multiple meshes
    if isinstance(mesh, trimesh.Scene):
        print("Loaded a scene, extracting and combining meshes...")
        meshes = []
        for _, geometry in mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        if not meshes:
            raise ValueError("No valid meshes found in the scene")

        # Combine all meshes
        mesh = trimesh.util.concatenate(meshes)
    elif not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")

    print("Mesh statistics:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Surface area: {mesh.area:.2f} square units")
    print(f"  Bounding box dimensions: {mesh.extents}")

    # Convert to Open3D mesh for advanced sampling
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Step 1: Initial sampling - choose method based on parameters
    points = []

    if poisson_sampling:
        print("Performing Poisson disk sampling for initial point distribution...")
        # Calculate number of points based on surface area and resolution
        N = int(mesh.area / (voxel_size * voxel_size) * 0.8)
        # Use Poisson disk sampling for uniform initial distribution
        poisson_pcd = o3d_mesh.sample_points_poisson_disk(N, init_factor=5)
        points = np.asarray(poisson_pcd.points)
        print(f"Poisson disk sampling generated {len(points)} points")
    else:
        # Uniform sampling
        N = int(mesh.area / (voxel_size * voxel_size))
        uniform_pcd = o3d_mesh.sample_points_uniformly(N)
        points = np.asarray(uniform_pcd.points)
        print(f"Uniform sampling generated {len(points)} points")

    # Step 2: Apply blue noise sampling if requested
    if blue_noise and len(points) > 0:
        print("Applying blue noise sampling for improved uniformity...")
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Voxel downsample first to reduce points
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Run statistical outlier removal to get more uniform distribution
        filtered, _ = downsampled.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        # Convert back to numpy
        points = np.asarray(filtered.points)
        print(f"After blue noise filtering: {len(points)} points")

    # Step 3: Density equalization
    if uniform_density and len(points) > 0:
        print("Performing density equalization...")

        # Build KD-tree for neighbor searches
        kdtree = cKDTree(points)

        # Compute point density (using inverse of distance to nearest neighbors)
        k_neighbors = min(30, len(points) - 1)
        distances, _ = kdtree.query(
            points, k=k_neighbors + 1
        )  # +1 because first is self

        # Average distance to neighbors (excluding self)
        mean_distances = np.mean(distances[:, 1:], axis=1)

        # Density is inversely proportional to distance
        densities = 1.0 / (mean_distances + 1e-10)

        # Normalize densities
        normalized_densities = densities / np.max(densities)

        # Calculate probability of keeping each point (inversely proportional to density)
        # Points in high-density areas have lower probability of being kept
        keep_probabilities = 1.0 - (normalized_densities * 0.8)

        # Randomly select points based on their keep probability
        keep_mask = np.random.random(len(points)) < keep_probabilities
        equalized_points = points[keep_mask]

        if (
            len(equalized_points) > len(points) / 3
        ):  # Ensure we don't remove too many points
            points = equalized_points
            print(f"After density equalization: {len(points)} points")
        else:
            print("Density equalization removed too many points, skipping this step")

    # Step 4: Final uniform voxel grid downsampling to ensure consistency
    print("Performing final voxel grid downsampling...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(downsampled.points)

    print("\nPoint cloud analysis:")
    print(f"  Final point cloud size: {len(points)} points")
    print(f"  Processing time: {time.time() - start_time:.2f} seconds")

    # Evaluate uniformity
    if len(points) > 0:
        print("\nEvaluating point cloud uniformity...")
        kdtree = cKDTree(points)
        k = min(6, len(points) - 1)
        distances, _ = kdtree.query(points, k=k + 1)
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

        uniformity_score = np.std(mean_distances) / np.mean(mean_distances)
        print(f"  Uniformity score: {uniformity_score:.4f} (lower is better)")
        print(f"  Min neighbor distance: {np.min(mean_distances):.4f}")
        print(f"  Max neighbor distance: {np.max(mean_distances):.4f}")
        print(f"  Average neighbor distance: {np.mean(mean_distances):.4f}")

    # Save the point cloud if output path is specified
    if output_path and len(points) > 0:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save as PLY file
        if not output_path.endswith(".ply"):
            output_path += ".ply"

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud to {output_path}")

    # Visualize the point cloud if requested
    if visualize and len(points) > 0:
        print("Visualizing point cloud... (Close window to continue)")
        # Create point cloud for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 0.651, 0.929])  # Blue color

        # Create coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )

        # Visualize
        o3d.visualization.draw_geometries(
            [pcd, coord_frame], window_name="Uniform Point Cloud Result"
        )

    return points


def main():
    parser = argparse.ArgumentParser(
        description="Convert mesh to uniform point cloud for motion planning"
    )
    parser.add_argument("input", help="Input mesh file path (DAE, OBJ, STL, etc.)")
    parser.add_argument(
        "-o", "--output", help="Output point cloud file path (PLY format)"
    )
    parser.add_argument(
        "-v",
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel size for sampling resolution (default: 0.02)",
    )
    parser.add_argument(
        "--no-poisson", action="store_true", help="Disable Poisson disk sampling"
    )
    parser.add_argument(
        "--no-blue-noise", action="store_true", help="Disable blue noise sampling"
    )
    parser.add_argument(
        "--no-uniform-density", action="store_true", help="Disable density equalization"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the resulting point cloud"
    )

    args = parser.parse_args()

    # Generate default output paths if not specified
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_pointcloud.ply"

    # Convert mesh to point cloud
    convert_mesh_to_pointcloud(
        args.input,
        args.output,
        voxel_size=args.voxel_size,
        poisson_sampling=not args.no_poisson,
        blue_noise=not args.no_blue_noise,
        uniform_density=not args.no_uniform_density,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
