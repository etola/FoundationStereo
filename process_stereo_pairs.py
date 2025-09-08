#!/usr/bin/env python3
"""
Process stereo pairs from COLMAP reconstructions using FoundationStereo.

This script:
1. Loads COLMAP reconstruction and finds best image pairs
2. Computes stereo rectification for each pair
3. Handles both horizontal and vertical rectifications
4. For vertical rectifications, transposes images to horizontal layout
5. Uses FoundationStereo to compute disparity maps
6. Transforms coordinates back to original image space
7. Generates point clouds from disparity information
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
import torch
import open3d as o3d
from omegaconf import OmegaConf

# Add project root to path
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

from colmap_utils import ColmapReconstruction
from rectify_stereo import (
    compute_stereo_rectification, 
    rectify_images,
    determine_rectification_type,
    transform_coordinates_from_rectified_vectorized
)
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


def convert_numpy_types_for_json(data: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        data: Data that may contain numpy types
        
    Returns:
        Data with numpy types converted to Python native types
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, dict):
        return {key: convert_numpy_types_for_json(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_numpy_types_for_json(item) for item in data]
    else:
        return data


def transpose_image_counter_clockwise(image: np.ndarray) -> np.ndarray:
    """
    Transpose image counter-clockwise (rotate 90 degrees clockwise).
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Transposed image
    """
    return np.rot90(image, k=-1)  # k=-1 rotates counter-clockwise


def transpose_disparity_counter_clockwise(disparity: np.ndarray) -> np.ndarray:
    """
    Transpose disparity map counter-clockwise (rotate 90 degrees clockwise).
    
    Args:
        disparity: Input disparity map as numpy array
        
    Returns:
        Transposed disparity map
    """
    return np.rot90(disparity, k=-1)  # k=-1 rotates counter-clockwise


def transpose_disparity_clockwise(disparity: np.ndarray) -> np.ndarray:
    """
    Transpose disparity map clockwise (rotate 90 degrees clockwise).
    This is the inverse of transpose_image_counter_clockwise.
    
    Args:
        disparity: Input disparity map as numpy array
        
    Returns:
        Transposed disparity map
    """
    return np.rot90(disparity, k=1)  # k=1 rotates clockwise


def process_stereo_pair_with_foundation_stereo(
    left_image: np.ndarray, 
    right_image: np.ndarray, 
    model: Any, 
    args: Any
) -> np.ndarray:
    """
    Process stereo pair using FoundationStereo model.
    
    Args:
        left_image: Left image as numpy array
        right_image: Right image as numpy array
        model: FoundationStereo model
        args: Arguments containing scale and other parameters
        
    Returns:
        Disparity map as numpy array (at scaled resolution)
    """
    # Scale images
    scale = args.scale
    assert scale <= 1, "scale must be <=1"
    
    left_scaled = cv2.resize(left_image, None, fx=scale, fy=scale)
    right_scaled = cv2.resize(right_image, None, fx=scale, fy=scale)
    H_scaled, W_scaled = left_scaled.shape[:2]
        
    # Convert to tensors
    left_tensor = torch.as_tensor(left_scaled).cuda().float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right_scaled).cuda().float()[None].permute(0, 3, 1, 2)
    
    # Pad images
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_padded, right_padded = padder.pad(left_tensor, right_tensor)
    
    # Run model
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(left_padded, right_padded, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(left_padded, right_padded, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    
    # Unpad and convert to numpy
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H_scaled, W_scaled)
    
    # Return disparity at scaled resolution (don't scale back)
    return disp


def remove_invisible_points(disparity: np.ndarray) -> np.ndarray:
    """
    Remove points where disparity would move matches outside image bounds.
    
    Args:
        disparity: Disparity map
        
    Returns:
        Disparity map with invisible points set to inf
    """
    H, W = disparity.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    us_right = xx - disparity
    invalid = us_right < 0
    disparity[invalid] = np.inf
    return disparity


def compute_matching_coordinates_from_disparity(
    disparity: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute matching coordinate arrays from horizontal disparity information.
    For horizontal rectification: right_x = left_x - disparity
    
    Args:
        disparity: Disparity map (at scaled resolution)
        
    Returns:
        Tuple of (left_coords, right_coords) where each is Nx2 array (at scaled resolution)
    """
    H, W = disparity.shape
    
    # Create coordinate grids
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # For horizontal rectification: right_x = left_x - disparity
    left_coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
    right_coords = np.stack([xx.flatten() - disparity.flatten(), yy.flatten()], axis=1)
    
    # Remove invalid points
    valid_mask = np.isfinite(disparity.flatten())
    left_coords = left_coords[valid_mask]
    right_coords = right_coords[valid_mask]
    
    return left_coords, right_coords


def compute_point_cloud_from_matching_coords(
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    left_image: np.ndarray,
    right_image: np.ndarray,
    z_far: float = 10.0
) -> o3d.geometry.PointCloud:
    """
    Compute point cloud from matching coordinates using stereo triangulation.
    
    Args:
        left_coords: Left image coordinates (Nx2)
        right_coords: Right image coordinates (Nx2)
        K1, K2: Camera intrinsic matrices
        R1, R2: Camera rotation matrices
        t1, t2: Camera translation vectors
        left_image: Left image for colors
        right_image: Right image for colors
        z_far: Maximum depth to keep
        
    Returns:
        Open3D point cloud
    """
    # Convert to homogeneous coordinates
    left_coords_hom = np.hstack([left_coords, np.ones((left_coords.shape[0], 1))])
    right_coords_hom = np.hstack([right_coords, np.ones((right_coords.shape[0], 1))])
    
    # Create projection matrices
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    
    # Vectorized triangulation using OpenCV
    # cv2.triangulatePoints expects points in shape (2, N) for each camera
    left_points = left_coords.T.astype(np.float32)  # Shape: (2, N)
    right_points = right_coords.T.astype(np.float32)  # Shape: (2, N)
    
    # Triangulate all points at once
    points_4d = cv2.triangulatePoints(P1, P2, left_points, right_points)
    
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]  # Shape: (3, N)
    points_3d = points_3d.T  # Shape: (N, 3)
    
    # Filter points by depth bounds
    valid_depth_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] <= z_far)
    points_3d = points_3d[valid_depth_mask]
    valid_left_coords = left_coords[valid_depth_mask]
    
    # Get colors from left image (vectorized)
    x_coords = valid_left_coords[:, 0].astype(int)
    y_coords = valid_left_coords[:, 1].astype(int)
    
    # Create mask for valid coordinates
    valid_coords_mask = (
        (x_coords >= 0) & (x_coords < left_image.shape[1]) &
        (y_coords >= 0) & (y_coords < left_image.shape[0])
    )
    
    # Initialize colors array
    colors = np.full((len(valid_left_coords), 3), [128, 128, 128], dtype=np.uint8)
    
    # Extract colors for valid coordinates
    if np.any(valid_coords_mask):
        colors[valid_coords_mask] = left_image[y_coords[valid_coords_mask], x_coords[valid_coords_mask]]
    
    if len(points_3d) == 0:
        return o3d.geometry.PointCloud()
    
    # Create point cloud
    pcd = toOpen3dCloud(points_3d, colors)
    return pcd


def process_single_pair(
    reconstruction: ColmapReconstruction,
    img1_id: int,
    img2_id: int,
    output_dir: Path,
    model: Any,
    args: Any
) -> bool:
    """
    Process a single stereo pair.
    
    Args:
        reconstruction: COLMAP reconstruction
        img1_id: First image ID
        img2_id: Second image ID
        output_dir: Output directory for this pair
        model: FoundationStereo model
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        images_path = output_dir.parent.parent / 'images'

        # Compute stereo rectification
        rect_params = compute_stereo_rectification(reconstruction, img1_id, img2_id, images_path, output_dir)
        
        # Convert numpy types to Python native types for JSON serialization
        rect_params_json = convert_numpy_types_for_json(rect_params)
        
        with open(output_dir / 'rectification.json', 'w') as f:
            json.dump(rect_params_json, f, indent=2)

        logging.info(f"Processing pair: {rect_params['img1_name']} (ID: {img1_id}) and {rect_params['img2_name']} (ID: {img2_id})")

        # Rectify images
        rect1_img, rect2_img = rectify_images(rect_params)

        left_img_path = output_dir / 'left.jpg'
        right_img_path = output_dir / 'right.jpg'
        imageio.imwrite(str(left_img_path), rect1_img)
        imageio.imwrite(str(right_img_path), rect2_img)


        # Process with FoundationStereo (returns disparity at scaled resolution)
        disparity = process_stereo_pair_with_foundation_stereo(
            rect1_img, rect2_img, model, args
        )
        
        # Visualize disparity
        vis = vis_disparity(disparity)
        imageio.imwrite(str(output_dir / "disparity_visualization.png"), vis)

        # Remove invisible points
        # disparity_horizontal = remove_invisible_points(disparity_horizontal)
        
        # Compute matching coordinates from disparity (at scaled resolution)
        img1_coords, img2_coords = compute_matching_coordinates_from_disparity(disparity)
        
        # Scale coordinates up to rectified image resolution
        scale = args.scale
        img1_coords_rect = img1_coords / scale
        img2_coords_rect = img2_coords / scale
        
        # Transform coordinates back to original image space
        img1_coords_orig, img2_coords_orig = transform_coordinates_from_rectified_vectorized(
            rect_params, img1_coords_rect, img2_coords_rect
        )
        
        # Get camera parameters for triangulation
        K1 = reconstruction.get_camera_calibration_matrix(img1_id)
        K2 = reconstruction.get_camera_calibration_matrix(img2_id)
        R1 = reconstruction.get_image_cam_from_world(img1_id).rotation.matrix()
        t1 = reconstruction.get_image_cam_from_world(img1_id).translation
        R2 = reconstruction.get_image_cam_from_world(img2_id).rotation.matrix()
        t2 = reconstruction.get_image_cam_from_world(img2_id).translation
        
        # Load original images for colors
        img1_orig = imageio.imread(rect_params['img1_path'])
        img2_orig = imageio.imread(rect_params['img2_path'])
        
        # Compute point cloud
        pcd = compute_point_cloud_from_matching_coords(
            img1_coords_orig, img2_coords_orig,
            K1, K2, R1, R2, t1, t2,
            img1_orig, img2_orig,
            args.z_far
        )
        
        # Save point cloud
        o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), pcd)
        logging.info(f"Point cloud saved to {output_dir / 'point_cloud.ply'}")
        
        logging.info(f"Successfully processed pair {img1_id}-{img2_id}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to process pair {img1_id}-{img2_id}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Process stereo pairs from COLMAP reconstructions')
    parser.add_argument('-s', '--scene_folder', required=True,
                       help='Path to scene folder containing sparse/ and images/')
    parser.add_argument('-o', '--output_folder', required=True,
                       help='Output folder name (will be created under scene_folder)')
    parser.add_argument('--scale', default=0.25, type=float,
                       help='Scale factor for image processing (default: 0.25)')
    parser.add_argument('--ckpt_dir', 
                       default=f'{code_dir}/pretrained_models/23-51-11/model_best_bp2.pth',
                       type=str, help='Pretrained model path')
    parser.add_argument('--hiera', default=0, type=int,
                       help='Hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float,
                       help='Maximum depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32,
                       help='Number of flow-field updates during forward pass')
    parser.add_argument('--min_points', type=int, default=100,
                       help='Minimum number of 3D points for pair selection')
    parser.add_argument('--pairs_per_image', type=int, default=1,
                       help='Number of pairs to select per image')
    
    args = parser.parse_args()
    
    # Validate paths
    scene_folder = Path(args.scene_folder)
    if not scene_folder.exists():
        logging.error(f"Scene folder {scene_folder} does not exist")
        sys.exit(1)
    
    sparse_path = scene_folder / 'sparse'
    images_path = scene_folder / 'images'
    
    if not sparse_path.exists():
        logging.error(f"Sparse reconstruction folder {sparse_path} does not exist")
        sys.exit(1)
    
    if not images_path.exists():
        logging.error(f"Images folder {images_path} does not exist")
        sys.exit(1)
    
    # Setup logging
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    
    # Load COLMAP reconstruction
    logging.info(f"Loading COLMAP reconstruction from {sparse_path}")
    reconstruction = ColmapReconstruction(str(sparse_path))
    
    # Get best image pairs
    logging.info("Finding best image pairs...")
    pairs = reconstruction.get_best_pairs(
        min_points=args.min_points,
        pairs_per_image=args.pairs_per_image
    )
    
    # Convert pairs to list of tuples
    pair_list = []
    for img1_id, partner_ids in pairs.items():
        for img2_id in partner_ids:
            if img1_id < img2_id:  # Avoid duplicates
                pair_list.append((img1_id, img2_id))
    
    logging.info(f"Found {len(pair_list)} stereo pairs to process")
    
    # Load FoundationStereo model
    logging.info(f"Loading FoundationStereo model from {args.ckpt_dir}")
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    
    # Create output directory
    output_dir = scene_folder / args.output_folder
    output_dir.mkdir(exist_ok=True)
    
    # Process each pair
    successful_pairs = 0
    for i, (img1_id, img2_id) in enumerate(pair_list):
        logging.info(f"\n=== Processing pair {i+1}/{len(pair_list)}: {img1_id}-{img2_id} ===")
        
        # Create pair-specific output directory
        pair_output_dir = output_dir / f"pair_{i:02d}"
        pair_output_dir.mkdir(exist_ok=True)
        
        # Process the pair
        success = process_single_pair(
            reconstruction, img1_id, img2_id, pair_output_dir, model, args
        )
        
        if success:
            successful_pairs += 1
        break
    
    logging.info(f"\nCompleted processing {successful_pairs}/{len(pair_list)} pairs successfully")
    logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
