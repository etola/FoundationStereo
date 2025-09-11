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
from typing import Dict, List, Tuple, Any, Optional
import cv2
import torch
import open3d as o3d
import imageio.v2 as imageio
from omegaconf import OmegaConf

# Add project root to path
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

from colmap_utils import ColmapReconstruction
from rectify_stereo import (
    initalize_rectification,
    save_rectified_intrinsics,
    save_rectification_json,
    rectify_images,
    transform_coordinates_from_rectified_vectorized
)
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


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
    disparity: np.ndarray,
    select_points: int = 0,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute matching coordinate arrays from horizontal disparity information.
    For horizontal rectification: right_x = left_x - disparity
    
    Args:
        disparity: Disparity map (at scaled resolution)
        select_points: If > 0, perform N×N uniform sampling where N = select_points
        
    Returns:
        Tuple of (left_coords, right_coords, selected_coord_tuple) where:
        - left_coords: Nx2 array of left image coordinates (at scaled resolution)
        - right_coords: Nx2 array of right image coordinates (at scaled resolution)  
        - selected_coord_tuple: Mx3 array of (x, y, disparity) for uniformly selected points, or None
    """
    H, W = disparity.shape
    
    # Create coordinate grids
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Remove invalid points
    valid_mask = np.isfinite(disparity.flatten())
    
    # For horizontal rectification: right_x = left_x - disparity
    left_coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
    right_coords = np.stack([xx.flatten() - disparity.flatten(), yy.flatten()], axis=1)
    
    # Filter out points where right_x would be negative
    valid_mask = valid_mask & (right_coords[:, 0] >= 0)

    left_coords = left_coords[valid_mask]
    right_coords = right_coords[valid_mask]
    
    # Uniform N×N point selection if requested
    selected_coords = None
    if select_points > 0:
        # Create uniform grid of sample points
        y_indices = np.linspace(0, H-1, select_points, dtype=int)
        x_indices = np.linspace(0, W-1, select_points, dtype=int)
        
        # Vectorized coordinate selection with random disturbance
        y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
        grid_coords = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
        
        # Apply random disturbance to all coordinates at once
        disturbance = np.random.randint(-10, 11, size=grid_coords.shape)
        coords_disturbed = np.clip(grid_coords + disturbance, [0, 0], [W-1, H-1])
        
        # Sample disparity values for all coordinates
        disp_vals = disparity[coords_disturbed[:, 1], coords_disturbed[:, 0]]
        
        # Create validity mask
        valid_mask = (
            np.isfinite(disp_vals) & 
            ((coords_disturbed[:, 0] - disp_vals) >= 0)
        )
        
        # Filter valid coordinates and create final array
        if np.any(valid_mask):
            valid_coords = coords_disturbed[valid_mask]
            valid_disp = disp_vals[valid_mask]
            selected_coords = np.column_stack([valid_coords, valid_disp])
        else:
            selected_coords = None

        if verbose:
            print(f"Selected {np.sum(valid_mask)} points")
    
    return left_coords, right_coords, selected_coords


def create_validity_mask_from_padding(rect_params: dict, image_shape: Tuple[int, int], image_id: int = 1) -> np.ndarray:
    """
    Create validity mask for rectified image based on padding information.
    
    Args:
        rect_params: Rectification parameters containing custom_padding info
        image_shape: Shape of the rectified image (H, W)
        image_id: 1 for left image, 2 for right image
        
    Returns:
        Validity mask where True=valid pixels, False=padded/invalid pixels
    """
    H, W = image_shape
    mask = np.ones((H, W), dtype=bool)
    
    # If no custom padding, entire image is valid
    if 'custom_padding' not in rect_params:
        return mask
    
    padding_info = rect_params['custom_padding']
    
    # Get padding values for the specified image
    if image_id == 1:
        pad_top = padding_info.get('pad_top_1', 0)
        pad_bottom = padding_info.get('pad_bottom_1', 0)
        pad_right = padding_info.get('pad_right_1', 0)
    else:
        pad_top = padding_info.get('pad_top_2', 0)
        pad_bottom = padding_info.get('pad_bottom_2', 0)
        pad_right = padding_info.get('pad_right_2', 0)
    
    # Mark padded regions as invalid
    if pad_top > 0:
        mask[:pad_top, :] = False  # Top padding
    if pad_bottom > 0:
        mask[H-pad_bottom:, :] = False  # Bottom padding
    if pad_right > 0:
        mask[:, W-pad_right:] = False  # Right padding
    
    return mask


def apply_validity_masks_to_disparity(disparity: np.ndarray, rect_params: dict, 
                                     left_image_shape: Tuple[int, int], right_image_shape: Tuple[int, int],
                                     scale_factor: float = 1.0, output_dir: Path = None, verbose: bool = False) -> np.ndarray:
    """
    Apply validity masks to disparity map to exclude padded regions from both left and right images.
    
    Args:
        disparity: Disparity map to apply masks to
        rect_params: Rectification parameters containing custom_padding info
        left_image_shape: Shape of the left rectified image (H, W)
        right_image_shape: Shape of the right rectified image (H, W)
        scale_factor: Scale factor applied to disparity computation
        output_dir: Optional directory to save mask visualizations
        verbose: Whether to save mask visualizations
        
    Returns:
        Modified disparity map with invalid regions set to np.inf
    """
    # Apply validity masks to disparity to exclude padded regions
    validity_mask_left = create_validity_mask_from_padding(rect_params, left_image_shape, image_id=1)
    validity_mask_right = create_validity_mask_from_padding(rect_params, right_image_shape, image_id=2)
    
    # Scale masks to match disparity resolution
    if scale_factor != 1.0:
        validity_mask_left_scaled = cv2.resize(validity_mask_left.astype(np.uint8), 
                                             (disparity.shape[1], disparity.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST).astype(bool)
        validity_mask_right_scaled = cv2.resize(validity_mask_right.astype(np.uint8), 
                                              (disparity.shape[1], disparity.shape[0]), 
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        validity_mask_left_scaled = validity_mask_left
        validity_mask_right_scaled = validity_mask_right
    
    # Create coordinate grids for disparity validation
    H, W = disparity.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Calculate corresponding right image coordinates
    right_x = xx - disparity
    
    # Check validity for both left and right image positions
    valid_left = validity_mask_left_scaled  # Left image validity
    
    # Vectorized validity check for right image
    # Create valid mask for finite disparity values and in-bounds coordinates
    finite_disparity = np.isfinite(disparity)
    in_bounds = (right_x >= 0) & (right_x < W)
    
    # Initialize right validity mask
    valid_right = np.zeros_like(validity_mask_left_scaled, dtype=bool)
    
    # For valid coordinates, sample from right validity mask
    valid_coords = finite_disparity & in_bounds
    if np.any(valid_coords):
        # Get row and column indices for valid coordinates
        valid_rows, valid_cols = np.where(valid_coords)
        right_x_coords = right_x[valid_coords].astype(int)
        
        # Sample right validity mask at corresponding coordinates
        valid_right[valid_rows, valid_cols] = validity_mask_right_scaled[valid_rows, right_x_coords]
    
    # Combined validity: both left pixel and corresponding right pixel must be valid
    combined_validity = valid_left & valid_right
    
    # Set invalid regions to inf in disparity map
    disparity[~combined_validity] = np.inf

    left_invalid = np.sum(~valid_left)
    right_invalid = np.sum(~valid_right)
    total_invalid = np.sum(~combined_validity)
    logging.info(f"Applied validity masks: {left_invalid} pixels invalid in left, {right_invalid} in right, {total_invalid} total invalid")
    
    # Save validity mask visualizations (only if verbose mode is enabled)
    if verbose and output_dir is not None:
        mask_left_vis = (valid_left * 255).astype(np.uint8)
        mask_right_vis = (valid_right * 255).astype(np.uint8)
        mask_combined_vis = (combined_validity * 255).astype(np.uint8)
        
        imageio.imwrite(str(output_dir / "validity_mask_left.png"), mask_left_vis)
        imageio.imwrite(str(output_dir / "validity_mask_right.png"), mask_right_vis)
        imageio.imwrite(str(output_dir / "validity_mask_combined.png"), mask_combined_vis)
        logging.info("Saved validity mask visualizations (verbose mode)")
    elif not verbose:
        logging.debug("Validity mask visualizations not saved (use --verbose to enable)")
    
    return disparity


def visualize_matches(left_image: np.ndarray, right_image: np.ndarray, 
                     left_coords: np.ndarray, right_coords: np.ndarray,
                     max_matches: int = 1000) -> np.ndarray:
    """
    Visualize matching points between two images by concatenating them and drawing lines.
    
    Args:
        left_image: Left image as numpy array (H, W, 3)
        right_image: Right image as numpy array (H, W, 3)
        left_coords: Left image coordinates as Nx2 array [x, y]
        right_coords: Right image coordinates as Nx2 array [x, y]
        max_matches: Maximum number of matches to draw (for visual clarity)
        
    Returns:
        Visualization image with both images concatenated and matches drawn
    """
    # Ensure images have same height for concatenation
    h1, w1 = left_image.shape[:2]
    h2, w2 = right_image.shape[:2]
    
    if h1 != h2:
        # Resize to same height
        target_height = max(h1, h2)
        if h1 != target_height:
            left_image = cv2.resize(left_image, (int(w1 * target_height / h1), target_height))
        if h2 != target_height:
            right_image = cv2.resize(right_image, (int(w2 * target_height / h2), target_height))
        
        # Update dimensions
        h1, w1 = left_image.shape[:2]
        h2, w2 = right_image.shape[:2]
    
    # Concatenate images horizontally
    concat_image = np.hstack([left_image, right_image])
    
    # Limit number of matches for visual clarity
    num_matches = min(len(left_coords), len(right_coords), max_matches)
    if num_matches == 0:
        return concat_image
    
    # Sample matches uniformly if we have more than max_matches
    if len(left_coords) > max_matches:
        indices = np.linspace(0, len(left_coords) - 1, max_matches, dtype=int)
        left_coords_vis = left_coords[indices]
        right_coords_vis = right_coords[indices]
    else:
        left_coords_vis = left_coords
        right_coords_vis = right_coords
    
    # Draw matches
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, (left_pt, right_pt) in enumerate(zip(left_coords_vis, right_coords_vis)):
        # Convert to integers
        left_x, left_y = int(left_pt[0]), int(left_pt[1])
        right_x, right_y = int(right_pt[0]), int(right_pt[1])
        
        # Offset right coordinates by left image width
        right_x_offset = right_x + w1
        
        # Check bounds
        if (0 <= left_x < w1 and 0 <= left_y < h1 and 
            0 <= right_x < w2 and 0 <= right_y < h2):
            
            color = colors[i % len(colors)]
            
            # Draw circles on keypoints
            cv2.circle(concat_image, (left_x, left_y), 10, color, -1)
            cv2.circle(concat_image, (right_x_offset, right_y), 10, color, -1)
            
            # Draw line connecting the matches
            cv2.line(concat_image, (left_x, left_y), (right_x_offset, right_y), color, 5)
    
    return concat_image


def compute_point_cloud_from_matching_coords(
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    reconstruction: ColmapReconstruction,
    img1_id: int,
    img2_id: int,
    left_image: np.ndarray,
    right_image: np.ndarray,
    args: Any,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """
    Compute point cloud from matching coordinates using stereo triangulation.
    
    Args:
        left_coords: Left image coordinates (Nx2)
        right_coords: Right image coordinates (Nx2)
        reconstruction: ColmapReconstruction object
        img1_id: First image ID
        img2_id: Second image ID
        left_image: Left image for colors
        right_image: Right image for colors
        args: Arguments containing z_far and other parameters
        bbox_min: Optional minimum bounding box coordinates (3D)
        bbox_max: Optional maximum bounding box coordinates (3D)
        
    Returns:
        Open3D point cloud
    """
    # Get camera parameters for triangulation
    K1 = reconstruction.get_camera_calibration_matrix(img1_id)
    K2 = reconstruction.get_camera_calibration_matrix(img2_id)
    R1 = reconstruction.get_image_cam_from_world(img1_id).rotation.matrix()
    t1 = reconstruction.get_image_cam_from_world(img1_id).translation
    R2 = reconstruction.get_image_cam_from_world(img2_id).rotation.matrix()
    t2 = reconstruction.get_image_cam_from_world(img2_id).translation
    
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
    valid_depth_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] <= args.z_far)
    
    # Apply bounding box filtering if provided
    if bbox_min is not None and bbox_max is not None:
        bbox_mask = (
            (points_3d[:, 0] >= bbox_min[0]) & (points_3d[:, 0] <= bbox_max[0]) &
            (points_3d[:, 1] >= bbox_min[1]) & (points_3d[:, 1] <= bbox_max[1]) &
            (points_3d[:, 2] >= bbox_min[2]) & (points_3d[:, 2] <= bbox_max[2])
        )
        combined_mask = valid_depth_mask & bbox_mask
        logging.info(f"Point filtering: {np.sum(valid_depth_mask)} points passed depth filter, "
                    f"{np.sum(bbox_mask)} passed bbox filter, {np.sum(combined_mask)} passed both")
    else:
        combined_mask = valid_depth_mask
        logging.info(f"Point filtering: {np.sum(valid_depth_mask)} points passed depth filter (no bbox filter)")
    
    points_3d = points_3d[combined_mask]
    valid_left_coords = left_coords[combined_mask]
    
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

    if args.denoise_cloud:
        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        pcd = inlier_cloud

    return pcd


def process_single_pair(
    reconstruction: ColmapReconstruction,
    u_img1_id: int,
    u_img2_id: int,
    output_dir: Path,
    model: Any,
    args: Any,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None
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
        bbox_min: Optional minimum bounding box coordinates (3D)
        bbox_max: Optional maximum bounding box coordinates (3D)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        images_path = output_dir.parent.parent / 'images'

        # Compute stereo rectification
        rect_params = initalize_rectification(reconstruction, u_img1_id, u_img2_id, images_path, output_dir, verbose=args.verbose)
        img1_id = rect_params['img1_id']
        img2_id = rect_params['img2_id']
        save_rectified_intrinsics(output_dir, rect_params)
        save_rectification_json(output_dir, rect_params)

        logging.info(f"Processing pair: {rect_params['img1_name']} (ID: {img1_id}) and {rect_params['img2_name']} (ID: {img2_id})")

        # Rectify images
        print("Rectifying images...")
        rect1_img, rect2_img = rectify_images(rect_params)

        left_img_path = output_dir / 'left.jpg'
        right_img_path = output_dir / 'right.jpg'
        imageio.imwrite(str(left_img_path), rect1_img)
        imageio.imwrite(str(right_img_path), rect2_img)
        print("Rectifying images done")


        print("Disparity computation...")
        # Process with FoundationStereo (returns disparity at scaled resolution)
        disparity = process_stereo_pair_with_foundation_stereo(
            rect1_img, rect2_img, model, args
        )
        print("Disparity computation done")


        print("Applying validity masks to disparity...")
        disparity = apply_validity_masks_to_disparity(
            disparity, rect_params, rect1_img.shape[:2], rect2_img.shape[:2], 
            scale_factor=args.scale, output_dir=output_dir, verbose=args.verbose
        )
        print("Applying validity masks to disparity done")


        # Visualize disparity
        vis = vis_disparity(disparity)
        imageio.imwrite(str(output_dir / "disparity_visualization.png"), vis)

        
        # Compute matching coordinates from disparity (at scaled resolution)
        img1_coords, img2_coords, selected_coords = compute_matching_coordinates_from_disparity(
            disparity, select_points=args.select_points, verbose=args.verbose
        )
        
        # Scale coordinates up to rectified image resolution
        scale = args.scale
        img1_coords_rect = img1_coords / scale
        img2_coords_rect = img2_coords / scale
        
        # Visualize matches on rectified images if uniform sampling was used
        if selected_coords is not None and len(selected_coords) > 0:
            # Generate matching coordinates for selected points in rectified space
            selected_left_coords = selected_coords[:, :2] / scale  # [x, y] scaled to rect resolution
            selected_right_coords = np.column_stack([
                selected_coords[:, 0] / scale - selected_coords[:, 2] / scale,  # x - disparity, scaled
                selected_coords[:, 1] / scale  # y, scaled
            ])
            
            # Create rectified match visualization
            rect_match_vis = visualize_matches(
                rect1_img, rect2_img, 
                selected_left_coords, selected_right_coords
            )
            imageio.imwrite(str(output_dir / "rectified_matches.png"), rect_match_vis)
            logging.info(f"Rectified matches visualization saved with {len(selected_coords)} points")
        
        # Transform coordinates back to original image space
        img1_coords_orig, img2_coords_orig = transform_coordinates_from_rectified_vectorized(
            rect_params, img1_coords_rect, img2_coords_rect
        )
        
        
        # Load original images for colors
        img1_orig = imageio.imread(rect_params['img1_path'])
        img2_orig = imageio.imread(rect_params['img2_path'])
        
        # Visualize matches on original images if uniform sampling was used
        if selected_coords is not None and len(selected_coords) > 0:
            # Transform selected coordinates to original image space
            selected_img1_coords_orig, selected_img2_coords_orig = transform_coordinates_from_rectified_vectorized(
                rect_params, selected_left_coords, selected_right_coords
            )
            
            # Create original match visualization
            orig_match_vis = visualize_matches(
                img1_orig, img2_orig,
                selected_img1_coords_orig, selected_img2_coords_orig
            )
            imageio.imwrite(str(output_dir / "original_matches.png"), orig_match_vis)
            logging.info(f"Original matches visualization saved with {len(selected_coords)} points")
        
        
        # Compute point cloud
        pcd = compute_point_cloud_from_matching_coords(
            img1_coords_orig, img2_coords_orig,
            reconstruction, img1_id, img2_id,
            img1_orig, img2_orig,
            args,
            bbox_min=bbox_min,
            bbox_max=bbox_max
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
    parser.add_argument('-s', '--scene_folder', required=True, help='Path to scene folder containing sparse/ and images/')
    parser.add_argument('-o', '--output_folder', required=True, help='Output folder name (will be created under scene_folder)')
    parser.add_argument('--scale', default=0.25, type=float, help='Scale factor for image processing (default: 0.25)')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='Pretrained model path')
    parser.add_argument('--hiera', default=0, type=int, help='Hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=100, type=float, help='Maximum depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='Number of flow-field updates during forward pass')
    parser.add_argument('--min_points', type=int, default=100, help='Minimum number of 3D points for pair selection')
    parser.add_argument('--pairs_per_image', type=int, default=1, help='Number of pairs to select per image')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    parser.add_argument('--denoise_cloud', type=int, default=0, help='whether to denoise the point cloud')
    parser.add_argument('--pair', nargs=2, type=int, metavar=('IMG1_ID', 'IMG2_ID'), help='Process a single image pair with the specified frame IDs (e.g., --pair 11 10)')
    parser.add_argument('--select_points', type=int, default=0, help='Number of uniform grid points for match visualization (N×N grid, 0=disabled)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output and save additional debug visualizations (validity masks)')
    parser.add_argument('--disable_bbox_filter', action='store_true', help='Disable point cloud filtering using robust bounding box from COLMAP reconstruction')
    parser.add_argument('--bbox_min_visibility', type=int, default=3, help='Minimum visibility for points used in bounding box computation (default: 3)')
    parser.add_argument('--bbox_padding', type=float, default=0.1, help='Padding factor for bounding box as fraction of size (default: 0.1)')

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
    
    # Handle single pair processing or find best image pairs
    if args.pair:
        img1_id, img2_id = args.pair
        logging.info(f"Processing single pair: {img1_id} and {img2_id}")
        
        # Validate that the image IDs exist in the reconstruction
        if not reconstruction.has_image(img1_id):
            available_ids = reconstruction.get_all_image_ids()
            logging.error(f"Image ID {img1_id} not found in reconstruction. Available IDs: {sorted(available_ids)}")
            sys.exit(1)
        if not reconstruction.has_image(img2_id):
            available_ids = reconstruction.get_all_image_ids()
            logging.error(f"Image ID {img2_id} not found in reconstruction. Available IDs: {sorted(available_ids)}")
            sys.exit(1)
        
        pair_list = [(img1_id, img2_id)]
    else:
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
    
    # Compute bounding box once if filtering is enabled (default: enabled)
    bbox_min, bbox_max = None, None
    if not args.disable_bbox_filter:
        logging.info("Computing robust bounding box for point cloud filtering...")
        bbox_min, bbox_max = reconstruction.compute_robust_bounding_box(
            min_visibility=args.bbox_min_visibility,
            padding_factor=args.bbox_padding,
            verbose=args.verbose
        )
        if bbox_min is None or bbox_max is None:
            logging.warning("Failed to compute bounding box, proceeding without bbox filtering")
            bbox_min, bbox_max = None, None
    else:
        logging.info("Bounding box filtering disabled by user")
    
    # Process each pair
    successful_pairs = 0
    for i, (img1_id, img2_id) in enumerate(pair_list):
        logging.info(f"\n=== Processing pair {i+1}/{len(pair_list)}: {img1_id}-{img2_id} ===")
        
        # Create pair-specific output directory
        if args.pair:
            # For single pair processing, use the actual frame IDs in the directory name
            pair_output_dir = output_dir / f"pair_{img1_id}_{img2_id}"
        else:
            # For batch processing, use sequential numbering
            pair_output_dir = output_dir / f"pair_{i:02d}"
        pair_output_dir.mkdir(exist_ok=True)
        
        # Process the pair
        success = process_single_pair(
            reconstruction, img1_id, img2_id, pair_output_dir, model, args,
            bbox_min=bbox_min, bbox_max=bbox_max
        )
        
        if success:
            successful_pairs += 1
    
    logging.info(f"\nCompleted processing {successful_pairs}/{len(pair_list)} pairs successfully")
    logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
