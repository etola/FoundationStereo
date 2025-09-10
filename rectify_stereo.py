#!/usr/bin/env python3
"""
Stereo Rectification Script for COLMAP Reconstructions
=====================================================

This script loads a COLMAP reconstruction and computes stereo rectification
for a pair of images, saving the rectified images and parameters.

Usage:
    python rectify_stereo.py -s scene_folder -o out_folder img_id1 img_id2

Example:
    python rectify_stereo.py -s /path/to/scene -o rectified 1 2
"""

import argparse
import json
from operator import is_
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import imageio

import cv2
import numpy as np
from colmap_utils import ColmapReconstruction


def _determine_camera_rotations(R1: np.ndarray, R2: np.ndarray, t_rel: np.ndarray, is_vertical: bool) -> Tuple[int, int]:
    """
    Determine individual rotations needed for each camera to align up vectors and make stereo horizontal.
    
    Args:
        R1: Rotation matrix of first camera (cam_from_world)
        R2: Rotation matrix of second camera (cam_from_world)
        t_rel: Relative translation vector between cameras
        is_vertical: Whether the stereo pair is vertically aligned
    Returns:
        Tuple of (rotation_angle_1, rotation_angle_2) in degrees: 0, 90, 180, or 270
    """
    # Extract up vectors (Y axis) from camera rotation matrices
    # Camera coordinate system: X=right, Y=down, Z=forward
    # So we need to check -Y (up direction in world coordinates)
    up1_cam = -R1[1, :]  # -Y axis of camera 1 in world coordinates
    up2_cam = -R2[1, :]  # -Y axis of camera 2 in world coordinates
    
    # Target up direction in world coordinates (e.g., negative Y for standard "up")
    target_up = np.array([0, -1, 0])

    def _find_rotation_to_align_up(up_vector):
        """Find the rotation angle (0, 90, 180, 270) that best aligns up_vector with target_up"""
        # Test each rotation and see which gives the best alignment
        rotations = [0, 90, 180, 270]
        best_rotation = 0
        best_dot = -2  # Worst possible dot product
        
        for angle in rotations:
            if angle == 0:
                rotated_up = up_vector
            elif angle == 90:
                # 90-degree rotation around Z: (x,y,z) -> (y,-x,z)
                rotated_up = np.array([up_vector[1], -up_vector[0], up_vector[2]])
            elif angle == 180:
                # 180-degree rotation around Z: (x,y,z) -> (-x,-y,z)
                rotated_up = np.array([-up_vector[0], -up_vector[1], up_vector[2]])
            elif angle == 270:
                # 270-degree rotation around Z: (x,y,z) -> (-y,x,z)
                rotated_up = np.array([-up_vector[1], up_vector[0], up_vector[2]])
            
            dot_product = np.dot(rotated_up, target_up)
            if dot_product > best_dot:
                best_dot = dot_product
                best_rotation = angle
        
        return best_rotation
    
    # Find optimal rotation for each camera
    rotation1 = _find_rotation_to_align_up(up1_cam)
    rotation2 = _find_rotation_to_align_up(up2_cam)
    
    # Apply rotations to see the resulting baseline
    def _rotate_vector(vec, angle):
        if angle == 0:
            return vec
        elif angle == 90:
            return np.array([vec[1], -vec[0], vec[2]])
        elif angle == 180:
            return np.array([-vec[0], -vec[1], vec[2]])
        elif angle == 270:
            return np.array([-vec[1], vec[0], vec[2]])
    
    # Check if baseline becomes horizontal with these rotations
    t_rel_rotated = _rotate_vector(t_rel, rotation1)  # Apply camera 1's rotation to baseline
    
    # If baseline is still primarily vertical, we need to adjust
    t_rel_norm = np.linalg.norm(t_rel_rotated)
    if t_rel_norm > 1e-6:
        t_rel_normalized = t_rel_rotated / t_rel_norm
        y_dominance = abs(t_rel_normalized[1])
        
        # If still vertical, apply additional 90-degree rotation to make it horizontal
        if y_dominance > 0.7:
            rotation1 = (rotation1 + 90) % 360
            rotation2 = (rotation2 + 90) % 360

    R_rotation1 = _get_rotation_matrix(rotation1)
    
    R1_rotated = R_rotation1 @ R1
    up_vector = R1_rotated[1, :]

    print(f"bef {rotation1} {rotation2}")

    if is_vertical:
        if np.dot(up_vector, np.array([-1, 0, 0])) < 0:
            rotation1 = (rotation1 + 180) % 360
            rotation2 = (rotation2 + 180) % 360
    else:
        if np.dot(up_vector, np.array([0, -1, 0])) > 0:
            rotation1 = (rotation1 + 180) % 360
            rotation2 = (rotation2 + 180) % 360

    print(f"aft {rotation1} {rotation2}")


    return rotation1, rotation2


def _get_rotation_matrix(rotation_angle: int) -> np.ndarray:
    """
    Get rotation matrix for image rotation.
    
    Args:
        rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
        
    Returns:
        3x3 rotation matrix
    """
    if rotation_angle == 0:
        return np.eye(3)
    elif rotation_angle == 90:
        # 90-degree clockwise rotation: (x,y) -> (h-y, x)
        return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    elif rotation_angle == 180:
        # 180-degree rotation: (x,y) -> (w-x, h-y)
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif rotation_angle == 270:
        # 270-degree clockwise rotation: (x,y) -> (y, w-x)
        return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation_angle}")


def _update_intrinsics_for_rotation(K: np.ndarray, image_size: Tuple[int, int], 
                                   rotation_angle: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Update intrinsic matrix for image rotation.
    
    Args:
        K: Original intrinsic matrix
        image_size: Original image size (width, height)
        rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
        
    Returns:
        Tuple of (updated_K, updated_image_size)
    """
    w, h = image_size
    
    if rotation_angle == 0:
        return K.copy(), image_size
    elif rotation_angle == 90:
        # 90-degree clockwise: swap fx↔fy, cx↔cy, adjust for new dimensions
        K_rot = np.array([
            [K[1,1], 0, K[1,2]],  # fy, 0, cy
            [0, K[0,0], h - K[0,2]],  # 0, fx, height-cx
            [0, 0, 1]
        ])
        return K_rot, (h, w)  # Swap width and height
    elif rotation_angle == 180:
        # 180-degree rotation: keep fx, fy, adjust cx, cy
        K_rot = np.array([
            [K[0,0], 0, w - K[0,2]],  # fx, 0, width-cx
            [0, K[1,1], h - K[1,2]],  # 0, fy, height-cy
            [0, 0, 1]
        ])
        return K_rot, image_size  # Keep same dimensions
    elif rotation_angle == 270:
        # 270-degree clockwise: swap fx↔fy, adjust for new dimensions
        K_rot = np.array([
            [K[1,1], 0, w - K[1,2]],  # fy, 0, width-cy
            [0, K[0,0], K[0,2]],  # 0, fx, cx
            [0, 0, 1]
        ])
        return K_rot, (h, w)  # Swap width and height
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation_angle}")


def _apply_rotation_to_coordinates(coords: Tuple[float, float], 
                                  rotation_angle: int, 
                                  image_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Apply rotation to image coordinates.
    
    Args:
        coords: (x, y) coordinates
        rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
        image_size: Original image size (width, height)
        
    Returns:
        Rotated coordinates
    """
    x, y = coords
    w, h = image_size
    
    if rotation_angle == 0:
        return x, y
    elif rotation_angle == 90:
        # 90-degree clockwise: (x, y) -> (h-y, x)
        return h - y, x
    elif rotation_angle == 180:
        # 180-degree: (x, y) -> (w-x, h-y)
        return w - x, h - y
    elif rotation_angle == 270:
        # 270-degree clockwise: (x, y) -> (y, w-x)
        return y, w - x
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation_angle}")


def _apply_inverse_rotation_to_coordinates(coords: Tuple[float, float], 
                                          rotation_angle: int, 
                                          image_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Apply inverse rotation to image coordinates.
    
    Args:
        coords: (x, y) coordinates
        rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
        image_size: Original image size (width, height)
        
    Returns:
        Inverse rotated coordinates
    """
    x, y = coords
    w, h = image_size
    
    if rotation_angle == 0:
        return x, y
    elif rotation_angle == 90:
        # Inverse of 90-degree clockwise: (x, y) -> (y, h-x)
        return y, h - x
    elif rotation_angle == 180:
        # Inverse of 180-degree: (x, y) -> (w-x, h-y)
        return w - x, h - y
    elif rotation_angle == 270:
        # Inverse of 270-degree clockwise: (x, y) -> (w-y, x)
        return w - y, x
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation_angle}")

def _is_vertical_alignment(C1: np.ndarray, C2: np.ndarray) -> bool:

    # Calculate the vector between camera centers
    center_diff = C2 - C1

    # Determine if alignment is mostly vertical or horizontal
    abs_dx = abs(center_diff[0])  # Horizontal separation
    abs_dy = abs(center_diff[1])  # Vertical separation

    return abs_dy > abs_dx


def _determine_image_order_by_camera_centers(C1: np.ndarray, C2: np.ndarray, img1_id: int, img2_id: int) -> Tuple[int, int]:
    """
    Determine the order of images based on camera center positions.
    
    Args:
        C1: Camera center of first image in world coordinates
        C2: Camera center of second image in world coordinates  
        img1_id: First image ID
        img2_id: Second image ID
        
    Returns:
        Tuple of (first_image_id, second_image_id) in proper order
    """

    if _is_vertical_alignment(C1, C2):
        # Mostly vertical alignment - order by Y coordinate (smaller Y first)
        print(f"  Debug: Vertical alignment detected")
        if C1[1] < C2[1]:
            return img1_id, img2_id
        else:
            return img2_id, img1_id
    else:
        # Mostly horizontal alignment - order by X coordinate (smaller X first)
        print(f"  Debug: Horizontal alignment detected")
        if C1[0] < C2[0]:
            return img2_id, img1_id
        else:
            return img1_id, img2_id



def compute_stereo_rectification(reconstruction: ColmapReconstruction, 
                                img1_id: int, img2_id: int, images_path: Path, output_dir: Path, alpha: float = 1.0) -> Dict[str, Any]:
    """
    Compute stereo rectification parameters for two images.
    If images are vertically aligned, applies 90-degree rotation first, then horizontal rectification.
    
    Args:
        reconstruction: ColmapReconstruction object
        img1_id: First image ID
        img2_id: Second image ID
        images_path: Path to images directory
        output_dir: Output directory
        alpha: Free scaling parameter (0.0=more crop, 1.0=less crop, default: 1.0)
        
    Returns:
        Dictionary containing rectification parameters
    """
    # Get camera parameters
    K1 = reconstruction.get_camera_calibration_matrix(img1_id)
    K2 = reconstruction.get_camera_calibration_matrix(img2_id)
    
    # Get camera poses
    R1 = reconstruction.get_image_cam_from_world(img1_id).rotation.matrix()
    t1 = reconstruction.get_image_cam_from_world(img1_id).translation
    R2 = reconstruction.get_image_cam_from_world(img2_id).rotation.matrix()
    t2 = reconstruction.get_image_cam_from_world(img2_id).translation
    
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    # Get distortion parameters
    _, dist1 = reconstruction.get_camera_distortion_params(img1_id)
    _, dist2 = reconstruction.get_camera_distortion_params(img2_id)
    
    # Get image dimensions
    camera1 = reconstruction.get_image_camera(img1_id)
    camera2 = reconstruction.get_image_camera(img2_id)
    image_size = (camera1.width, camera1.height)
    
    # Compute relative pose
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    
    # Determine required rotations for each camera based on their orientations and baseline
    rotation_angle1, rotation_angle2 = _determine_camera_rotations(R1, R2, t_rel, _is_vertical_alignment(C1, C2))
    
    # Apply rotations if needed
    R_rotation1 = _get_rotation_matrix(rotation_angle1)
    R_rotation2 = _get_rotation_matrix(rotation_angle2)
    K1_rotated, image_size_rotated1 = _update_intrinsics_for_rotation(K1, image_size, rotation_angle1)
    K2_rotated, image_size_rotated2 = _update_intrinsics_for_rotation(K2, image_size, rotation_angle2)
    
    R1_rotated = R_rotation1 @ R1
    R2_rotated = R_rotation2 @ R2

    # For simplicity, use the first camera's rotated image size
    # (both should be the same if rotations are 0/180 or both are 90/270)
    image_size_rotated = image_size_rotated1
    
    # Transform the relative pose based on individual rotations
    # The relative pose needs to account for both cameras being rotated individually
    if rotation_angle1 == 0 and rotation_angle2 == 0:
        R_rel_rotated = R_rel
        t_rel_rotated = t_rel
    else:
        # When cameras are rotated individually, we need to transform the relative pose
        # R_rel_rotated = R_rotation2 @ R_rel @ R_rotation1.T
        # t_rel_rotated = R_rotation2 @ t_rel (because translation is from camera 1 to camera 2)
        R_rel_rotated = R_rotation2 @ R_rel @ R_rotation1.T
        t_rel_rotated = R_rotation2 @ t_rel
    
    # Check if we need any rotation (for backward compatibility)
    is_vertical = rotation_angle1 != 0 or rotation_angle2 != 0
    
    # Add debug output for stereo rectification
    print(f"  Debug: Camera rotations: Image1={rotation_angle1}°, Image2={rotation_angle2}°")
    print(f"  Debug: Stereo rectification input:")
    print(f"    Image size (rotated): {image_size_rotated}")
    print(f"    K1_rotated fx: {K1_rotated[0,0]:.2f}, fy: {K1_rotated[1,1]:.2f}, cx: {K1_rotated[0,2]:.2f}, cy: {K1_rotated[1,2]:.2f}")
    print(f"    K2_rotated fx: {K2_rotated[0,0]:.2f}, fy: {K2_rotated[1,1]:.2f}, cx: {K2_rotated[0,2]:.2f}, cy: {K2_rotated[1,2]:.2f}")
    print(f"    R_rel_rotated determinant: {np.linalg.det(R_rel_rotated):.6f}")
    print(f"    t_rel        : [{t_rel[0]:.6f}, {t_rel[1]:.6f}, {t_rel[2]:.6f}]")
    print(f"    t_rel_rotated: [{t_rel_rotated[0]:.6f}, {t_rel_rotated[1]:.6f}, {t_rel_rotated[2]:.6f}]")
    print(f"    t_rel_rotated norm: {np.linalg.norm(t_rel_rotated):.6f}")
    

    y_world = -R_rel[1, :]
    y_world_rotated = -R_rel_rotated[1, :]
    print(f"    y_world        : [{y_world[0]:.6f}, {y_world[1]:.6f}, {y_world[2]:.6f}]")
    print(f"    y_world_rotated: [{y_world_rotated[0]:.6f}, {y_world_rotated[1]:.6f}, {y_world_rotated[2]:.6f}]")


    # Stereo rectification using OpenCV functions (now always horizontal)
    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1_rotated, dist1, K2_rotated, dist2, image_size_rotated, R_rel_rotated, t_rel_rotated,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
    )
    
    # Check for degenerate stereo rectification results when alpha=1.0 --> opencv bug happens sometimes for alpha=1.0
    original_alpha = alpha
    roi1_valid = roi1[2] > 0 and roi1[3] > 0  # width > 0 and height > 0
    roi2_valid = roi2[2] > 0 and roi2[3] > 0
    focal_lengths_valid = P1[0,0] > 0 and P1[1,1] > 0 and P2[0,0] > 0 and P2[1,1] > 0
    
    if alpha == 1.0 and (not roi1_valid or not roi2_valid or not focal_lengths_valid):
        print(f"  Debug: Alpha=1.0 produced degenerate results - falling back to alpha=0.0")
        print(f"    ROI validity: ROI1={roi1_valid}, ROI2={roi2_valid}")
        print(f"    Focal length validity: {focal_lengths_valid}")
        print(f"    Original ROI1: {roi1}, ROI2: {roi2}")
        print(f"    Original P1 fx: {P1[0,0]:.2f}, P2 fx: {P2[0,0]:.2f}")
        
        # Fallback to alpha=0.0
        alpha = 0.0
        R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1_rotated, dist1, K2_rotated, dist2, image_size_rotated, R_rel_rotated, t_rel_rotated,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
        )
        print(f"    Fallback ROI1: {roi1}, ROI2: {roi2}")
        print(f"    Fallback P1 fx: {P1[0,0]:.2f}, P2 fx: {P2[0,0]:.2f}")
    
    print(f"  Debug: Stereo rectification output:")
    print(f"    Alpha parameter: {original_alpha} (used: {alpha})")
    print(f"    P1 fx: {P1[0,0]:.2f}, fy: {P1[1,1]:.2f}, cx: {P1[0,2]:.2f}, cy: {P1[1,2]:.2f}")
    print(f"    P2 fx: {P2[0,0]:.2f}, fy: {P2[1,1]:.2f}, cx: {P2[0,2]:.2f}, cy: {P2[1,2]:.2f}")
    print(f"    ROI1: {roi1}, ROI2: {roi2}")

    img1_name = reconstruction.get_image_name(img1_id)
    img2_name = reconstruction.get_image_name(img2_id)

    rect_info = {
        'img1_id': img1_id,
        'img2_id': img2_id,
        'img1_name': img1_name,
        'img2_name': img2_name,
        'img1_path': str(images_path / img1_name),
        'img2_path': str(images_path / img2_name),
        'rect1_path': os.path.join(output_dir, f"{Path(img1_name).stem}_rectified.jpg"),
        'rect2_path': os.path.join(output_dir, f"{Path(img2_name).stem}_rectified.jpg"),
        'K1': K1.tolist(),
        'K2': K2.tolist(),
        'K1_rotated': K1_rotated.tolist(),
        'K2_rotated': K2_rotated.tolist(),
        'dist1': dist1.tolist(),
        'dist2': dist2.tolist(),
        'R1': R1.tolist(),
        'R2': R2.tolist(),
        't1': t1.tolist(),
        't2': t2.tolist(),
        'C1': C1.tolist(),
        'C2': C2.tolist(),
        'R1_rotated': R1_rotated.tolist(),
        'R2_rotated': R2_rotated.tolist(),
        'R_rel': R_rel.tolist(),
        't_rel': t_rel.tolist(),
        'R_rel_rotated': R_rel_rotated.tolist(),
        't_rel_rotated': t_rel_rotated.tolist(),
        'R_rotation1': R_rotation1.tolist(),
        'R_rotation2': R_rotation2.tolist(),
        'rotation_angle1': rotation_angle1,
        'rotation_angle2': rotation_angle2,
        'is_vertical': is_vertical,
        'R1_rect': R1_rect.tolist(),
        'R2_rect': R2_rect.tolist(),
        'P1': P1.tolist(),
        'P2': P2.tolist(),
        'Q': Q.tolist(),
        'image_size': image_size,
        'image_size_rotated': image_size_rotated,
        'alpha': alpha,
        'original_alpha': original_alpha,  # Store the originally requested alpha
        'roi1': roi1,
        'roi2': roi2
    }
    
    # Since we always do horizontal rectification now, set the type accordingly
    rect_info['type'] = 'horizontal'
    rect_info['left'] = img1_id
    rect_info['right'] = img2_id
    
    return rect_info



def apply_custom_roi_cropping_with_alignment(img1_rect: np.ndarray, img2_rect: np.ndarray, 
                                           rect_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply custom ROI cropping with alignment padding to maintain rectification property.
    
    Args:
        img1_rect: First rectified image
        img2_rect: Second rectified image
        rect_params: Rectification parameters
        
    Returns:
        Tuple of (img1_cropped, img2_cropped, mask1, mask2, updated_rect_params)
    """
    # Get original ROI values
    roi1 = rect_params.get('roi1', (0, 0, img1_rect.shape[1], img1_rect.shape[0]))
    roi2 = rect_params.get('roi2', (0, 0, img2_rect.shape[1], img2_rect.shape[0]))
    
    # Extract ROI values (x, y, width, height)
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    
    # Crop each image to its valid ROI region only
    img1_valid = img1_rect[y1:y1+h1, x1:x1+w1]
    img2_valid = img2_rect[y2:y2+h2, x2:x2+w2]
    
    # Create masks for valid regions (before padding)
    mask1_valid = np.ones((h1, w1), dtype=np.uint8) * 255
    mask2_valid = np.ones((h2, w2), dtype=np.uint8) * 255
    
    # Determine alignment padding needed
    min_y_offset = min(y1, y2)
    pad_top_1 = y1 - min_y_offset
    pad_top_2 = y2 - min_y_offset
    
    # Apply top padding for vertical alignment
    img1_cropped = img1_valid
    img2_cropped = img2_valid
    mask1 = mask1_valid
    mask2 = mask2_valid
    
    if pad_top_1 > 0:
        if len(img1_cropped.shape) == 3:  # Color image
            top_padding = np.zeros((pad_top_1, img1_cropped.shape[1], img1_cropped.shape[2]), dtype=img1_cropped.dtype)
        else:  # Grayscale image
            top_padding = np.zeros((pad_top_1, img1_cropped.shape[1]), dtype=img1_cropped.dtype)
        img1_cropped = np.vstack([top_padding, img1_cropped])
        
        # Add padding to mask (0 = invalid/padded region)
        mask_padding = np.zeros((pad_top_1, mask1.shape[1]), dtype=np.uint8)
        mask1 = np.vstack([mask_padding, mask1])
    
    if pad_top_2 > 0:
        if len(img2_cropped.shape) == 3:  # Color image
            top_padding = np.zeros((pad_top_2, img2_cropped.shape[1], img2_cropped.shape[2]), dtype=img2_cropped.dtype)
        else:  # Grayscale image
            top_padding = np.zeros((pad_top_2, img2_cropped.shape[1]), dtype=img2_cropped.dtype)
        img2_cropped = np.vstack([top_padding, img2_cropped])
        
        # Add padding to mask
        mask_padding = np.zeros((pad_top_2, mask2.shape[1]), dtype=np.uint8)
        mask2 = np.vstack([mask_padding, mask2])
    
    # Make both images the same size by padding to maximum dimensions
    max_width = max(img1_cropped.shape[1], img2_cropped.shape[1])
    max_height = max(img1_cropped.shape[0], img2_cropped.shape[0])
    
    # Calculate padding needed for each image
    pad_right_1 = max_width - img1_cropped.shape[1]
    pad_bottom_1 = max_height - img1_cropped.shape[0]
    pad_right_2 = max_width - img2_cropped.shape[1]
    pad_bottom_2 = max_height - img2_cropped.shape[0]
    
    # Apply right and bottom padding to image 1
    if pad_right_1 > 0 or pad_bottom_1 > 0:
        if len(img1_cropped.shape) == 3:  # Color image
            img1_cropped = np.pad(img1_cropped, 
                                ((0, pad_bottom_1), (0, pad_right_1), (0, 0)), 
                                mode='constant', constant_values=0)
        else:  # Grayscale image
            img1_cropped = np.pad(img1_cropped, 
                                ((0, pad_bottom_1), (0, pad_right_1)), 
                                mode='constant', constant_values=0)
        
        # Apply padding to mask
        mask1 = np.pad(mask1, ((0, pad_bottom_1), (0, pad_right_1)), 
                      mode='constant', constant_values=0)
    
    # Apply right and bottom padding to image 2
    if pad_right_2 > 0 or pad_bottom_2 > 0:
        if len(img2_cropped.shape) == 3:  # Color image
            img2_cropped = np.pad(img2_cropped, 
                                ((0, pad_bottom_2), (0, pad_right_2), (0, 0)), 
                                mode='constant', constant_values=0)
        else:  # Grayscale image
            img2_cropped = np.pad(img2_cropped, 
                                ((0, pad_bottom_2), (0, pad_right_2)), 
                                mode='constant', constant_values=0)
        
        # Apply padding to mask
        mask2 = np.pad(mask2, ((0, pad_bottom_2), (0, pad_right_2)), 
                      mode='constant', constant_values=0)
    
    # Update rect_params with new cropping information for coordinate transforms
    updated_rect_params = rect_params.copy()
    
    # The new effective ROI starts from min_y_offset and has the final dimensions
    new_roi1 = (x1, min_y_offset, img1_cropped.shape[1], img1_cropped.shape[0])
    new_roi2 = (x2, min_y_offset, img2_cropped.shape[1], img2_cropped.shape[0])
    
    # Store original ROIs and new custom ROIs
    updated_rect_params['roi1_original'] = roi1
    updated_rect_params['roi2_original'] = roi2
    updated_rect_params['roi1_custom'] = new_roi1
    updated_rect_params['roi2_custom'] = new_roi2
    
    # Store padding information for coordinate transformations
    updated_rect_params['custom_padding'] = {
        'pad_top_1': pad_top_1,
        'pad_top_2': pad_top_2,
        'pad_right_1': pad_right_1,
        'pad_bottom_1': pad_bottom_1,
        'pad_right_2': pad_right_2,
        'pad_bottom_2': pad_bottom_2,
        'min_y_offset': min_y_offset,
        'final_size': (img1_cropped.shape[0], img1_cropped.shape[1])  # (height, width)
    }
    
    # Update the roi1 and roi2 in rect_params to reflect custom cropping
    updated_rect_params['roi1'] = new_roi1
    updated_rect_params['roi2'] = new_roi2
    
    return img1_cropped, img2_cropped, mask1, mask2, updated_rect_params


def rectify_images(rect_params: Dict[str, Any], debug_output_dir: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify two images using the computed rectification parameters.
    Handles rotation for vertically aligned images.
    
    Args:
        rect_params: Rectification parameters
        debug_output_dir: Optional directory to save intermediate debug images
        
    Returns:
        Tuple of (rectified_img1, rectified_img2)
    """
    # Load images
    img1 = imageio.imread(rect_params['img1_path'])
    img2 = imageio.imread(rect_params['img2_path'])

    if img1 is None:
        raise ValueError(f"Could not load image: {rect_params['img1_path']}")
    if img2 is None:
        raise ValueError(f"Could not load image: {rect_params['img2_path']}")
    
    # Save original images for debugging
    if debug_output_dir:
        debug_output_dir.mkdir(exist_ok=True)
        imageio.imwrite(str(debug_output_dir / 'img1_0_original.jpg'), img1)
        imageio.imwrite(str(debug_output_dir / 'img2_0_original.jpg'), img2)
        print(f"  Debug: Saved original images to {debug_output_dir}")
    
    # Check if rotation is needed for each image individually
    rotation_angle1 = rect_params.get('rotation_angle1', 0)
    rotation_angle2 = rect_params.get('rotation_angle2', 0)
    print(f"  Debug: Applying rotations - Image1: {rotation_angle1}°, Image2: {rotation_angle2}°")
    
    def apply_rotation(img, angle):
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return img
    
    # Apply individual rotations
    img1 = apply_rotation(img1, rotation_angle1)
    img2 = apply_rotation(img2, rotation_angle2)
    
    # Save rotated images for debugging
    if debug_output_dir and (rotation_angle1 != 0 or rotation_angle2 != 0):
        imageio.imwrite(str(debug_output_dir / f'img1_1_rotated.jpg'), img1)
        imageio.imwrite(str(debug_output_dir / f'img2_1_rotated.jpg'), img2)
        print(f"  Debug: Saved rotated images to {debug_output_dir}")
    
    # Reconstruct rectification parameters
    K1_rotated = np.array(rect_params['K1_rotated'])
    K2_rotated = np.array(rect_params['K2_rotated'])
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    image_size_rotated = tuple(rect_params['image_size_rotated'])
    
    # Compute rectification maps
    map1_x, map1_y = cv2.initUndistortRectifyMap(K1_rotated, dist1, R1_rect, P1, image_size_rotated, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K2_rotated, dist2, R2_rect, P2, image_size_rotated, cv2.CV_32FC1)
    
    # Apply rectification
    img1_rect = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)
    
    # Apply custom cropping when original alpha=1 to maintain rectification and identical sizes
    # Even if we fell back to alpha=0 due to degenerate results, we still want custom cropping
    original_alpha = rect_params.get('original_alpha', rect_params.get('alpha', 0.0))
    if original_alpha == 1.0:
        img1_cropped, img2_cropped, mask1, mask2, updated_rect_params = apply_custom_roi_cropping_with_alignment(
            img1_rect, img2_rect, rect_params
        )
        
        # Update rect_params with new cropping information
        rect_params.update(updated_rect_params)
        
        # Use the cropped images as the final output
        img1_rect = img1_cropped
        img2_rect = img2_cropped
    
    # Save rectified images for debugging
    if debug_output_dir:
        if original_alpha == 1.0:
            # Save original rectified images (before custom cropping)
            img1_rect_full = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
            img2_rect_full = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)
            imageio.imwrite(str(debug_output_dir / 'img1_2_rectified_full.jpg'), img1_rect_full)
            imageio.imwrite(str(debug_output_dir / 'img2_2_rectified_full.jpg'), img2_rect_full)
            
            # Save custom cropped and aligned images
            imageio.imwrite(str(debug_output_dir / 'img1_3_cropped.jpg'), img1_rect)
            imageio.imwrite(str(debug_output_dir / 'img2_3_cropped.jpg'), img2_rect)
            
            # Save masks
            imageio.imwrite(str(debug_output_dir / 'img1_4_mask.jpg'), mask1)
            imageio.imwrite(str(debug_output_dir / 'img2_4_mask.jpg'), mask2)
            
            # Print debug information
            padding_info = rect_params['custom_padding']
            used_alpha = rect_params.get('alpha', 0.0)
            print(f"  Debug: Saved rectified images with custom cropping to {debug_output_dir}")
            if used_alpha != original_alpha:
                print(f"  Debug: Applied fallback from alpha={original_alpha} to alpha={used_alpha} due to degenerate results")
            print(f"  Debug: Original ROI1: {rect_params['roi1_original']}")
            print(f"  Debug: Original ROI2: {rect_params['roi2_original']}")
            print(f"  Debug: Custom ROI1: {rect_params['roi1_custom']}")
            print(f"  Debug: Custom ROI2: {rect_params['roi2_custom']}")
            print(f"  Debug: Padding applied - top: ({padding_info['pad_top_1']}, {padding_info['pad_top_2']}), "
                  f"right: ({padding_info['pad_right_1']}, {padding_info['pad_right_2']}), "
                  f"bottom: ({padding_info['pad_bottom_1']}, {padding_info['pad_bottom_2']})")
            print(f"  Debug: Final image sizes: {padding_info['final_size']} (both images now identical)")
            print(f"  Debug: Rectification maintained - both images start from y={padding_info['min_y_offset']}")
        else:
            # Standard debugging for other alpha values
            imageio.imwrite(str(debug_output_dir / 'img1_2_rectified.jpg'), img1_rect)
            imageio.imwrite(str(debug_output_dir / 'img2_2_rectified.jpg'), img2_rect)
            print(f"  Debug: Saved final rectified images to {debug_output_dir}")
        
        # Save rectification info as text for debugging
        debug_info_path = debug_output_dir / 'rectification_debug.txt'
        with open(debug_info_path, 'w') as f:
            f.write(f"Rotation angle 1: {rotation_angle1}°\n")
            f.write(f"Rotation angle 2: {rotation_angle2}°\n")
            f.write(f"Alpha parameter: {rect_params.get('alpha', 0.0)}\n")
            f.write(f"Original image size: {rect_params['image_size']}\n")
            f.write(f"Rotated image size: {image_size_rotated}\n")
            f.write(f"Left image ID: {rect_params['left']}\n")
            f.write(f"Right image ID: {rect_params['right']}\n")
            f.write(f"Rectification type: {rect_params.get('type', 'unknown')}\n")
            f.write(f"\nOriginal K1:\n{np.array(rect_params['K1'])}\n")
            f.write(f"\nRotated K1:\n{K1_rotated}\n")
            f.write(f"\nOriginal K2:\n{np.array(rect_params['K2'])}\n")
            f.write(f"\nRotated K2:\n{K2_rotated}\n")
            f.write(f"\nP1 (rectified projection matrix 1):\n{P1}\n")
            f.write(f"\nP2 (rectified projection matrix 2):\n{P2}\n")
            f.write(f"\nROI1: {rect_params.get('roi1', 'unknown')}\n")
            f.write(f"ROI2: {rect_params.get('roi2', 'unknown')}\n")
        print(f"  Debug: Saved rectification info to {debug_info_path}")
    
    return img1_rect, img2_rect




def transform_coordinates_to_rectified(rect_params: Dict[str, Any], 
                                     coords_img1: Tuple[float, float], 
                                     coords_img2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Transform coordinates from original images to rectified images.
    Pipeline: Original → Rotation (if vertical) → Rectification → Cropping (if alpha=0)
    
    Args:
        rect_params: Rectification parameters
        coords_img1: (x, y) coordinates in first original image
        coords_img2: (x, y) coordinates in second original image
        
    Returns:
        Tuple of ((x1_rect, y1_rect), (x2_rect, y2_rect)) in rectified images
    """
    # Reconstruct rectification parameters
    K1 = np.array(rect_params['K1'])
    K2 = np.array(rect_params['K2'])
    K1_rotated = np.array(rect_params['K1_rotated'])
    K2_rotated = np.array(rect_params['K2_rotated'])
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    rotation_angle1 = rect_params.get('rotation_angle1', 0)
    rotation_angle2 = rect_params.get('rotation_angle2', 0)
    image_size = tuple(rect_params['image_size'])
    
    x1, y1 = coords_img1
    x2, y2 = coords_img2
    
    # Step 1: Apply individual rotations if needed
    if rotation_angle1 != 0:
        # Apply rotation to coordinates for image 1
        x1_rot, y1_rot = _apply_rotation_to_coordinates((x1, y1), rotation_angle1, image_size)
        K1_use = K1_rotated
    else:
        x1_rot, y1_rot = x1, y1
        K1_use = K1
    
    if rotation_angle2 != 0:
        # Apply rotation to coordinates for image 2
        x2_rot, y2_rot = _apply_rotation_to_coordinates((x2, y2), rotation_angle2, image_size)
        K2_use = K2_rotated
    else:
        x2_rot, y2_rot = x2, y2
        K2_use = K2
    
    # Step 2: Apply rectification using OpenCV's undistortPoints
    point1_rect = cv2.undistortPoints(
        np.array([[[x1_rot, y1_rot]]], dtype=np.float32), 
        K1_use, None, R=R1_rect, P=P1
    )[0, 0]
    
    point2_rect = cv2.undistortPoints(
        np.array([[[x2_rot, y2_rot]]], dtype=np.float32), 
        K2_use, None, R=R2_rect, P=P2
    )[0, 0]
    
    # Step 3: Apply cropping (alpha=0 means cropping is already applied in P1, P2)
    # For custom padding (alpha=1), adjust coordinates to account for padding
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        # Adjust for custom cropping and padding
        # First, convert to custom ROI coordinates
        x1_custom = point1_rect[0] - rect_params['roi1_custom'][0]
        y1_custom = point1_rect[1] - rect_params['roi1_custom'][1] + padding_info['pad_top_1']
        
        x2_custom = point2_rect[0] - rect_params['roi2_custom'][0]
        y2_custom = point2_rect[1] - rect_params['roi2_custom'][1] + padding_info['pad_top_2']
        
        return (x1_custom, y1_custom), (x2_custom, y2_custom)
    
    # The coordinates returned are already in the cropped rectified image space
    return (point1_rect[0], point1_rect[1]), (point2_rect[0], point2_rect[1])


def transform_coordinates_from_rectified(rect_params: Dict[str, Any], 
                                       coords_rect1: Tuple[float, float], 
                                       coords_rect2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Transform coordinates from rectified images back to original images.
    Pipeline: Cropping → Rectification → Rotation (if vertical) → Original
    
    Args:
        rect_params: Rectification parameters
        coords_rect1: (x, y) coordinates in first rectified image
        coords_rect2: (x, y) coordinates in second rectified image
        
    Returns:
        Tuple of ((x1_orig, y1_orig), (x2_orig, y2_orig)) in original images
    """
    # Reconstruct rectification parameters
    K1 = np.array(rect_params['K1'])
    K2 = np.array(rect_params['K2'])
    K1_rotated = np.array(rect_params['K1_rotated'])
    K2_rotated = np.array(rect_params['K2_rotated'])
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    roi1 = rect_params.get('roi1', (0, 0, 0, 0))
    roi2 = rect_params.get('roi2', (0, 0, 0, 0))
    rotation_angle1 = rect_params.get('rotation_angle1', 0)
    rotation_angle2 = rect_params.get('rotation_angle2', 0)
    image_size = tuple(rect_params['image_size'])
    
    x1_rect, y1_rect = coords_rect1
    x2_rect, y2_rect = coords_rect2
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
    # Handle custom padding for alpha=1
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        # Convert from custom padded coordinates back to original rectified coordinates
        x1_uncropped = x1_rect + rect_params['roi1_custom'][0]
        y1_uncropped = y1_rect + rect_params['roi1_custom'][1] - padding_info['pad_top_1']
        
        x2_uncropped = x2_rect + rect_params['roi2_custom'][0]
        y2_uncropped = y2_rect + rect_params['roi2_custom'][1] - padding_info['pad_top_2']
    else:
        # When alpha=0, OpenCV crops the rectified images using ROI
        # ROI format is (x, y, width, height), so we add the offset
        x1_uncropped = x1_rect + roi1[0]  # Add ROI x offset
        y1_uncropped = y1_rect + roi1[1]  # Add ROI y offset
        x2_uncropped = x2_rect + roi2[0]  # Add ROI x offset
        y2_uncropped = y2_rect + roi2[1]  # Add ROI y offset
    
    # Step 2: Convert uncropped rectified coordinates to normalized coordinates
    # Extract the rectified camera matrix from P1 and P2
    K_rect1 = P1[:, :3]  # Intrinsic matrix for rectified camera 1
    K_rect2 = P2[:, :3]  # Intrinsic matrix for rectified camera 2
    
    # Convert to normalized coordinates
    point1_rect_normalized = np.linalg.inv(K_rect1) @ np.array([x1_uncropped, y1_uncropped, 1])
    point2_rect_normalized = np.linalg.inv(K_rect2) @ np.array([x2_uncropped, y2_uncropped, 1])
    
    # Step 3: Apply inverse rectification rotation
    point1_original_normalized = R1_rect.T @ point1_rect_normalized
    point2_original_normalized = R2_rect.T @ point2_rect_normalized
    
    # Step 4: Project back to rotated image coordinates using individual rotations
    if rotation_angle1 != 0:
        K1_use = K1_rotated
    else:
        K1_use = K1
    
    if rotation_angle2 != 0:
        K2_use = K2_rotated
    else:
        K2_use = K2
    
    point1_rotated_homogeneous = K1_use @ point1_original_normalized
    x1_rotated = point1_rotated_homogeneous[0] / point1_rotated_homogeneous[2]
    y1_rotated = point1_rotated_homogeneous[1] / point1_rotated_homogeneous[2]
    
    point2_rotated_homogeneous = K2_use @ point2_original_normalized
    x2_rotated = point2_rotated_homogeneous[0] / point2_rotated_homogeneous[2]
    y2_rotated = point2_rotated_homogeneous[1] / point2_rotated_homogeneous[2]
    
    # Step 5: Apply individual inverse rotations if needed
    if rotation_angle1 != 0:
        # Apply inverse rotation to coordinates for image 1
        x1_orig, y1_orig = _apply_inverse_rotation_to_coordinates((x1_rotated, y1_rotated), rotation_angle1, image_size)
    else:
        x1_orig, y1_orig = x1_rotated, y1_rotated
    
    if rotation_angle2 != 0:
        # Apply inverse rotation to coordinates for image 2
        x2_orig, y2_orig = _apply_inverse_rotation_to_coordinates((x2_rotated, y2_rotated), rotation_angle2, image_size)
    else:
        x2_orig, y2_orig = x2_rotated, y2_rotated
    
    return (x1_orig, y1_orig), (x2_orig, y2_orig)


def transform_single_image_coordinates_to_rectified(rect_params: Dict[str, Any], 
                                                  coords: Tuple[float, float], 
                                                  image_id: int) -> Tuple[float, float]:
    """
    Transform coordinates from a specific original image to its rectified version.
    Pipeline: Original → Rotation (if vertical) → Rectification → Cropping (if alpha=0)
    
    Args:
        rect_params: Rectification parameters
        coords: (x, y) coordinates in the original image
        image_id: Image ID (1 for first image, 2 for second image)
        
    Returns:
        (x_rect, y_rect) coordinates in the rectified image
    """
    # Reconstruct rectification parameters
    if image_id == 1:
        K = np.array(rect_params['K1'])
        K_rotated = np.array(rect_params['K1_rotated'])
        dist = np.array(rect_params['dist1'])
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
        rotation_angle = rect_params.get('rotation_angle1', 0)
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        K_rotated = np.array(rect_params['K2_rotated'])
        dist = np.array(rect_params['dist2'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
        rotation_angle = rect_params.get('rotation_angle2', 0)
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    x, y = coords
    image_size = tuple(rect_params['image_size'])
    
    # Step 1: Apply rotation if needed
    if rotation_angle != 0:
        # Apply rotation to coordinates
        x_rot, y_rot = _apply_rotation_to_coordinates((x, y), rotation_angle, image_size)
        K_use = K_rotated
    else:
        x_rot, y_rot = x, y
        K_use = K
    
    # Step 2: Apply rectification using OpenCV's undistortPoints
    point_rect = cv2.undistortPoints(
        np.array([[[x_rot, y_rot]]], dtype=np.float32), 
        K_use, None, R=R_rect, P=P
    )[0, 0]
    
    # Step 3: Apply custom padding adjustments if available
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        if image_id == 1:
            roi_custom = rect_params['roi1_custom']
            pad_top = padding_info['pad_top_1']
        else:
            roi_custom = rect_params['roi2_custom']
            pad_top = padding_info['pad_top_2']
        
        # Adjust for custom cropping and padding
        x_custom = point_rect[0] - roi_custom[0]
        y_custom = point_rect[1] - roi_custom[1] + pad_top
        
        return (x_custom, y_custom)
    
    return (point_rect[0], point_rect[1])


def transform_single_image_coordinates_from_rectified(rect_params: Dict[str, Any], 
                                                    coords_rect: Tuple[float, float], 
                                                    image_id: int) -> Tuple[float, float]:
    """
    Transform coordinates from a specific rectified image back to its original version.
    Pipeline: Cropping → Rectification → Rotation (if vertical) → Original
    
    Args:
        rect_params: Rectification parameters
        coords_rect: (x, y) coordinates in the rectified image
        image_id: Image ID (1 for first image, 2 for second image)
        
    Returns:
        (x_orig, y_orig) coordinates in the original image
    """
    # Reconstruct rectification parameters
    if image_id == 1:
        K = np.array(rect_params['K1'])
        K_rotated = np.array(rect_params['K1_rotated'])
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
        roi = rect_params.get('roi1', (0, 0, 0, 0))
        rotation_angle = rect_params.get('rotation_angle1', 0)
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        K_rotated = np.array(rect_params['K2_rotated'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
        roi = rect_params.get('roi2', (0, 0, 0, 0))
        rotation_angle = rect_params.get('rotation_angle2', 0)
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    x_rect, y_rect = coords_rect
    image_size = tuple(rect_params['image_size'])
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
    # Handle custom padding for alpha=1
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        if image_id == 1:
            roi_custom = rect_params['roi1_custom']
            pad_top = padding_info['pad_top_1']
        else:
            roi_custom = rect_params['roi2_custom']
            pad_top = padding_info['pad_top_2']
        
        # Convert from custom padded coordinates back to original rectified coordinates
        x_uncropped = x_rect + roi_custom[0]
        y_uncropped = y_rect + roi_custom[1] - pad_top
    else:
        x_uncropped = x_rect + roi[0]  # Add ROI x offset
        y_uncropped = y_rect + roi[1]  # Add ROI y offset
    
    # Step 2: Convert uncropped rectified coordinates to normalized coordinates
    K_rect = P[:, :3]  # Intrinsic matrix for rectified camera
    point_rect_normalized = np.linalg.inv(K_rect) @ np.array([x_uncropped, y_uncropped, 1])
    
    # Step 3: Apply inverse rectification rotation
    point_original_normalized = R_rect.T @ point_rect_normalized
    
    # Step 4: Project back to rotated image coordinates
    if rotation_angle != 0:
        K_use = K_rotated
    else:
        K_use = K
    
    point_rotated_homogeneous = K_use @ point_original_normalized
    x_rotated = point_rotated_homogeneous[0] / point_rotated_homogeneous[2]
    y_rotated = point_rotated_homogeneous[1] / point_rotated_homogeneous[2]
    
    # Step 5: Apply inverse rotation if needed
    if rotation_angle != 0:
        # Apply inverse rotation to coordinates
        x_orig, y_orig = _apply_inverse_rotation_to_coordinates((x_rotated, y_rotated), rotation_angle, image_size)
    else:
        x_orig, y_orig = x_rotated, y_rotated
    
    return (x_orig, y_orig)


def transform_coordinates_to_rectified_vectorized(rect_params: Dict[str, Any], 
                                                coords_img1: np.ndarray, 
                                                coords_img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version: Transform arrays of coordinates from original images to rectified images.
    Pipeline: Original → Rotation (if vertical) → Rectification → Cropping (if alpha=0)
    
    Args:
        rect_params: Rectification parameters
        coords_img1: Nx2 array of (x, y) coordinates in first original image
        coords_img2: Nx2 array of (x, y) coordinates in second original image
        
    Returns:
        Tuple of (coords_rect1, coords_rect2) where each is Nx2 array of rectified coordinates
    """
    # Reconstruct rectification parameters
    K1 = np.array(rect_params['K1'])
    K2 = np.array(rect_params['K2'])
    K1_rotated = np.array(rect_params['K1_rotated'])
    K2_rotated = np.array(rect_params['K2_rotated'])
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    rotation_angle1 = rect_params.get('rotation_angle1', 0)
    rotation_angle2 = rect_params.get('rotation_angle2', 0)
    image_size = tuple(rect_params['image_size'])
    
    # Ensure inputs are numpy arrays
    coords_img1 = np.array(coords_img1, dtype=np.float32)
    coords_img2 = np.array(coords_img2, dtype=np.float32)
    
    # Step 1: Apply individual rotations if needed
    w, h = image_size
    
    # Handle image 1 rotation
    if rotation_angle1 != 0:
        if rotation_angle1 == 90:
            coords_img1_rot = np.column_stack([h - coords_img1[:, 1], coords_img1[:, 0]])
        elif rotation_angle1 == 180:
            coords_img1_rot = np.column_stack([w - coords_img1[:, 0], h - coords_img1[:, 1]])
        elif rotation_angle1 == 270:
            coords_img1_rot = np.column_stack([coords_img1[:, 1], w - coords_img1[:, 0]])
        K1_use = K1_rotated
    else:
        coords_img1_rot = coords_img1
        K1_use = K1
        
    # Handle image 2 rotation
    if rotation_angle2 != 0:
        if rotation_angle2 == 90:
            coords_img2_rot = np.column_stack([h - coords_img2[:, 1], coords_img2[:, 0]])
        elif rotation_angle2 == 180:
            coords_img2_rot = np.column_stack([w - coords_img2[:, 0], h - coords_img2[:, 1]])
        elif rotation_angle2 == 270:
            coords_img2_rot = np.column_stack([coords_img2[:, 1], w - coords_img2[:, 0]])
        K2_use = K2_rotated
    else:
        coords_img2_rot = coords_img2
        K2_use = K2
    
    # Reshape to (N, 1, 2) for OpenCV
    points1 = coords_img1_rot.reshape(-1, 1, 2)
    points2 = coords_img2_rot.reshape(-1, 1, 2)
    
    # Step 2: Apply rectification using OpenCV's undistortPoints
    points1_rect = cv2.undistortPoints(points1, K1_use, None, R=R1_rect, P=P1)
    points2_rect = cv2.undistortPoints(points2, K2_use, None, R=R2_rect, P=P2)
    
    # Reshape back to (N, 2)
    coords_rect1 = points1_rect.reshape(-1, 2)
    coords_rect2 = points2_rect.reshape(-1, 2)
    
    # Step 3: Apply custom padding adjustments if available
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        # Adjust for custom cropping and padding
        # First, convert to custom ROI coordinates
        coords_rect1[:, 0] -= rect_params['roi1_custom'][0]
        coords_rect1[:, 1] = coords_rect1[:, 1] - rect_params['roi1_custom'][1] + padding_info['pad_top_1']
        
        coords_rect2[:, 0] -= rect_params['roi2_custom'][0]
        coords_rect2[:, 1] = coords_rect2[:, 1] - rect_params['roi2_custom'][1] + padding_info['pad_top_2']
    
    return coords_rect1, coords_rect2


def transform_coordinates_from_rectified_vectorized(rect_params: Dict[str, Any], 
                                                  coords_rect1: np.ndarray, 
                                                  coords_rect2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version: Transform arrays of coordinates from rectified images back to original images.
    Pipeline: Cropping → Rectification → Rotation (if vertical) → Original
    
    Args:
        rect_params: Rectification parameters
        coords_rect1: Nx2 array of (x, y) coordinates in first rectified image
        coords_rect2: Nx2 array of (x, y) coordinates in second rectified image
        
    Returns:
        Tuple of (coords_orig1, coords_orig2) where each is Nx2 array of original coordinates
    """
    # Reconstruct rectification parameters
    K1 = np.array(rect_params['K1'])
    K2 = np.array(rect_params['K2'])
    K1_rotated = np.array(rect_params['K1_rotated'])
    K2_rotated = np.array(rect_params['K2_rotated'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    roi1 = rect_params.get('roi1', (0, 0, 0, 0))
    roi2 = rect_params.get('roi2', (0, 0, 0, 0))
    rotation_angle1 = rect_params.get('rotation_angle1', 0)
    rotation_angle2 = rect_params.get('rotation_angle2', 0)
    image_size = tuple(rect_params['image_size'])
    
    # Ensure inputs are numpy arrays
    coords_rect1 = np.array(coords_rect1, dtype=np.float64)
    coords_rect2 = np.array(coords_rect2, dtype=np.float64)
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
    # Handle custom padding for alpha=1
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        # Convert from custom padded coordinates back to original rectified coordinates
        coords_uncropped1 = coords_rect1.copy()
        coords_uncropped1[:, 0] += rect_params['roi1_custom'][0]
        coords_uncropped1[:, 1] = coords_uncropped1[:, 1] + rect_params['roi1_custom'][1] - padding_info['pad_top_1']
        
        coords_uncropped2 = coords_rect2.copy()
        coords_uncropped2[:, 0] += rect_params['roi2_custom'][0]
        coords_uncropped2[:, 1] = coords_uncropped2[:, 1] + rect_params['roi2_custom'][1] - padding_info['pad_top_2']
    else:
        # Use original ROI handling for alpha=0
        coords_uncropped1 = coords_rect1 + np.array([roi1[0], roi1[1]])
        coords_uncropped2 = coords_rect2 + np.array([roi2[0], roi2[1]])
    
    # Step 2: Convert to homogeneous coordinates
    ones = np.ones((coords_uncropped1.shape[0], 1))
    coords_hom1 = np.hstack([coords_uncropped1, ones])
    coords_hom2 = np.hstack([coords_uncropped2, ones])
    
    # Step 3: Convert to normalized coordinates
    K_rect1 = P1[:, :3]
    K_rect2 = P2[:, :3]
    coords_norm1 = (np.linalg.inv(K_rect1) @ coords_hom1.T).T
    coords_norm2 = (np.linalg.inv(K_rect2) @ coords_hom2.T).T
    
    # Step 4: Apply inverse rectification rotation
    coords_orig_norm1 = (R1_rect.T @ coords_norm1.T).T
    coords_orig_norm2 = (R2_rect.T @ coords_norm2.T).T
    
    # Step 5: Project back to rotated image coordinates using individual rotations
    if rotation_angle1 != 0:
        K1_use = K1_rotated
    else:
        K1_use = K1
        
    if rotation_angle2 != 0:
        K2_use = K2_rotated
    else:
        K2_use = K2
    
    coords_rotated_hom1 = (K1_use @ coords_orig_norm1.T).T
    coords_rotated_hom2 = (K2_use @ coords_orig_norm2.T).T
    
    # Convert back to 2D coordinates
    coords_rotated1 = coords_rotated_hom1[:, :2] / coords_rotated_hom1[:, 2:3]
    coords_rotated2 = coords_rotated_hom2[:, :2] / coords_rotated_hom2[:, 2:3]
    
    # Step 6: Apply individual inverse rotations if needed
    w, h = image_size
    
    # Handle image 1 inverse rotation
    if rotation_angle1 != 0:
        if rotation_angle1 == 90:
            coords_orig1 = np.column_stack([coords_rotated1[:, 1], h - coords_rotated1[:, 0]])
        elif rotation_angle1 == 180:
            coords_orig1 = np.column_stack([w - coords_rotated1[:, 0], h - coords_rotated1[:, 1]])
        elif rotation_angle1 == 270:
            coords_orig1 = np.column_stack([w - coords_rotated1[:, 1], coords_rotated1[:, 0]])
    else:
        coords_orig1 = coords_rotated1
        
    # Handle image 2 inverse rotation
    if rotation_angle2 != 0:
        if rotation_angle2 == 90:
            coords_orig2 = np.column_stack([coords_rotated2[:, 1], h - coords_rotated2[:, 0]])
        elif rotation_angle2 == 180:
            coords_orig2 = np.column_stack([w - coords_rotated2[:, 0], h - coords_rotated2[:, 1]])
        elif rotation_angle2 == 270:
            coords_orig2 = np.column_stack([w - coords_rotated2[:, 1], coords_rotated2[:, 0]])
    else:
        coords_orig2 = coords_rotated2
    
    return coords_orig1, coords_orig2


def transform_single_image_coordinates_to_rectified_vectorized(rect_params: Dict[str, Any], 
                                                            coords: np.ndarray, 
                                                            image_id: int) -> np.ndarray:
    """
    Vectorized version: Transform array of coordinates from a specific original image to its rectified version.
    Pipeline: Original → Rotation (if vertical) → Rectification → Cropping (if alpha=0)
    
    Args:
        rect_params: Rectification parameters
        coords: Nx2 array of (x, y) coordinates in the original image
        image_id: Image ID (1 for first image, 2 for second image)
        
    Returns:
        Nx2 array of (x_rect, y_rect) coordinates in the rectified image
    """
    # Reconstruct rectification parameters
    if image_id == 1:
        K = np.array(rect_params['K1'])
        K_rotated = np.array(rect_params['K1_rotated'])
        dist = np.array(rect_params['dist1'])
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
        rotation_angle = rect_params.get('rotation_angle1', 0)
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        K_rotated = np.array(rect_params['K2_rotated'])
        dist = np.array(rect_params['dist2'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
        rotation_angle = rect_params.get('rotation_angle2', 0)
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    # Ensure input is numpy array
    coords = np.array(coords, dtype=np.float32)
    image_size = tuple(rect_params['image_size'])
    
    # Step 1: Apply rotation if needed
    
    if rotation_angle != 0:
        # Apply rotation to coordinates vectorized
        w, h = image_size
        if rotation_angle == 90:
            coords_rot = np.column_stack([h - coords[:, 1], coords[:, 0]])
        elif rotation_angle == 180:
            coords_rot = np.column_stack([w - coords[:, 0], h - coords[:, 1]])
        elif rotation_angle == 270:
            coords_rot = np.column_stack([coords[:, 1], w - coords[:, 0]])
        K_use = K_rotated
    else:
        coords_rot = coords
        K_use = K
    
    # Reshape to (N, 1, 2) for OpenCV
    points = coords_rot.reshape(-1, 1, 2)
    
    # Step 2: Apply rectification using OpenCV's undistortPoints
    points_rect = cv2.undistortPoints(points, K_use, None, R=R_rect, P=P)
    
    # Reshape back to (N, 2)
    coords_rect = points_rect.reshape(-1, 2)
    
    # Step 3: Apply custom padding adjustments if available
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        if image_id == 1:
            roi_custom = rect_params['roi1_custom']
            pad_top = padding_info['pad_top_1']
        else:  # image_id == 2
            roi_custom = rect_params['roi2_custom']
            pad_top = padding_info['pad_top_2']
        
        # Adjust for custom cropping and padding
        coords_rect[:, 0] -= roi_custom[0]
        coords_rect[:, 1] = coords_rect[:, 1] - roi_custom[1] + pad_top
    
    return coords_rect


def transform_single_image_coordinates_from_rectified_vectorized(rect_params: Dict[str, Any], 
                                                              coords_rect: np.ndarray, 
                                                              image_id: int) -> np.ndarray:
    """
    Vectorized version: Transform array of coordinates from a specific rectified image back to its original version.
    Pipeline: Cropping → Rectification → Rotation (if vertical) → Original
    
    Args:
        rect_params: Rectification parameters
        coords_rect: Nx2 array of (x, y) coordinates in the rectified image
        image_id: Image ID (1 for first image, 2 for second image)
        
    Returns:
        Nx2 array of (x_orig, y_orig) coordinates in the original image
    """
    # Reconstruct rectification parameters
    if image_id == 1:
        K = np.array(rect_params['K1'])
        K_rotated = np.array(rect_params['K1_rotated'])
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
        roi = rect_params.get('roi1', (0, 0, 0, 0))
        rotation_angle = rect_params.get('rotation_angle1', 0)
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        K_rotated = np.array(rect_params['K2_rotated'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
        roi = rect_params.get('roi2', (0, 0, 0, 0))
        rotation_angle = rect_params.get('rotation_angle2', 0)
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    # Ensure input is numpy array
    coords_rect = np.array(coords_rect, dtype=np.float64)
    image_size = tuple(rect_params['image_size'])
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
    # Handle custom padding for alpha=1
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        
        if image_id == 1:
            roi_custom = rect_params['roi1_custom']
            pad_top = padding_info['pad_top_1']
        else:  # image_id == 2
            roi_custom = rect_params['roi2_custom']
            pad_top = padding_info['pad_top_2']
        
        # Convert from custom padded coordinates back to original rectified coordinates
        coords_uncropped = coords_rect.copy()
        coords_uncropped[:, 0] += roi_custom[0]
        coords_uncropped[:, 1] = coords_uncropped[:, 1] + roi_custom[1] - pad_top
    else:
        # Use original ROI handling for alpha=0
        coords_uncropped = coords_rect + np.array([roi[0], roi[1]])
    
    # Step 2: Convert to homogeneous coordinates
    ones = np.ones((coords_uncropped.shape[0], 1))
    coords_hom = np.hstack([coords_uncropped, ones])
    
    # Step 3: Convert to normalized coordinates
    K_rect = P[:, :3]
    coords_norm = (np.linalg.inv(K_rect) @ coords_hom.T).T
    
    # Step 4: Apply inverse rectification rotation
    coords_orig_norm = (R_rect.T @ coords_norm.T).T
    
    # Step 5: Project back to rotated image coordinates
    if rotation_angle != 0:
        K_use = K_rotated
    else:
        K_use = K
    
    coords_rotated_hom = (K_use @ coords_orig_norm.T).T
    
    # Convert back to 2D coordinates
    coords_rotated = coords_rotated_hom[:, :2] / coords_rotated_hom[:, 2:3]
    
    # Step 6: Apply inverse rotation if needed
    if rotation_angle != 0:
        # Apply inverse rotation to coordinates vectorized
        w, h = image_size
        if rotation_angle == 90:
            coords_orig = np.column_stack([coords_rotated[:, 1], h - coords_rotated[:, 0]])
        elif rotation_angle == 180:
            coords_orig = np.column_stack([w - coords_rotated[:, 0], h - coords_rotated[:, 1]])
        elif rotation_angle == 270:
            coords_orig = np.column_stack([w - coords_rotated[:, 1], coords_rotated[:, 0]])
    else:
        coords_orig = coords_rotated
    
    return coords_orig

def initalize_rectification(reconstruction: ColmapReconstruction, img1_id: int, img2_id: int, images_path: Path, output_dir: Path, alpha: float = 1.0) -> Dict[str, Any]:
    C1 = reconstruction.get_camera_center(img1_id)
    C2 = reconstruction.get_camera_center(img2_id)
    im_left, im_right = _determine_image_order_by_camera_centers(C1, C2, img1_id, img2_id)
    return compute_stereo_rectification(reconstruction, im_left, im_right, images_path, output_dir, alpha)


def process_single_pair(reconstruction: ColmapReconstruction, img1_id: int, img2_id: int, 
                       images_path: Path, output_dir: Path, pair_index: int = None, debug: bool = False, alpha: float = 1.0) -> bool:
    """
    Process a single stereo pair for rectification.
    
    Args:
        reconstruction: ColmapReconstruction object
        img1_id: First image ID
        img2_id: Second image ID
        images_path: Path to images directory
        output_dir: Output directory for this pair
        pair_index: Optional pair index for consecutive naming (pair_00, pair_01, etc.)
        debug: Whether to save intermediate debug images
        alpha: Free scaling parameter for rectification (0.0=more crop, 1.0=less crop, default: 1.0)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create pair-specific output directory
        if pair_index is not None:
            pair_dir = output_dir / f"pair_{pair_index:02d}"
        else:
            pair_dir = output_dir / f"pair_{img1_id:03d}_{img2_id:03d}"
        pair_dir.mkdir(exist_ok=True)
        
        print(f"Processing pair ({pair_index}): {img1_id} - {img2_id}")
        
        # Compute rectification
        rect_info = initalize_rectification(reconstruction, img1_id, img2_id, images_path, pair_dir, alpha)
        
        print(f"  Images: {rect_info['img1_name']} (ID: {rect_info['img1_id']}) and {rect_info['img2_name']} (ID: {rect_info['img2_id']})")
        print(f"  Rectification type: {rect_info['type']}")
        print(f"  Left image: {rect_info['left']}")
        print(f"  Right image: {rect_info['right']}")
        rotation_angle1 = rect_info.get('rotation_angle1', 0)
        rotation_angle2 = rect_info.get('rotation_angle2', 0)
        if rotation_angle1 != 0 or rotation_angle2 != 0:
            print(f"  Note: Images were rotated (Image1: {rotation_angle1}°, Image2: {rotation_angle2}°) to align orientations before rectification")
        
        # Rectify images
        print("  Rectifying images...")
        debug_dir = pair_dir / 'debug_stages' if debug else None
        rect1_img, rect2_img = rectify_images(rect_info, debug_dir)
        
        # Save rectified images as left.jpg and right.jpg
        left_img_path = pair_dir / 'left.jpg'
        right_img_path = pair_dir / 'right.jpg'
        imageio.imwrite(str(left_img_path), rect1_img)
        imageio.imwrite(str(right_img_path), rect2_img)
        
        # Update rectification info with new image paths
        rect_info['rect1_path'] = str(left_img_path)
        rect_info['rect2_path'] = str(right_img_path)
        
        # Save rectification info
        rect_info_path = pair_dir / 'rectification.json'
        
        # Convert numpy types to Python native types for JSON serialization
        rect_info_json = {}
        for key, value in rect_info.items():
            if isinstance(value, np.ndarray):
                rect_info_json[key] = value.tolist()
            elif isinstance(value, np.bool_):
                rect_info_json[key] = bool(value)
            elif isinstance(value, np.integer):
                rect_info_json[key] = int(value)
            elif isinstance(value, np.floating):
                rect_info_json[key] = float(value)
            else:
                rect_info_json[key] = value
        
        with open(rect_info_path, 'w') as f:
            json.dump(rect_info_json, f, indent=2)
        
        # Save intrinsics.txt file
        intrinsics_path = pair_dir / 'intrinsics.txt'
        
        # Extract rectified K matrix from projection matrices
        P1 = np.array(rect_info['P1'])
        P2 = np.array(rect_info['P2'])
        
        # The rectified K matrix is the same for both cameras (K_rect = P1[:, :3] = P2[:, :3])
        K_rect = P1[:, :3]
        
        # Calculate baseline from relative translation
        t_rel = np.array(rect_info['t_rel'])
        baseline = np.linalg.norm(t_rel)
        
        with open(intrinsics_path, 'w') as f:
            # Write rectified K matrix
            f.write(f"{K_rect[0,0]:.6f} {K_rect[0,1]:.6f} {K_rect[0,2]:.6f} {K_rect[1,0]:.6f} {K_rect[1,1]:.6f} {K_rect[1,2]:.6f} {K_rect[2,0]:.6f} {K_rect[2,1]:.6f} {K_rect[2,2]:.6f}\n")
            # Write baseline
            f.write(f"{baseline:.6f}\n")
        
        print(f"  ✓ Pair {img1_id}-{img2_id} completed successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing pair {img1_id}-{img2_id}: {e}")
        return False


def process_all_pairs(reconstruction: ColmapReconstruction, images_path: Path, output_dir: Path, pairs_per_image: int = 1, debug: bool = False, alpha: float = 1.0) -> None:
    """
    Process all stereo pairs from the reconstruction.
    
    Args:
        reconstruction: ColmapReconstruction object
        images_path: Path to images directory
        output_dir: Output directory
        pairs_per_image: Number of stereo pairs per image to process
        debug: Whether to save intermediate debug images
        alpha: Free scaling parameter for rectification (0.0=more crop, 1.0=less crop, default: 1.0)
    """
    print("Getting best stereo pairs from reconstruction...")
    
    # Get all pairs from the reconstruction
    pairs = reconstruction.get_best_pairs(pairs_per_image=pairs_per_image)

    # Convert pairs to list of tuples
    pair_list = []
    for img1_id, partner_ids in pairs.items():
        for img2_id in partner_ids:
            if img1_id < img2_id:  # Avoid duplicates
                pair_list.append((img1_id, img2_id))

    total_pairs = len(pair_list)
    print(f"Found {total_pairs} stereo pairs to process")
    
    if total_pairs == 0:
        print("No stereo pairs found in the reconstruction")
        return
    
    # Process each pair
    successful_pairs = 0
    failed_pairs = 0
    pair_index = 0
    
    for i, (img1_id, img2_id) in enumerate(pair_list):
        success = process_single_pair(reconstruction, img1_id, img2_id, images_path, output_dir, pair_index, debug, alpha)
        if success:
            successful_pairs += 1
        else:
            failed_pairs += 1
        pair_index += 1
    
    print(f"\nProcessing complete!")
    print(f"Successful pairs: {successful_pairs}")
    print(f"Failed pairs: {failed_pairs}")
    print(f"Total pairs: {total_pairs}")


def main():
    parser = argparse.ArgumentParser(description='Stereo rectification for COLMAP reconstructions')
    parser.add_argument('-s', '--scene_folder', required=True, 
                       help='Path to scene folder containing sparse/ and images/')
    parser.add_argument('-o', '--out_folder', required=True,
                       help='Output folder name (will be created under scene_folder)')
    parser.add_argument('--all-pairs', action='store_true',
                       help='Process all stereo pairs from the reconstruction')
    parser.add_argument('-p', '--pairs_per_image', type=int, default=1,
                       help='Number of stereo pairs per image to process (default: 1)')
    parser.add_argument('--debug', action='store_true',
                       help='Save intermediate debug images at each transformation stage')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Alpha parameter for stereo rectification (0.0=more crop, 1.0=less crop, default: 1.0)')
    parser.add_argument('img_id1', type=int, nargs='?', help='First image ID (required if not using --all-pairs)')
    parser.add_argument('img_id2', type=int, nargs='?', help='Second image ID (required if not using --all-pairs)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_pairs and (args.img_id1 is None or args.img_id2 is None):
        parser.error("Either --all-pairs must be specified, or both img_id1 and img_id2 must be provided")
    
    if args.all_pairs and (args.img_id1 is not None or args.img_id2 is not None):
        parser.error("Cannot specify both --all-pairs and individual image IDs")
    
    # Validate alpha parameter
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("Alpha parameter must be between 0.0 and 1.0")
    
    # Validate paths
    scene_folder = Path(args.scene_folder)
    if not scene_folder.exists():
        print(f"Error: Scene folder {scene_folder} does not exist")
        sys.exit(1)
    
    sparse_path = scene_folder / 'sparse'
    images_path = scene_folder / 'images'
    
    if not sparse_path.exists():
        print(f"Error: Sparse reconstruction folder {sparse_path} does not exist")
        sys.exit(1)
    
    if not images_path.exists():
        print(f"Error: Images folder {images_path} does not exist")
        sys.exit(1)
    
    # Load COLMAP reconstruction
    print(f"Loading COLMAP reconstruction from {sparse_path}")
    try:
        reconstruction = ColmapReconstruction(str(sparse_path))
    except Exception as e:
        print(f"Error loading reconstruction: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = scene_folder / args.out_folder
    os.makedirs(output_dir, exist_ok=True)

    if args.all_pairs:
        # Process all stereo pairs
        process_all_pairs(reconstruction, images_path, output_dir, args.pairs_per_image, args.debug, args.alpha)
    else:
        # Process single pair
        # Validate image IDs
        if not reconstruction.has_image(args.img_id1):
            print(f"Error: Image ID {args.img_id1} not found in reconstruction")
            sys.exit(1)
        
        if not reconstruction.has_image(args.img_id2):
            print(f"Error: Image ID {args.img_id2} not found in reconstruction")
            sys.exit(1)
        
        # Process single pair
        success = process_single_pair(reconstruction, args.img_id1, args.img_id2, images_path, output_dir, debug=args.debug, alpha=args.alpha)
        
        if success:
            print(f"\nRectification complete!")
            print(f"Output saved to: {output_dir}")
        else:
            print(f"\nRectification failed!")
            sys.exit(1)

if __name__ == '__main__':
    main()
