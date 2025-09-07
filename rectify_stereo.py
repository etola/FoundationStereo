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
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from colmap_utils import ColmapReconstruction


def compute_stereo_rectification(reconstruction: ColmapReconstruction, 
                                img1_id: int, img2_id: int, images_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Compute stereo rectification parameters for two images.
    
    Args:
        reconstruction: ColmapReconstruction object
        img1_id: First image ID
        img2_id: Second image ID
        
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
    
    # Compute essential matrix
    t_skew = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    E = t_skew @ R_rel
    
    # Stereo rectification using OpenCV functions
    R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1, K2, dist2, image_size, R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

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
        'dist1': dist1.tolist(),
        'dist2': dist2.tolist(),
        'R1': R1.tolist(),
        'R2': R2.tolist(),
        't1': t1.tolist(),
        't2': t2.tolist(),
        'R_rel': R_rel.tolist(),
        't_rel': t_rel.tolist(),
        'E': E.tolist(),
        'R1_rect': R1_rect.tolist(),
        'R2_rect': R2_rect.tolist(),
        'P1': P1.tolist(),
        'P2': P2.tolist(),
        'Q': Q.tolist(),
        'image_size': image_size,
        'roi1': roi1,
        'roi2': roi2
    }
    determine_rectification_type(rect_info)
    return rect_info

def determine_rectification_type(rect_params: Dict[str, Any]) -> None:
    """
    Determine if rectification is horizontal or vertical and update rect_params with image IDs.
    
    Args:
        rect_params: Rectification parameters dictionary (will be modified)
        
    Returns:
        Rectification type: 'horizontal' or 'vertical'
    """
    # Extract rectified projection matrices
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    
    # Compute fundamental matrix from rectified projection matrices
    # F = [e2]_x * P2 * P1^+
    # where P1^+ is the pseudo-inverse of P1 and e2 is the epipole in image 2
    
    # For rectified stereo, the fundamental matrix should be:
    # F = [0, 0, 0; 0, 0, -1; 0, 1, 0] for horizontal rectification
    # F = [0, 0, 1; 0, 0, 0; -1, 0, 0] for vertical rectification
    
    # Compute epipole in second image
    # e2 = P2 * C1 where C1 is camera center of first camera
    # For rectified cameras, C1 = [0, 0, 0, 1] in rectified coordinate system
    C1_rect = np.array([0, 0, 0, 1])
    e2 = P2 @ C1_rect
    e2 = e2[:3] / e2[2] if e2[2] != 0 else e2[:3]
    
    # Create skew-symmetric matrix for epipole
    e2_skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    
    # Compute fundamental matrix
    P1_pinv = np.linalg.pinv(P1)
    F = e2_skew @ P2 @ P1_pinv
    
    # Check the structure of the fundamental matrix
    # For horizontal rectification: F[1,2] = -1, F[2,1] = 1, others ≈ 0
    # For vertical rectification: F[0,2] = 1, F[2,0] = -1, others ≈ 0

    rect_params['left'] = None
    rect_params['right'] = None
    rect_params['top'] = None
    rect_params['bottom'] = None

    img1_id = rect_params['img1_id']
    img2_id = rect_params['img2_id']

    # Check horizontal rectification
    if abs(F[1, 2] + 1) < 0.1 and abs(F[2, 1] - 1) < 0.1:
        # For horizontal rectification, determine which camera is left/right
        # Compare the x-coordinate of camera centers in world coordinates
        t_rel = np.array(rect_params['t_rel'])
        
        # If t_rel[0] > 0, camera 2 is to the right of camera 1
        # If t_rel[0] < 0, camera 1 is to the right of camera 2
        if t_rel[0] > 0:
            # Camera 1 is left, camera 2 is right
            rect_params['left'] = img1_id
            rect_params['right'] = img2_id
        else:
            # Camera 2 is left, camera 1 is right
            rect_params['left'] = img2_id
            rect_params['right'] = img1_id
        rect_params['type'] = 'horizontal'
    # Check vertical rectification  
    elif abs(F[0, 2] - 1) < 0.1 and abs(F[2, 0] + 1) < 0.1:
        # For vertical rectification, determine which camera is higher
        # Compare the y-coordinate of camera centers in world coordinates
        t_rel = np.array(rect_params['t_rel'])
        
        # If t_rel[1] > 0, camera 2 is higher than camera 1
        # If t_rel[1] < 0, camera 1 is higher than camera 2
        if t_rel[1] > 0:
            # Camera 2 is higher (top), camera 1 is lower (bottom)
            rect_params['top'] = img2_id
            rect_params['bottom'] = img1_id
        else:
            # Camera 1 is higher (top), camera 2 is lower (bottom)
            rect_params['top'] = img1_id
            rect_params['bottom'] = img2_id
        rect_params['type'] = 'vertical'
    else:
        # Fallback: check which direction has the largest off-diagonal elements
        horizontal_strength = abs(F[1, 2]) + abs(F[2, 1])
        vertical_strength = abs(F[0, 2]) + abs(F[2, 0])
        
        if horizontal_strength > vertical_strength:
            # Horizontal fallback
            t_rel = np.array(rect_params['t_rel'])
            if t_rel[0] > 0:
                rect_params['left'] = img1_id
                rect_params['right'] = img2_id
            else:
                rect_params['left'] = img2_id
                rect_params['right'] = img1_id
            rect_params['type'] = 'horizontal'
        else:
            # For vertical fallback, use the same logic
            t_rel = np.array(rect_params['t_rel'])
            if t_rel[1] > 0:
                rect_params['top'] = img2_id
                rect_params['bottom'] = img1_id
            else:
                rect_params['top'] = img1_id
                rect_params['bottom'] = img2_id
            rect_params['type'] = 'vertical'


def rectify_images(rect_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify two images using the computed rectification parameters.
    
    Args:
        rect_params: Rectification parameters
        
    Returns:
        Tuple of (rectified_img1, rectified_img2)
    """
    # Load images
    img1 = cv2.imread(rect_params['img1_path'])
    img2 = cv2.imread(rect_params['img2_path'])
    
    if img1 is None:
        raise ValueError(f"Could not load image: {rect_params['img1_path']}")
    if img2 is None:
        raise ValueError(f"Could not load image: {rect_params['img2_path']}")
    
    # Reconstruct rectification parameters
    K1 = np.array(rect_params['K1'])
    K2 = np.array(rect_params['K2'])
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    image_size = tuple(rect_params['image_size'])
    
    # Compute rectification maps
    map1_x, map1_y = cv2.initUndistortRectifyMap(K1, dist1, R1_rect, P1, image_size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K2, dist2, R2_rect, P2, image_size, cv2.CV_32FC1)
    
    # Apply rectification
    img1_rect = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)
    
    return img1_rect, img2_rect




def transform_coordinates_to_rectified(rect_params: Dict[str, Any], 
                                     coords_img1: Tuple[float, float], 
                                     coords_img2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Transform coordinates from original images to rectified images using OpenCV's undistortPoints.
    This is more reliable than manual matrix operations.
    
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
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    
    x1, y1 = coords_img1
    x2, y2 = coords_img2
    
    # Use OpenCV's undistortPoints for reliable transformation (assuming no distortion)
    # This handles the undistortion and rectification in one step
    point1_rect = cv2.undistortPoints(
        np.array([[[x1, y1]]], dtype=np.float32), 
        K1, None, R=R1_rect, P=P1
    )[0, 0]
    
    point2_rect = cv2.undistortPoints(
        np.array([[[x2, y2]]], dtype=np.float32), 
        K2, None, R=R2_rect, P=P2
    )[0, 0]
    
    return (point1_rect[0], point1_rect[1]), (point2_rect[0], point2_rect[1])


def transform_coordinates_from_rectified(rect_params: Dict[str, Any], 
                                       coords_rect1: Tuple[float, float], 
                                       coords_rect2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Transform coordinates from rectified images back to original images.
    Based on understanding of OpenCV's stereoRectify with alpha=0 cropping.
    
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
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    roi1 = rect_params.get('roi1', (0, 0, 0, 0))
    roi2 = rect_params.get('roi2', (0, 0, 0, 0))
    
    x1_rect, y1_rect = coords_rect1
    x2_rect, y2_rect = coords_rect2
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
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
    
    # Step 4: Project back to original image coordinates
    # For the first camera, we need to account for the translation in P1
    # P1 = K_rect * [R1_rect | 0], so the translation is zero
    point1_orig_homogeneous = K1 @ point1_original_normalized
    x1_orig = point1_orig_homogeneous[0] / point1_orig_homogeneous[2]
    y1_orig = point1_orig_homogeneous[1] / point1_orig_homogeneous[2]
    
    # For the second camera, P2 = K_rect * [R2_rect | t_rect]
    # The translation in P2 represents the baseline between cameras
    # In stereo rectification, both cameras have the same rectification rotation
    # So we can use the same approach as the first camera
    point2_orig_homogeneous = K2 @ point2_original_normalized
    x2_orig = point2_orig_homogeneous[0] / point2_orig_homogeneous[2]
    y2_orig = point2_orig_homogeneous[1] / point2_orig_homogeneous[2]
    
    return (x1_orig, y1_orig), (x2_orig, y2_orig)


def transform_single_image_coordinates_to_rectified(rect_params: Dict[str, Any], 
                                                  coords: Tuple[float, float], 
                                                  image_id: int) -> Tuple[float, float]:
    """
    Transform coordinates from a specific original image to its rectified version.
    
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
        dist = np.array(rect_params['dist1'])
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        dist = np.array(rect_params['dist2'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    x, y = coords
    
    # Use OpenCV's undistortPoints for reliable transformation (assuming no distortion)
    point_rect = cv2.undistortPoints(
        np.array([[[x, y]]], dtype=np.float32), 
        K, None, R=R_rect, P=P
    )[0, 0]
    
    return (point_rect[0], point_rect[1])


def transform_single_image_coordinates_from_rectified(rect_params: Dict[str, Any], 
                                                    coords_rect: Tuple[float, float], 
                                                    image_id: int) -> Tuple[float, float]:
    """
    Transform coordinates from a specific rectified image back to its original version.
    
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
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
        roi = rect_params.get('roi1', (0, 0, 0, 0))
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
        roi = rect_params.get('roi2', (0, 0, 0, 0))
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    x_rect, y_rect = coords_rect
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
    x_uncropped = x_rect + roi[0]  # Add ROI x offset
    y_uncropped = y_rect + roi[1]  # Add ROI y offset
    
    # Step 2: Convert uncropped rectified coordinates to normalized coordinates
    K_rect = P[:, :3]  # Intrinsic matrix for rectified camera
    point_rect_normalized = np.linalg.inv(K_rect) @ np.array([x_uncropped, y_uncropped, 1])
    
    # Step 3: Apply inverse rectification rotation
    point_original_normalized = R_rect.T @ point_rect_normalized
    
    # Step 4: Project back to original image coordinates
    point_orig_homogeneous = K @ point_original_normalized
    x_orig = point_orig_homogeneous[0] / point_orig_homogeneous[2]
    y_orig = point_orig_homogeneous[1] / point_orig_homogeneous[2]
    
    return (x_orig, y_orig)


def transform_coordinates_to_rectified_vectorized(rect_params: Dict[str, Any], 
                                                coords_img1: np.ndarray, 
                                                coords_img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version: Transform arrays of coordinates from original images to rectified images.
    
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
    dist1 = np.array(rect_params['dist1'])
    dist2 = np.array(rect_params['dist2'])
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    
    # Ensure inputs are numpy arrays
    coords_img1 = np.array(coords_img1, dtype=np.float32)
    coords_img2 = np.array(coords_img2, dtype=np.float32)
    
    # Reshape to (N, 1, 2) for OpenCV
    points1 = coords_img1.reshape(-1, 1, 2)
    points2 = coords_img2.reshape(-1, 1, 2)
    
    # Use OpenCV's undistortPoints for vectorized transformation (assuming no distortion)
    points1_rect = cv2.undistortPoints(points1, K1, None, R=R1_rect, P=P1)
    points2_rect = cv2.undistortPoints(points2, K2, None, R=R2_rect, P=P2)
    
    # Reshape back to (N, 2)
    coords_rect1 = points1_rect.reshape(-1, 2)
    coords_rect2 = points2_rect.reshape(-1, 2)
    
    return coords_rect1, coords_rect2


def transform_coordinates_from_rectified_vectorized(rect_params: Dict[str, Any], 
                                                  coords_rect1: np.ndarray, 
                                                  coords_rect2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version: Transform arrays of coordinates from rectified images back to original images.
    
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
    R1_rect = np.array(rect_params['R1_rect'])
    R2_rect = np.array(rect_params['R2_rect'])
    P1 = np.array(rect_params['P1'])
    P2 = np.array(rect_params['P2'])
    roi1 = rect_params.get('roi1', (0, 0, 0, 0))
    roi2 = rect_params.get('roi2', (0, 0, 0, 0))
    
    # Ensure inputs are numpy arrays
    coords_rect1 = np.array(coords_rect1, dtype=np.float64)
    coords_rect2 = np.array(coords_rect2, dtype=np.float64)
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
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
    
    # Step 5: Project back to original image coordinates
    coords_orig_hom1 = (K1 @ coords_orig_norm1.T).T
    coords_orig_hom2 = (K2 @ coords_orig_norm2.T).T
    
    # Step 6: Convert back to 2D coordinates
    coords_orig1 = coords_orig_hom1[:, :2] / coords_orig_hom1[:, 2:3]
    coords_orig2 = coords_orig_hom2[:, :2] / coords_orig_hom2[:, 2:3]
    
    return coords_orig1, coords_orig2


def transform_single_image_coordinates_to_rectified_vectorized(rect_params: Dict[str, Any], 
                                                            coords: np.ndarray, 
                                                            image_id: int) -> np.ndarray:
    """
    Vectorized version: Transform array of coordinates from a specific original image to its rectified version.
    
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
        dist = np.array(rect_params['dist1'])
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        dist = np.array(rect_params['dist2'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    # Ensure input is numpy array
    coords = np.array(coords, dtype=np.float32)
    
    # Reshape to (N, 1, 2) for OpenCV
    points = coords.reshape(-1, 1, 2)
    
    # Use OpenCV's undistortPoints for vectorized transformation (assuming no distortion)
    points_rect = cv2.undistortPoints(points, K, None, R=R_rect, P=P)
    
    # Reshape back to (N, 2)
    coords_rect = points_rect.reshape(-1, 2)
    
    return coords_rect


def transform_single_image_coordinates_from_rectified_vectorized(rect_params: Dict[str, Any], 
                                                              coords_rect: np.ndarray, 
                                                              image_id: int) -> np.ndarray:
    """
    Vectorized version: Transform array of coordinates from a specific rectified image back to its original version.
    
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
        R_rect = np.array(rect_params['R1_rect'])
        P = np.array(rect_params['P1'])
        roi = rect_params.get('roi1', (0, 0, 0, 0))
    elif image_id == 2:
        K = np.array(rect_params['K2'])
        R_rect = np.array(rect_params['R2_rect'])
        P = np.array(rect_params['P2'])
        roi = rect_params.get('roi2', (0, 0, 0, 0))
    else:
        raise ValueError(f"Invalid image_id: {image_id}. Must be 1 or 2.")
    
    # Ensure input is numpy array
    coords_rect = np.array(coords_rect, dtype=np.float64)
    
    # Step 1: Convert cropped rectified coordinates to uncropped coordinates
    coords_uncropped = coords_rect + np.array([roi[0], roi[1]])
    
    # Step 2: Convert to homogeneous coordinates
    ones = np.ones((coords_uncropped.shape[0], 1))
    coords_hom = np.hstack([coords_uncropped, ones])
    
    # Step 3: Convert to normalized coordinates
    K_rect = P[:, :3]
    coords_norm = (np.linalg.inv(K_rect) @ coords_hom.T).T
    
    # Step 4: Apply inverse rectification rotation
    coords_orig_norm = (R_rect.T @ coords_norm.T).T
    
    # Step 5: Project back to original image coordinates
    coords_orig_hom = (K @ coords_orig_norm.T).T
    
    # Step 6: Convert back to 2D coordinates
    coords_orig = coords_orig_hom[:, :2] / coords_orig_hom[:, 2:3]
    
    return coords_orig

def initalize_rectification(reconstruction: ColmapReconstruction, img1_id: int, img2_id: int, images_path: Path, output_dir: Path) -> Dict[str, Any]:
    rect_info = compute_stereo_rectification(reconstruction, img1_id, img2_id, images_path, output_dir)
    if rect_info['type'] == 'vertical':
        if rect_info['top'] == rect_info['img1_id'] and rect_info['bottom'] == rect_info['img2_id']:
            return rect_info
        else:
            return compute_stereo_rectification(reconstruction, img2_id, img1_id, images_path, output_dir)
    elif rect_info['type'] == 'horizontal':
        if rect_info['left'] == rect_info['img1_id'] and rect_info['right'] == rect_info['img2_id']:
            return rect_info
        else:
            return compute_stereo_rectification(reconstruction, img2_id, img1_id, images_path, output_dir)
    else:
        raise ValueError(f"Invalid rectification type: {rect_info['type']}")


def main():
    parser = argparse.ArgumentParser(description='Stereo rectification for COLMAP reconstructions')
    parser.add_argument('-s', '--scene_folder', required=True, 
                       help='Path to scene folder containing sparse/ and images/')
    parser.add_argument('-o', '--out_folder', required=True,
                       help='Output folder name (will be created under scene_folder)')
    parser.add_argument('img_id1', type=int, help='First image ID')
    parser.add_argument('img_id2', type=int, help='Second image ID')
    
    args = parser.parse_args()
    
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

    # Validate image IDs
    if not reconstruction.has_image(args.img_id1):
        print(f"Error: Image ID {args.img_id1} not found in reconstruction")
        sys.exit(1)
    
    if not reconstruction.has_image(args.img_id2):
        print(f"Error: Image ID {args.img_id2} not found in reconstruction")
        sys.exit(1)
    
    rect_info = initalize_rectification(reconstruction, args.img_id1, args.img_id2, images_path, output_dir)
    print(f"Processing images: {rect_info['img1_name']} (ID: {rect_info['img1_id']}) and {rect_info['img2_name']} (ID: {rect_info['img2_id']})")
    
    print(f"Rectification type: {rect_info['type']}")
    if rect_info['type'] == 'vertical':
        print(f"Top image: {rect_info['top']}")
        print(f"Bottom image: {rect_info['bottom']}")
    elif rect_info['type'] == 'horizontal':
        print(f"Left image: {rect_info['left']}")
        print(f"Right image: {rect_info['right']}")

    # Rectify images
    print("Rectifying images...")
    rect1_img, rect2_img = rectify_images(rect_info)

    cv2.imwrite(rect_info['rect1_path'], rect1_img)
    cv2.imwrite(rect_info['rect2_path'], rect2_img)

    # Save rectification info
    rect_info_path = output_dir / 'rectification.json'
    with open(rect_info_path, 'w') as f:
        json.dump(rect_info, f, indent=2)
    
    print(f"Rectification complete!")
    print(f"Rectified images saved to: {output_dir}")
    print(f"Rectification info saved to: {rect_info_path}")
    print(f"Rectification type: {rect_info['type']}")

if __name__ == '__main__':
    main()
