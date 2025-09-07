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
                                img1_id: int, img2_id: int) -> Dict[str, Any]:
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
    
    return {
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


def check_rectification_type_and_order(rect_params: Dict[str, Any], img1_name: str, img2_name: str) -> Tuple[str, str, str, str, str]:
    """
    Check if rectification is horizontal or vertical and determine spatial order.
    
    Args:
        rect_params: Rectification parameters dictionary
        img1_name: Name of first image
        img2_name: Name of second image
        
    Returns:
        Tuple of (rectification_type, top_image_name, bottom_image_name, left_image_name, right_image_name)
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
    
    # Check horizontal rectification
    if abs(F[1, 2] + 1) < 0.1 and abs(F[2, 1] - 1) < 0.1:
        # For horizontal rectification, determine which camera is left/right
        # Compare the x-coordinate of camera centers in world coordinates
        t_rel = np.array(rect_params['t_rel'])
        
        # If t_rel[0] > 0, camera 2 is to the right of camera 1
        # If t_rel[0] < 0, camera 1 is to the right of camera 2
        if t_rel[0] > 0:
            # Camera 1 is left, camera 2 is right
            left_name = f"{Path(img1_name).stem}_rectified.jpg"
            right_name = f"{Path(img2_name).stem}_rectified.jpg"
        else:
            # Camera 2 is left, camera 1 is right
            left_name = f"{Path(img2_name).stem}_rectified.jpg"
            right_name = f"{Path(img1_name).stem}_rectified.jpg"
        
        return 'horizontal', '', '', left_name, right_name
    # Check vertical rectification  
    elif abs(F[0, 2] - 1) < 0.1 and abs(F[2, 0] + 1) < 0.1:
        # For vertical rectification, determine which camera is higher
        # Compare the y-coordinate of camera centers in world coordinates
        t_rel = np.array(rect_params['t_rel'])
        
        # If t_rel[1] > 0, camera 2 is higher than camera 1
        # If t_rel[1] < 0, camera 1 is higher than camera 2
        if t_rel[1] > 0:
            # Camera 2 is higher (top), camera 1 is lower (bottom)
            top_name = f"{Path(img2_name).stem}_rectified.jpg"
            bottom_name = f"{Path(img1_name).stem}_rectified.jpg"
        else:
            # Camera 1 is higher (top), camera 2 is lower (bottom)
            top_name = f"{Path(img1_name).stem}_rectified.jpg"
            bottom_name = f"{Path(img2_name).stem}_rectified.jpg"
        
        return 'vertical', top_name, bottom_name, '', ''
    else:
        # Fallback: check which direction has the largest off-diagonal elements
        horizontal_strength = abs(F[1, 2]) + abs(F[2, 1])
        vertical_strength = abs(F[0, 2]) + abs(F[2, 0])
        
        if horizontal_strength > vertical_strength:
            # Horizontal fallback
            t_rel = np.array(rect_params['t_rel'])
            if t_rel[0] > 0:
                left_name = f"{Path(img1_name).stem}_rectified.jpg"
                right_name = f"{Path(img2_name).stem}_rectified.jpg"
            else:
                left_name = f"{Path(img2_name).stem}_rectified.jpg"
                right_name = f"{Path(img1_name).stem}_rectified.jpg"
            return 'horizontal', '', '', left_name, right_name
        else:
            # For vertical fallback, use the same logic
            t_rel = np.array(rect_params['t_rel'])
            if t_rel[1] > 0:
                top_name = f"{Path(img2_name).stem}_rectified.jpg"
                bottom_name = f"{Path(img1_name).stem}_rectified.jpg"
            else:
                top_name = f"{Path(img1_name).stem}_rectified.jpg"
                bottom_name = f"{Path(img2_name).stem}_rectified.jpg"
            return 'vertical', top_name, bottom_name, '', ''


def rectify_images(img1_path: str, img2_path: str, rect_params: Dict[str, Any], 
                   output_dir: str) -> Tuple[str, str]:
    """
    Rectify two images using the computed rectification parameters.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        rect_params: Rectification parameters
        output_dir: Output directory for rectified images
        
    Returns:
        Tuple of (rectified_img1_path, rectified_img2_path)
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        raise ValueError(f"Could not load image: {img1_path}")
    if img2 is None:
        raise ValueError(f"Could not load image: {img2_path}")
    
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
    
    # Save rectified images
    os.makedirs(output_dir, exist_ok=True)
    
    img1_name = Path(img1_path).stem
    img2_name = Path(img2_path).stem
    
    rect1_path = os.path.join(output_dir, f"{img1_name}_rectified.jpg")
    rect2_path = os.path.join(output_dir, f"{img2_name}_rectified.jpg")
    
    cv2.imwrite(rect1_path, img1_rect)
    cv2.imwrite(rect2_path, img2_rect)
    
    return rect1_path, rect2_path


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
    
    # Validate image IDs
    if not reconstruction.has_image(args.img_id1):
        print(f"Error: Image ID {args.img_id1} not found in reconstruction")
        sys.exit(1)
    
    if not reconstruction.has_image(args.img_id2):
        print(f"Error: Image ID {args.img_id2} not found in reconstruction")
        sys.exit(1)
    
    # Get image names
    img1_name = reconstruction.get_image_name(args.img_id1)
    img2_name = reconstruction.get_image_name(args.img_id2)
    
    print(f"Processing images: {img1_name} (ID: {args.img_id1}) and {img2_name} (ID: {args.img_id2})")
    
    # Compute stereo rectification
    print("Computing stereo rectification parameters...")
    rect_params = compute_stereo_rectification(reconstruction, args.img_id1, args.img_id2)
    
    # Check rectification type and determine spatial order
    rect_type, top_image, bottom_image, left_image, right_image = check_rectification_type_and_order(rect_params, img1_name, img2_name)
    print(f"Rectification type: {rect_type}")
    if rect_type == 'vertical':
        print(f"Top image: {top_image}")
        print(f"Bottom image: {bottom_image}")
    elif rect_type == 'horizontal':
        print(f"Left image: {left_image}")
        print(f"Right image: {right_image}")
    
    # Get image paths
    img1_path = images_path / img1_name
    img2_path = images_path / img2_name
    
    if not img1_path.exists():
        print(f"Error: Image file {img1_path} does not exist")
        sys.exit(1)
    
    if not img2_path.exists():
        print(f"Error: Image file {img2_path} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir = scene_folder / args.out_folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Rectify images
    print("Rectifying images...")
    rect1_path, rect2_path = rectify_images(
        str(img1_path), str(img2_path), rect_params, str(output_dir)
    )
    
    # Prepare rectification info
    rect_info = {
        'image_ids': [args.img_id1, args.img_id2],
        'image_names': [img1_name, img2_name],
        'rectified_image_paths': [rect1_path, rect2_path],
        'rectification_type': rect_type,
        'rectification_parameters': rect_params
    }
    
    # Add spatial information based on rectification type
    if rect_type == 'vertical':
        rect_info['top'] = top_image
        rect_info['bottom'] = bottom_image
        rect_info['left'] = ""
        rect_info['right'] = ""
    elif rect_type == 'horizontal':
        rect_info['top'] = ""
        rect_info['bottom'] = ""
        rect_info['left'] = left_image
        rect_info['right'] = right_image
    else:
        rect_info['top'] = ""
        rect_info['bottom'] = ""
        rect_info['left'] = ""
        rect_info['right'] = ""
    
    # Save rectification info
    rect_info_path = output_dir / 'rectification.json'
    with open(rect_info_path, 'w') as f:
        json.dump(rect_info, f, indent=2)
    
    print(f"Rectification complete!")
    print(f"Rectified images saved to: {output_dir}")
    print(f"Rectification info saved to: {rect_info_path}")
    print(f"Rectification type: {rect_type}")


if __name__ == '__main__':
    main()
