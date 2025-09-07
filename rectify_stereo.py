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


def mark_coordinate_on_image(image_path: str, x: float, y: float, output_path: str, 
                           color: Tuple[int, int, int] = (0, 255, 0), radius: int = 10):
    """
    Mark a coordinate on an image and save it.
    
    Args:
        image_path: Path to input image
        x, y: Coordinates to mark
        output_path: Path to save marked image
        color: BGR color for the marker
        radius: Radius of the marker circle
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Draw circle at the coordinate
    cv2.circle(img, (int(x), int(y)), radius, color, -1)
    cv2.circle(img, (int(x), int(y)), radius + 2, (0, 0, 0), 2)  # Black border
    
    # Add text label
    cv2.putText(img, f"({x:.1f}, {y:.1f})", (int(x) + radius + 5, int(y) - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(output_path, img)


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
    
    # Use OpenCV's undistortPoints for reliable transformation
    # This handles the undistortion and rectification in one step
    point1_rect = cv2.undistortPoints(
        np.array([[[x1, y1]]], dtype=np.float32), 
        K1, dist1, R=R1_rect, P=P1
    )[0, 0]
    
    point2_rect = cv2.undistortPoints(
        np.array([[[x2, y2]]], dtype=np.float32), 
        K2, dist2, R=R2_rect, P=P2
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


def test_coordinate_transformations(rect_params: Dict[str, Any], 
                                  test_coords: Tuple[Tuple[float, float], Tuple[float, float]],
                                  tolerance: float = 5.0) -> bool:
    """
    Unit test for coordinate transformations.
    
    Args:
        rect_params: Rectification parameters
        test_coords: ((x1, y1), (x2, y2)) test coordinates
        tolerance: Maximum allowed error in pixels
        
    Returns:
        True if test passes, False otherwise
    """
    coords_orig1, coords_orig2 = test_coords
    
    # Forward transformation: original -> rectified
    coords_rect1, coords_rect2 = transform_coordinates_to_rectified(rect_params, coords_orig1, coords_orig2)
    
    # Reverse transformation: rectified -> original
    coords_back1, coords_back2 = transform_coordinates_from_rectified(rect_params, coords_rect1, coords_rect2)
    
    # Check if we get back close to the original coordinates
    error1 = np.sqrt((coords_orig1[0] - coords_back1[0])**2 + (coords_orig1[1] - coords_back1[1])**2)
    error2 = np.sqrt((coords_orig2[0] - coords_back2[0])**2 + (coords_orig2[1] - coords_back2[1])**2)
    
    print(f"Coordinate transformation test:")
    print(f"  Original coords: {coords_orig1}, {coords_orig2}")
    print(f"  Rectified coords: {coords_rect1}, {coords_rect2}")
    print(f"  Back-transformed coords: {coords_back1}, {coords_back2}")
    print(f"  Errors: {error1:.2f}, {error2:.2f} pixels")
    print(f"  Tolerance: {tolerance} pixels")
    
    success = error1 <= tolerance and error2 <= tolerance
    print(f"  Test {'PASSED' if success else 'FAILED'}")
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Stereo rectification for COLMAP reconstructions')
    parser.add_argument('-s', '--scene_folder', required=True, 
                       help='Path to scene folder containing sparse/ and images/')
    parser.add_argument('-o', '--out_folder', required=True,
                       help='Output folder name (will be created under scene_folder)')
    parser.add_argument('--debug', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'),
                       help='Debug mode: mark coordinates (x0,y0) in first image and (x1,y1) in second image')
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
    
    # Debug mode: mark coordinates and test transformations
    if args.debug:
        x0, y0, x1, y1 = args.debug
        print(f"\nDebug mode: Marking coordinates ({x0}, {y0}) and ({x1}, {y1})")
        
        # Mark coordinates on original images
        img1_marked_path = output_dir / f"{Path(img1_name).stem}_marked_original.jpg"
        img2_marked_path = output_dir / f"{Path(img2_name).stem}_marked_original.jpg"
        
        mark_coordinate_on_image(str(img1_path), x0, y0, str(img1_marked_path), (0, 255, 0))
        mark_coordinate_on_image(str(img2_path), x1, y1, str(img2_marked_path), (0, 255, 0))
        print(f"Marked original images saved to: {img1_marked_path}, {img2_marked_path}")
        
        # Transform coordinates to rectified space
        coords_rect1, coords_rect2 = transform_coordinates_to_rectified(
            rect_params, (x0, y0), (x1, y1)
        )
        print(f"Rectified coordinates: {coords_rect1}, {coords_rect2}")
        
        # Mark coordinates on rectified images
        rect1_marked_path = output_dir / f"{Path(img1_name).stem}_marked_rectified.jpg"
        rect2_marked_path = output_dir / f"{Path(img2_name).stem}_marked_rectified.jpg"
        
        mark_coordinate_on_image(rect1_path, coords_rect1[0], coords_rect1[1], str(rect1_marked_path), (255, 0, 0))
        mark_coordinate_on_image(rect2_path, coords_rect2[0], coords_rect2[1], str(rect2_marked_path), (255, 0, 0))
        print(f"Marked rectified images saved to: {rect1_marked_path}, {rect2_marked_path}")
        
        # Test reverse transformation
        coords_back1, coords_back2 = transform_coordinates_from_rectified(
            rect_params, coords_rect1, coords_rect2
        )
        print(f"Back-transformed coordinates: {coords_back1}, {coords_back2}")
        
        # Run unit test
        print("\nRunning coordinate transformation unit test...")
        test_success = test_coordinate_transformations(rect_params, ((x0, y0), (x1, y1)), tolerance=10.0)
        
        # Mark back-transformed coordinates on original images
        img1_back_marked_path = output_dir / f"{Path(img1_name).stem}_marked_back_transformed.jpg"
        img2_back_marked_path = output_dir / f"{Path(img2_name).stem}_marked_back_transformed.jpg"
        
        mark_coordinate_on_image(str(img1_path), coords_back1[0], coords_back1[1], str(img1_back_marked_path), (0, 0, 255))
        mark_coordinate_on_image(str(img2_path), coords_back2[0], coords_back2[1], str(img2_back_marked_path), (0, 0, 255))
        print(f"Back-transformed marked images saved to: {img1_back_marked_path}, {img2_back_marked_path}")
        
        if test_success:
            print("✓ All coordinate transformations working correctly!")
        else:
            print("⚠ Coordinate transformation test failed - check the results")
    
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
