#!/usr/bin/env python3
"""
Test script for stereo rectification coordinate transformations.

This script contains comprehensive tests for all coordinate transformation functions
in the rectify_stereo.py module, including debug functionality for marking coordinates
on images.

The script now supports alpha parameter testing (default: 1.0):
- alpha=0.0: Uses OpenCV's aggressive cropping (small ROI)
- alpha=1.0: Uses custom cropping with alignment padding (maintains rectification)

When alpha=1.0, additional debug images are generated:
- img1_2_rectified_full.jpg, img2_2_rectified_full.jpg: Original rectified images
- img1_3_cropped.jpg, img2_3_cropped.jpg: Custom cropped and aligned images
- img1_4_mask.jpg, img2_4_mask.jpg: Validity masks (255=valid, 0=padding)

Usage examples:
  python test_rectify_stereo.py -s ~/data/scene --alpha 1.0 -o test_out 25 10
  python test_rectify_stereo.py -s ~/data/scene --alpha 1.0 --debug 800 500 545 770 -o test_out 25 10
"""

import os
import numpy as np
import cv2
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple


def mark_coordinate_on_image(image_path: str, x: float, y: float, output_path: str, 
                           color: Tuple[int, int, int] = (0, 255, 0), radius: int = 10) -> None:
    """
    Mark a coordinate on an image and save it.
    
    Args:
        image_path: Path to the input image
        x, y: Coordinates to mark
        output_path: Path to save the marked image
        color: BGR color for the marker
        radius: Radius of the circle marker
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert coordinates to integers
    x_int, y_int = int(round(x)), int(round(y))
    
    # Draw circle
    cv2.circle(img, (x_int, y_int), radius, color, -1)
    
    # Draw border
    cv2.circle(img, (x_int, y_int), radius, (0, 0, 0), 2)
    
    # Add text label
    label = f"({x:.1f}, {y:.1f})"
    cv2.putText(img, label, (x_int + radius + 5, y_int), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save image
    cv2.imwrite(output_path, img)


def run_debug_tests(rect_params: Dict[str, Any], 
                   output_dir: Path,
                   x0: float, y0: float, x1: float, y1: float) -> bool:
    """
    Run debug tests with coordinate marking and comprehensive validation.
    
    Args:
        rect_params: Rectification parameters
        output_dir: Output directory for marked images
        x0, y0: Coordinates in first image
        x1, y1: Coordinates in second image
        
    Returns:
        True if all tests pass, False otherwise
    """
    from rectify_stereo import (
        transform_coordinates_to_rectified,
        transform_coordinates_from_rectified
    )
    
    print(f"\nDebug mode: Marking coordinates ({x0}, {y0}) and ({x1}, {y1})")

    img1_name = rect_params['img1_name']
    img2_name = rect_params['img2_name']
    img1_path = rect_params['img1_path']
    img2_path = rect_params['img2_path']
    rect1_path = rect_params['rect1_path']
    rect2_path = rect_params['rect2_path']

    # Mark original coordinates
    img1_marked_path = output_dir / f"{Path(img1_name).stem}_marked_original.jpg"
    img2_marked_path = output_dir / f"{Path(img2_name).stem}_marked_original.jpg"
    
    mark_coordinate_on_image(img1_path, x0, y0, str(img1_marked_path), (0, 255, 0))
    mark_coordinate_on_image(img2_path, x1, y1, str(img2_marked_path), (0, 255, 0))
    print(f"Marked original images saved to: {img1_marked_path}, {img2_marked_path}")
    
    # Transform to rectified coordinates
    coords_rect1, coords_rect2 = transform_coordinates_to_rectified(
        rect_params, (x0, y0), (x1, y1)
    )
    print(f"Rectified coordinates: {coords_rect1}, {coords_rect2}")
    
    # Print custom padding information if available
    if 'custom_padding' in rect_params:
        padding_info = rect_params['custom_padding']
        print(f"Custom padding info:")
        print(f"  Original ROI1: {rect_params.get('roi1_original', 'N/A')}")
        print(f"  Original ROI2: {rect_params.get('roi2_original', 'N/A')}")
        print(f"  Custom ROI1: {rect_params.get('roi1_custom', 'N/A')}")
        print(f"  Custom ROI2: {rect_params.get('roi2_custom', 'N/A')}")
        print(f"  Final image size: {padding_info['final_size']}")
        print(f"  Alignment Y offset: {padding_info['min_y_offset']}")
    
    # Mark rectified coordinates
    img1_rect_marked_path = output_dir / f"{Path(img1_name).stem}_marked_rectified.jpg"
    img2_rect_marked_path = output_dir / f"{Path(img2_name).stem}_marked_rectified.jpg"
    
    mark_coordinate_on_image(rect1_path, coords_rect1[0], coords_rect1[1], 
                           str(img1_rect_marked_path), (255, 0, 0))
    mark_coordinate_on_image(rect2_path, coords_rect2[0], coords_rect2[1], 
                           str(img2_rect_marked_path), (255, 0, 0))
    print(f"Marked rectified images saved to: {img1_rect_marked_path}, {img2_rect_marked_path}")
    
    # Back-transform to original coordinates
    coords_back1, coords_back2 = transform_coordinates_from_rectified(
        rect_params, coords_rect1, coords_rect2
    )
    print(f"Back-transformed coordinates: {coords_back1}, {coords_back2}")
    
    # Run all tests
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE COORDINATE TRANSFORMATION TESTS")
    print("="*60)
    
    test_coords = ((x0, y0), (x1, y1))
    all_success = run_all_tests(rect_params, test_coords, num_vectorized_points=20)
    
    # Mark back-transformed coordinates on original images
    img1_back_marked_path = output_dir / f"{Path(img1_name).stem}_marked_back_transformed.jpg"
    img2_back_marked_path = output_dir / f"{Path(img2_name).stem}_marked_back_transformed.jpg"
    
    mark_coordinate_on_image(img1_path, coords_back1[0], coords_back1[1], 
                           str(img1_back_marked_path), (0, 0, 255))
    mark_coordinate_on_image(img2_path, coords_back2[0], coords_back2[1], 
                           str(img2_back_marked_path), (0, 0, 255))
    print(f"Back-transformed marked images saved to: {img1_back_marked_path}, {img2_back_marked_path}")
    
    # Print information about available debug images
    if 'custom_padding' in rect_params:
        print(f"\nCustom cropping debug images available:")
        print(f"  - Full rectified images: img1_2_rectified_full.jpg, img2_2_rectified_full.jpg")
        print(f"  - Custom cropped images: img1_3_cropped.jpg, img2_3_cropped.jpg")
        print(f"  - Validity masks: img1_4_mask.jpg, img2_4_mask.jpg")
    
    if all_success:
        print("✓ All coordinate transformations working correctly!")
    else:
        print("⚠ Some coordinate transformation tests failed - check the results")
    
    return all_success


def test_coordinate_transformations(rect_params: Dict[str, Any], 
                                  test_coords: Tuple[Tuple[float, float], Tuple[float, float]],
                                  tolerance: float = 1.0) -> bool:
    """
    Unit test for coordinate transformations by checking round-trip accuracy.
    Updated to handle the new pipeline with rotation for vertical alignment.
    
    Args:
        rect_params: Rectification parameters
        test_coords: Tuple of ((x1, y1), (x2, y2)) test coordinates
        tolerance: Maximum allowed error in pixels
        
    Returns:
        True if test passes, False otherwise
    """
    from rectify_stereo import transform_coordinates_to_rectified, transform_coordinates_from_rectified
    
    coords_img1, coords_img2 = test_coords
    
    print("Coordinate transformation test:")
    print(f"  Original coords: {coords_img1}, {coords_img2}")
    rotation_angle = rect_params.get('rotation_angle', 0)
    if rotation_angle != 0:
        print(f"  Note: Testing with {rotation_angle}° rotation applied")
    
    # Forward transformation
    coords_rect1, coords_rect2 = transform_coordinates_to_rectified(rect_params, coords_img1, coords_img2)
    print(f"  Rectified coords: {coords_rect1}, {coords_rect2}")
    
    # Back transformation
    coords_back1, coords_back2 = transform_coordinates_from_rectified(rect_params, coords_rect1, coords_rect2)
    print(f"  Back-transformed coords: {coords_back1}, {coords_back2}")
    
    # Calculate errors
    error1 = np.sqrt((coords_img1[0] - coords_back1[0])**2 + (coords_img1[1] - coords_back1[1])**2)
    error2 = np.sqrt((coords_img2[0] - coords_back2[0])**2 + (coords_img2[1] - coords_back2[1])**2)
    
    print(f"  Errors: {error1:.2f}, {error2:.2f} pixels")
    print(f"  Tolerance: {tolerance} pixels")
    
    success = error1 <= tolerance and error2 <= tolerance
    print(f"  Test {'PASSED' if success else 'FAILED'}")
    
    return success


def test_single_image_coordinate_transformations(rect_params: Dict[str, Any], 
                                               test_coords: Tuple[Tuple[float, float], Tuple[float, float]],
                                               tolerance: float = 1.0) -> bool:
    """
    Unit test for single image coordinate transformations by comparing with the existing function.
    Updated to handle the new pipeline with rotation for vertical alignment.
    
    Args:
        rect_params: Rectification parameters
        test_coords: Tuple of ((x1, y1), (x2, y2)) test coordinates
        tolerance: Maximum allowed error in pixels
        
    Returns:
        True if test passes, False otherwise
    """
    from rectify_stereo import (
        transform_coordinates_to_rectified, 
        transform_coordinates_from_rectified,
        transform_single_image_coordinates_to_rectified,
        transform_single_image_coordinates_from_rectified
    )
    
    coords_img1, coords_img2 = test_coords
    
    print("\nTesting single image coordinate transformations:")
    print(f"  Test coordinates: {coords_img1}, {coords_img2}")
    rotation_angle = rect_params.get('rotation_angle', 0)
    if rotation_angle != 0:
        print(f"  Note: Testing with {rotation_angle}° rotation applied")
    
    # Test forward transformation comparison
    print(f"  Forward transformation comparison:")
    
    # Single image functions
    coords_rect1_single = transform_single_image_coordinates_to_rectified(rect_params, coords_img1, 1)
    coords_rect2_single = transform_single_image_coordinates_to_rectified(rect_params, coords_img2, 2)
    
    # Existing dual image function
    coords_rect1_existing, coords_rect2_existing = transform_coordinates_to_rectified(rect_params, coords_img1, coords_img2)
    
    # Compare results
    error1_forward = np.sqrt((coords_rect1_single[0] - coords_rect1_existing[0])**2 + 
                           (coords_rect1_single[1] - coords_rect1_existing[1])**2)
    error2_forward = np.sqrt((coords_rect2_single[0] - coords_rect2_existing[0])**2 + 
                           (coords_rect2_single[1] - coords_rect2_existing[1])**2)
    
    print(f"    Image 1 - Single: {coords_rect1_single}, Existing: {coords_rect1_existing}, Error: {error1_forward:.6f}")
    print(f"    Image 2 - Single: {coords_rect2_single}, Existing: {coords_rect2_existing}, Error: {error2_forward:.6f}")
    
    # Test reverse transformation comparison
    print(f"  Reverse transformation comparison:")
    
    # Single image functions
    coords_back1_single = transform_single_image_coordinates_from_rectified(rect_params, coords_rect1_single, 1)
    coords_back2_single = transform_single_image_coordinates_from_rectified(rect_params, coords_rect2_single, 2)
    
    # Existing dual image function
    coords_back1_existing, coords_back2_existing = transform_coordinates_from_rectified(rect_params, coords_rect1_single, coords_rect2_single)
    
    # Compare results
    error1_reverse = np.sqrt((coords_back1_single[0] - coords_back1_existing[0])**2 + 
                           (coords_back1_single[1] - coords_back1_existing[1])**2)
    error2_reverse = np.sqrt((coords_back2_single[0] - coords_back2_existing[0])**2 + 
                           (coords_back2_single[1] - coords_back2_existing[1])**2)
    
    print(f"    Image 1 - Single: {coords_back1_single}, Existing: {coords_back1_existing}, Error: {error1_reverse:.6f}")
    print(f"    Image 2 - Single: {coords_back2_single}, Existing: {coords_back2_existing}, Error: {error2_reverse:.6f}")
    
    # Test round-trip accuracy
    print(f"  Round-trip accuracy:")
    error1_roundtrip = np.sqrt((coords_img1[0] - coords_back1_single[0])**2 + (coords_img1[1] - coords_back1_single[1])**2)
    error2_roundtrip = np.sqrt((coords_img2[0] - coords_back2_single[0])**2 + (coords_img2[1] - coords_back2_single[1])**2)
    
    print(f"    Image 1 - Original: {coords_img1}, Back-transformed: {coords_back1_single}, Error: {error1_roundtrip:.6f}")
    print(f"    Image 2 - Original: {coords_img2}, Back-transformed: {coords_back2_single}, Error: {error2_roundtrip:.6f}")
    
    # Check if all tests pass
    forward_ok = error1_forward <= tolerance and error2_forward <= tolerance
    reverse_ok = error1_reverse <= tolerance and error2_reverse <= tolerance
    roundtrip_ok = error1_roundtrip <= tolerance and error2_roundtrip <= tolerance
    
    success = forward_ok and reverse_ok and roundtrip_ok
    
    print(f"  Tolerance: {tolerance} pixels")
    print(f"  Forward comparison: {'PASS' if forward_ok else 'FAIL'}")
    print(f"  Reverse comparison: {'PASS' if reverse_ok else 'FAIL'}")
    print(f"  Round-trip accuracy: {'PASS' if roundtrip_ok else 'FAIL'}")
    print(f"  Overall test: {'PASSED' if success else 'FAILED'}")
    
    return success


def test_vectorized_coordinate_transformations(rect_params: Dict[str, Any], 
                                             num_test_points: int = 10,
                                             tolerance: float = 1e-3) -> bool:
    """
    Comprehensive test for vectorized coordinate transformations by comparing with single coordinate versions.
    Updated to handle the new pipeline with rotation for vertical alignment.
    
    Args:
        rect_params: Rectification parameters
        num_test_points: Number of random test points to generate
        tolerance: Maximum allowed error in pixels
        
    Returns:
        True if all tests pass, False otherwise
    """
    from rectify_stereo import (
        transform_coordinates_to_rectified, 
        transform_coordinates_from_rectified,
        transform_single_image_coordinates_to_rectified,
        transform_coordinates_to_rectified_vectorized,
        transform_coordinates_from_rectified_vectorized,
        transform_single_image_coordinates_to_rectified_vectorized
    )
    
    print(f"\nTesting vectorized coordinate transformations with {num_test_points} random points:")
    rotation_angle = rect_params.get('rotation_angle', 0)
    if rotation_angle != 0:
        print(f"  Note: Testing with {rotation_angle}° rotation applied")
    
    # Generate random test coordinates within actual image bounds
    np.random.seed(42)  # For reproducible results
    
    # Get actual image dimensions
    image_size = rect_params.get('image_size', (2000, 1500))  # Default fallback
    width, height = image_size
    print(f"  Using image dimensions: {width}x{height}")
    
    # Generate coordinates within valid image bounds with some margin
    margin = 50  # Pixels from edge
    max_x, max_y = width - margin, height - margin
    print(f"  Coordinate range: [{margin}, {margin}] to [{max_x}, {max_y}]")
    
    coords_img1 = np.random.uniform([margin, margin], [max_x, max_y], (num_test_points, 2))
    coords_img2 = np.random.uniform([margin, margin], [max_x, max_y], (num_test_points, 2))
    
    print(f"  Generated test coordinates:")
    print(f"    Image 1: {coords_img1[:3]}... (showing first 3)")
    print(f"    Image 2: {coords_img2[:3]}... (showing first 3)")
    
    # Test 1: Dual image transformations
    print(f"\n  Test 1: Dual image transformations")
    
    # Vectorized version
    coords_rect1_vec, coords_rect2_vec = transform_coordinates_to_rectified_vectorized(
        rect_params, coords_img1, coords_img2
    )
    
    # Single coordinate version (loop)
    coords_rect1_single = []
    coords_rect2_single = []
    for i in range(num_test_points):
        rect1, rect2 = transform_coordinates_to_rectified(
            rect_params, tuple(coords_img1[i]), tuple(coords_img2[i])
        )
        coords_rect1_single.append(rect1)
        coords_rect2_single.append(rect2)
    
    coords_rect1_single = np.array(coords_rect1_single)
    coords_rect2_single = np.array(coords_rect2_single)
    
    # Compare results
    error1_forward = np.linalg.norm(coords_rect1_vec - coords_rect1_single, axis=1)
    error2_forward = np.linalg.norm(coords_rect2_vec - coords_rect2_single, axis=1)
    
    max_error1_forward = np.max(error1_forward)
    max_error2_forward = np.max(error2_forward)
    
    print(f"    Forward transformation - Max errors: Image1={max_error1_forward:.8f}, Image2={max_error2_forward:.8f}")
    
    # Test 2: Reverse transformations
    coords_back1_vec, coords_back2_vec = transform_coordinates_from_rectified_vectorized(
        rect_params, coords_rect1_vec, coords_rect2_vec
    )
    
    coords_back1_single = []
    coords_back2_single = []
    for i in range(num_test_points):
        back1, back2 = transform_coordinates_from_rectified(
            rect_params, tuple(coords_rect1_vec[i]), tuple(coords_rect2_vec[i])
        )
        coords_back1_single.append(back1)
        coords_back2_single.append(back2)
    
    coords_back1_single = np.array(coords_back1_single)
    coords_back2_single = np.array(coords_back2_single)
    
    error1_reverse = np.linalg.norm(coords_back1_vec - coords_back1_single, axis=1)
    error2_reverse = np.linalg.norm(coords_back2_vec - coords_back2_single, axis=1)
    
    max_error1_reverse = np.max(error1_reverse)
    max_error2_reverse = np.max(error2_reverse)
    
    print(f"    Reverse transformation - Max errors: Image1={max_error1_reverse:.8f}, Image2={max_error2_reverse:.8f}")
    
    # Test 3: Single image transformations
    print(f"\n  Test 2: Single image transformations")
    
    # Test image 1
    coords_rect1_single_img = transform_single_image_coordinates_to_rectified_vectorized(
        rect_params, coords_img1, 1
    )
    
    coords_rect1_single_img_loop = []
    for i in range(num_test_points):
        rect = transform_single_image_coordinates_to_rectified(
            rect_params, tuple(coords_img1[i]), 1
        )
        coords_rect1_single_img_loop.append(rect)
    coords_rect1_single_img_loop = np.array(coords_rect1_single_img_loop)
    
    error1_single_forward = np.linalg.norm(coords_rect1_single_img - coords_rect1_single_img_loop, axis=1)
    max_error1_single_forward = np.max(error1_single_forward)
    
    # Test image 2
    coords_rect2_single_img = transform_single_image_coordinates_to_rectified_vectorized(
        rect_params, coords_img2, 2
    )
    
    coords_rect2_single_img_loop = []
    for i in range(num_test_points):
        rect = transform_single_image_coordinates_to_rectified(
            rect_params, tuple(coords_img2[i]), 2
        )
        coords_rect2_single_img_loop.append(rect)
    coords_rect2_single_img_loop = np.array(coords_rect2_single_img_loop)
    
    error2_single_forward = np.linalg.norm(coords_rect2_single_img - coords_rect2_single_img_loop, axis=1)
    max_error2_single_forward = np.max(error2_single_forward)
    
    print(f"    Single image forward - Max errors: Image1={max_error1_single_forward:.8f}, Image2={max_error2_single_forward:.8f}")
    
    # Test 4: Round-trip accuracy
    print(f"\n  Test 3: Round-trip accuracy")
    
    error1_roundtrip = np.linalg.norm(coords_img1 - coords_back1_vec, axis=1)
    error2_roundtrip = np.linalg.norm(coords_img2 - coords_back2_vec, axis=1)
    
    max_error1_roundtrip = np.max(error1_roundtrip)
    max_error2_roundtrip = np.max(error2_roundtrip)
    
    print(f"    Round-trip - Max errors: Image1={max_error1_roundtrip:.8f}, Image2={max_error2_roundtrip:.8f}")
    
    # Check if all tests pass
    forward_ok = max_error1_forward <= tolerance and max_error2_forward <= tolerance
    reverse_ok = max_error1_reverse <= tolerance and max_error2_reverse <= tolerance
    single_forward_ok = max_error1_single_forward <= tolerance and max_error2_single_forward <= tolerance
    roundtrip_ok = max_error1_roundtrip <= tolerance and max_error2_roundtrip <= tolerance
    
    success = forward_ok and reverse_ok and single_forward_ok and roundtrip_ok
    
    print(f"\n  Results:")
    print(f"    Tolerance: {tolerance}")
    print(f"    Dual forward: {'PASS' if forward_ok else 'FAIL'}")
    print(f"    Dual reverse: {'PASS' if reverse_ok else 'FAIL'}")
    print(f"    Single forward: {'PASS' if single_forward_ok else 'FAIL'}")
    print(f"    Round-trip: {'PASS' if roundtrip_ok else 'FAIL'}")
    print(f"    Overall: {'PASSED' if success else 'FAILED'}")
    
    return success


def run_all_tests(rect_params: Dict[str, Any], 
                  test_coords: Tuple[Tuple[float, float], Tuple[float, float]],
                  num_vectorized_points: int = 20) -> bool:
    """
    Run all coordinate transformation tests.
    
    Args:
        rect_params: Rectification parameters
        test_coords: Tuple of ((x1, y1), (x2, y2)) test coordinates
        num_vectorized_points: Number of points for vectorized tests
        
    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("RUNNING ALL COORDINATE TRANSFORMATION TESTS")
    print("=" * 60)
    
    # Test 1: Basic coordinate transformations
    print("\n1. Basic coordinate transformation test:")
    test1_success = test_coordinate_transformations(rect_params, test_coords, tolerance=10.0)
    
    # Test 2: Single image coordinate transformations
    print("\n2. Single image coordinate transformation test:")
    test2_success = test_single_image_coordinate_transformations(rect_params, test_coords, tolerance=1.0)
    
    # Test 3: Vectorized coordinate transformations
    print("\n3. Vectorized coordinate transformation test:")
    test3_success = test_vectorized_coordinate_transformations(rect_params, num_vectorized_points, tolerance=1.0)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic transformations: {'PASS' if test1_success else 'FAIL'}")
    print(f"Single image transformations: {'PASS' if test2_success else 'FAIL'}")
    print(f"Vectorized transformations: {'PASS' if test3_success else 'FAIL'}")
    
    all_success = test1_success and test2_success and test3_success
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_success else 'SOME TESTS FAILED'}")
    print("=" * 60)
    
    return all_success


def test_rotation_scenarios() -> bool:
    """
    Test coordinate transformations for different rotation scenarios (0°, 90°, 180°, 270°).
    
    Returns:
        True if all rotation tests pass, False otherwise
    """
    print("\n" + "="*60)
    print("TESTING DIFFERENT ROTATION SCENARIOS")
    print("="*60)
    
    # Create synthetic rectification parameters for testing
    image_size = (6000, 4000)  # width x height
    
    # Synthetic camera parameters
    K = np.array([
        [4000.0, 0, 3000.0],
        [0, 4000.0, 2000.0],
        [0, 0, 1]
    ])
    
    # Synthetic rectification matrices (identity for simplicity)
    R_rect = np.eye(3)
    P = np.array([
        [4000.0, 0, 3000.0, 0],
        [0, 4000.0, 2000.0, 0],
        [0, 0, 1, 0]
    ])
    
    # Test coordinates
    test_coords = [(1500.0, 1000.0), (4500.0, 3000.0)]
    tolerance = 2.0  # Allow 2 pixel tolerance for synthetic tests
    
    all_tests_passed = True
    
    for rotation_angle in [0, 90, 180, 270]:
        print(f"\nTesting {rotation_angle}° rotation:")
        
        # Create rotated parameters
        if rotation_angle == 0:
            K_rotated = K.copy()
            image_size_rotated = image_size
        elif rotation_angle == 90:
            # 90-degree rotation: swap fx↔fy, cx↔cy, adjust for new dimensions
            K_rotated = np.array([
                [K[1,1], 0, K[1,2]],  # fy, 0, cy
                [0, K[0,0], image_size[1] - K[0,2]],  # 0, fx, height-cx
                [0, 0, 1]
            ])
            image_size_rotated = (image_size[1], image_size[0])  # Swap width and height
        elif rotation_angle == 180:
            # 180-degree rotation: keep fx, fy, adjust cx, cy
            K_rotated = np.array([
                [K[0,0], 0, image_size[0] - K[0,2]],  # fx, 0, width-cx
                [0, K[1,1], image_size[1] - K[1,2]],  # 0, fy, height-cy
                [0, 0, 1]
            ])
            image_size_rotated = image_size  # Keep same dimensions
        elif rotation_angle == 270:
            # 270-degree rotation: swap fx↔fy, adjust for new dimensions
            K_rotated = np.array([
                [K[1,1], 0, image_size[0] - K[1,2]],  # fy, 0, width-cy
                [0, K[0,0], K[0,2]],  # 0, fx, cx
                [0, 0, 1]
            ])
            image_size_rotated = (image_size[1], image_size[0])  # Swap width and height
        
        # Create synthetic rectification parameters
        rect_params = {
            'K1': K.tolist(),
            'K2': K.tolist(),
            'K1_rotated': K_rotated.tolist(),
            'K2_rotated': K_rotated.tolist(),
            'dist1': [0, 0, 0, 0, 0],  # No distortion
            'dist2': [0, 0, 0, 0, 0],  # No distortion
            'R1_rect': R_rect.tolist(),
            'R2_rect': R_rect.tolist(),
            'P1': P.tolist(),
            'P2': P.tolist(),
            'image_size': image_size,
            'image_size_rotated': image_size_rotated,
            'rotation_angle': rotation_angle,
            'is_vertical': rotation_angle != 0,  # For backward compatibility
            'roi1': (0, 0, 0, 0),
            'roi2': (0, 0, 0, 0)
        }
        
        # Test coordinate transformations for this rotation
        success = test_coordinate_transformations(rect_params, test_coords, tolerance)
        
        print(f"  {rotation_angle}° rotation test: {'PASSED' if success else 'FAILED'}")
        
        if not success:
            all_tests_passed = False
    
    print(f"\n{'='*60}")
    print(f"ROTATION SCENARIO TESTS: {'ALL PASSED' if all_tests_passed else 'SOME FAILED'}")
    print(f"{'='*60}")
    
    return all_tests_passed


def test_coordinate_rotation_functions() -> bool:
    """
    Test the coordinate rotation helper functions directly.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "="*60)
    print("TESTING COORDINATE ROTATION HELPER FUNCTIONS")
    print("="*60)
    
    from rectify_stereo import _apply_rotation_to_coordinates, _apply_inverse_rotation_to_coordinates
    
    # Test parameters
    image_size = (6000, 4000)  # width x height
    test_coords = [(1500.0, 1000.0), (4500.0, 3000.0), (3000.0, 2000.0)]
    tolerance = 1e-10  # Very strict tolerance for exact coordinate transformations
    
    all_tests_passed = True
    
    for rotation_angle in [0, 90, 180, 270]:
        print(f"\nTesting {rotation_angle}° coordinate rotation:")
        
        for i, (x, y) in enumerate(test_coords):
            # Apply rotation
            x_rot, y_rot = _apply_rotation_to_coordinates((x, y), rotation_angle, image_size)
            
            # Apply inverse rotation
            x_back, y_back = _apply_inverse_rotation_to_coordinates((x_rot, y_rot), rotation_angle, image_size)
            
            # Check if we get back the original coordinates
            error = np.sqrt((x - x_back)**2 + (y - y_back)**2)
            
            print(f"  Coord {i+1}: ({x}, {y}) -> ({x_rot:.1f}, {y_rot:.1f}) -> ({x_back:.1f}, {y_back:.1f}), Error: {error:.2e}")
            
            if error > tolerance:
                print(f"    ERROR: Round-trip error {error:.2e} exceeds tolerance {tolerance}")
                all_tests_passed = False
    
    print(f"\n{'='*60}")
    print(f"COORDINATE ROTATION TESTS: {'ALL PASSED' if all_tests_passed else 'SOME FAILED'}")
    print(f"{'='*60}")
    
    return all_tests_passed


def main():
    """Main function for running tests on COLMAP reconstructions."""
    parser = argparse.ArgumentParser(description='Test stereo rectification coordinate transformations')
    parser.add_argument('-s', '--scene_folder', required=False, 
                       help='Path to scene folder containing sparse/ and images/')
    parser.add_argument('-o', '--out_folder', required=False,
                       help='Output folder name (will be created under scene_folder)')
    parser.add_argument('--debug', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'),
                       help='Debug mode: mark coordinates (x0,y0) in first image and (x1,y1) in second image')
    parser.add_argument('--test-rotations', action='store_true',
                       help='Run unit tests for rotation scenarios only (no COLMAP data needed)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Alpha parameter for stereo rectification (0.0=more crop, 1.0=less crop, default: 1.0)')
    parser.add_argument('img_id1', type=int, nargs='?', help='First image ID')
    parser.add_argument('img_id2', type=int, nargs='?', help='Second image ID')
    
    args = parser.parse_args()
    
    # Validate alpha parameter
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("Alpha parameter must be between 0.0 and 1.0")
    
    # Handle rotation-only tests
    if args.test_rotations:
        print("Running rotation scenario tests...")
        
        # Test coordinate rotation functions
        rotation_func_success = test_coordinate_rotation_functions()
        
        # Test rotation scenarios
        rotation_scenario_success = test_rotation_scenarios()
        
        # Summary
        print("\n" + "="*60)
        print("ROTATION TEST SUMMARY")
        print("="*60)
        print(f"Coordinate rotation functions: {'PASS' if rotation_func_success else 'FAIL'}")
        print(f"Rotation scenarios: {'PASS' if rotation_scenario_success else 'FAIL'}")
        
        overall_success = rotation_func_success and rotation_scenario_success
        print(f"\nOverall result: {'ALL ROTATION TESTS PASSED' if overall_success else 'SOME ROTATION TESTS FAILED'}")
        print("="*60)
        
        if overall_success:
            print("\n🎉 All rotation tests passed! The new rotation logic is working correctly.")
        else:
            print("\n❌ Some rotation tests failed. Please check the output above for details.")
            sys.exit(1)
        
        return
    
    # Validate arguments for COLMAP tests
    if not args.scene_folder or not args.out_folder:
        parser.error("--scene_folder and --out_folder are required when not using --test-rotations")
    
    if args.img_id1 is None or args.img_id2 is None:
        parser.error("img_id1 and img_id2 are required when not using --test-rotations")
    
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
        from colmap_utils import ColmapReconstruction
        reconstruction = ColmapReconstruction(str(sparse_path))
        print(f"Loaded reconstruction with {len(reconstruction.reconstruction.images)} images and {len(reconstruction.reconstruction.points3D)} 3D points")
    except Exception as e:
        print(f"Error loading COLMAP reconstruction: {e}")
        sys.exit(1)
    
    # Validate image IDs
    if not reconstruction.has_image(args.img_id1):
        print(f"Error: Image ID {args.img_id1} not found in reconstruction")
        sys.exit(1)
    
    if not reconstruction.has_image(args.img_id2):
        print(f"Error: Image ID {args.img_id2} not found in reconstruction")
        sys.exit(1)
    
    # Get image names
    # Create output directory
    output_dir = scene_folder / args.out_folder
    output_dir.mkdir(exist_ok=True)
    
    # Compute stereo rectification parameters
    print("Computing stereo rectification parameters...")
    try:
        from rectify_stereo import initalize_rectification, rectify_images
        rect_params = initalize_rectification(reconstruction, args.img_id1, args.img_id2, images_path, output_dir, alpha=args.alpha)
        
        # Print information about the rectification
        print(f"Rectification type: {rect_params['type']}")
        print(f"Left image: {rect_params['left']}")
        print(f"Right image: {rect_params['right']}")
        print(f"Alpha parameter: {args.alpha}")
        if args.alpha == 1.0:
            print("Note: Using custom cropping with alignment padding to maintain rectification property")
            if 'custom_padding' in rect_params:
                padding_info = rect_params['custom_padding']
                print(f"Custom padding applied - final size: {padding_info['final_size']}")
        if rect_params.get('is_vertical', False):
            print("Note: Images were vertically aligned and rotated 90 degrees before rectification")
    except Exception as e:
        print(f"Error computing rectification: {e}")
        sys.exit(1)

    # Rectify images
    print("Rectifying images...")
    try:
        debug_dir = output_dir / 'debug_stages'
        rect1_img, rect2_img = rectify_images(rect_params, debug_dir)
        cv2.imwrite(rect_params['rect1_path'], rect1_img)
        cv2.imwrite(rect_params['rect2_path'], rect2_img)
    except Exception as e:
        print(f"Error rectifying images: {e}")
        sys.exit(1)
    
    if args.debug:
        # Debug mode: run tests with coordinate marking
        x0, y0, x1, y1 = args.debug
        
        # Run debug tests
        success = run_debug_tests(
            rect_params, 
            output_dir,
            x0, y0, x1, y1
        )
        
        if success:
            print("\n🎉 All tests passed! The coordinate transformation functions are working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the output above for details.")
            sys.exit(1)
    else:
        # Normal mode: run tests without coordinate marking
        print("\nRunning coordinate transformation tests...")
        print(f"Using alpha={args.alpha} for rectification")
        if args.alpha == 1.0 and 'custom_padding' in rect_params:
            print("Custom cropping and padding applied - coordinate transformations adjusted accordingly")
        
        # Use default test coordinates
        test_coords = ((1000.0, 1000.0), (2000.0, 1500.0))
        
        success = run_all_tests(rect_params, test_coords, num_vectorized_points=20)
        
        if success:
            print("\n🎉 All tests passed! The coordinate transformation functions are working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the output above for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()
