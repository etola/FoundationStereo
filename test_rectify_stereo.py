#!/usr/bin/env python3
"""
Test script for stereo rectification coordinate transformations.

This script contains comprehensive tests for all coordinate transformation functions
in the rectify_stereo.py module, including debug functionality for marking coordinates
on images.
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
    
    # Generate random test coordinates within reasonable image bounds
    np.random.seed(42)  # For reproducible results
    coords_img1 = np.random.uniform(100, 5000, (num_test_points, 2))
    coords_img2 = np.random.uniform(100, 5000, (num_test_points, 2))
    
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
    test3_success = test_vectorized_coordinate_transformations(rect_params, num_vectorized_points, tolerance=1e-3)
    
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


def main():
    """Main function for running tests on COLMAP reconstructions."""
    parser = argparse.ArgumentParser(description='Test stereo rectification coordinate transformations')
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
        rect_params = initalize_rectification(reconstruction, args.img_id1, args.img_id2)
    except Exception as e:
        print(f"Error computing rectification: {e}")
        sys.exit(1)

    img1_name = rect_params['img1_name']
    img2_name = rect_params['img2_name']

    rect_params['img1_path'] = str(images_path / img1_name)
    rect_params['img2_path'] = str(images_path / img2_name)

    rect_params['rect1_path'] = os.path.join(output_dir, f"{Path(img1_name).stem}_rectified.jpg")
    rect_params['rect2_path'] = os.path.join(output_dir, f"{Path(img2_name).stem}_rectified.jpg")

    # Rectify images
    print("Rectifying images...")
    try:
        img1_path = rect_params['img1_path']
        img2_path = rect_params['img2_path']
        rect1_img, rect2_img = rectify_images(rect_params)
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
