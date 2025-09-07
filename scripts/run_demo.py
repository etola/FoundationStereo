# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import glob
import re
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


def process_stereo_pair(left_file, right_file, intrinsic_file, out_dir, model, args):
  """Process a single stereo pair and save results to out_dir"""
  os.makedirs(out_dir, exist_ok=True)
  img0 = imageio.imread(left_file)
  img1 = imageio.imread(right_file)
  scale = args.scale
  assert scale<=1, "scale must be <=1"
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  logging.info(f"Processing {left_file} and {right_file}, img0: {img0.shape}")

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  with torch.cuda.amp.autocast(True):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W)
  vis = vis_disparity(disp)
  vis = np.concatenate([img0_ori, vis], axis=1)
  imageio.imwrite(f'{out_dir}/vis.png', vis)
  logging.info(f"Disparity visualization saved to {out_dir}/vis.png")

  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  if args.get_pc:
    with open(intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{out_dir}/cloud.ply', pcd)
    logging.info(f"Point cloud saved to {out_dir}/cloud.ply")

    if args.denoise_cloud:
      logging.info("[Optional step] denoise point cloud...")
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{out_dir}/cloud_denoise.ply', inlier_cloud)
      pcd = inlier_cloud

    return pcd
  return None


def discover_pairs(pairs_dir):
  """Discover all pair_XX folders and validate required files"""
  pair_folders = []
  if not os.path.exists(pairs_dir):
    logging.error(f"Pairs directory does not exist: {pairs_dir}")
    return pair_folders
  
  # Find all pair_XX folders
  pattern = os.path.join(pairs_dir, "pair_*")
  candidates = glob.glob(pattern)
  
  for folder in candidates:
    if not os.path.isdir(folder):
      continue
    
    folder_name = os.path.basename(folder)
    # Check if folder name matches pair_XX pattern
    if not re.match(r'pair_\d+', folder_name):
      continue
    
    # Check if required files exist
    left_file = os.path.join(folder, "left.jpg")
    right_file = os.path.join(folder, "right.jpg")
    intrinsic_file = os.path.join(folder, "intrinsics.txt")
    
    missing_files = []
    if not os.path.exists(left_file):
      missing_files.append("left.jpg")
    if not os.path.exists(right_file):
      missing_files.append("right.jpg")
    if not os.path.exists(intrinsic_file):
      missing_files.append("intrinsics.txt")
    
    if missing_files:
      logging.warning(f"Skipping {folder}: missing files {missing_files}")
      continue
    
    pair_folders.append({
      'folder': folder,
      'name': folder_name,
      'left_file': left_file,
      'right_file': right_file,
      'intrinsic_file': intrinsic_file
    })
  
  # Sort by pair number
  pair_folders.sort(key=lambda x: int(re.search(r'pair_(\d+)', x['name']).group(1)))
  logging.info(f"Discovered {len(pair_folders)} valid pair folders")
  return pair_folders


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--process_pairs', action='store_true', help='process multiple stereo pairs from pair_XX folders')
  parser.add_argument('--pairs_dir', type=str, help='directory containing pair_XX folders (required when --process_pairs is used)')
  args = parser.parse_args()

  # Validate arguments for pairs processing
  if args.process_pairs:
    if not args.pairs_dir:
      parser.error("--pairs_dir is required when --process_pairs is used")
    if not os.path.exists(args.pairs_dir):
      parser.error(f"Pairs directory does not exist: {args.pairs_dir}")

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir, weights_only=False)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  if args.process_pairs:
    # Process multiple pairs
    pairs = discover_pairs(args.pairs_dir)
    if not pairs:
      logging.error("No valid pair folders found")
      sys.exit(1)
    
    logging.info(f"Processing {len(pairs)} stereo pairs...")
    
    for i, pair_info in enumerate(pairs):
      logging.info(f"\n=== Processing {pair_info['name']} ({i+1}/{len(pairs)}) ===")
      
      # Save results in each pair folder
      out_dir = pair_info['folder']
      
      try:
        pcd = process_stereo_pair(
          pair_info['left_file'],
          pair_info['right_file'], 
          pair_info['intrinsic_file'],
          out_dir,
          model,
          args
        )
        logging.info(f"Successfully processed {pair_info['name']}")
      except Exception as e:
        logging.error(f"Failed to process {pair_info['name']}: {str(e)}")
        continue
    
    logging.info(f"\nCompleted processing {len(pairs)} pairs")
    
  else:
    # Process single pair (original behavior)
    os.makedirs(args.out_dir, exist_ok=True)
    
    pcd = process_stereo_pair(
      args.left_file,
      args.right_file,
      args.intrinsic_file,
      args.out_dir,
      model,
      args
    )
    
    # Show visualization for single pair processing
    if pcd is not None:
      logging.info("Visualizing point cloud. Press ESC to exit.")
      vis = o3d.visualization.Visualizer()
      vis.create_window()
      vis.add_geometry(pcd)
      vis.get_render_option().point_size = 1.0
      vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
      vis.run()
      vis.destroy_window()

