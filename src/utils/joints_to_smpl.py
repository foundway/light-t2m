"""
Convert joint positions to SMPL format (pkl) compatible with SMPL-to-FBX.
Mimics visualize/utils/simplify_loc2rot.py approach.
"""

import os
import numpy as np
import torch
import pickle
from typing import Optional, Tuple, Dict
import sys

log = None

def _get_logger():
    global log
    if log is None:
        from src.utils import RankedLogger
        log = RankedLogger(__name__, rank_zero_only=True)
    return log


def joints_to_smpl_pkl(
    joints: np.ndarray,
    output_pkl_path: str,
    device_id: int = 0,
    cuda: bool = True,
    num_smplify_iters: int = 150,
    init_params: Optional[Dict] = None
) -> bool:
    """Convert joint positions to SMPL format pkl file compatible with SMPL-to-FBX.
    
    Args:
        joints: Joint positions array of shape [num_frames, 22, 3] (HumanML3D format)
        output_pkl_path: Path to save output pkl file
        device_id: Device ID for computation (default: 0)
        cuda: Whether to use CUDA (default: True)
        num_smplify_iters: Number of SMPLify optimization iterations (default: 150)
        init_params: Optional dict with 'betas', 'pose', 'cam' for initialization
        
    Returns:
        True if successful, False otherwise
        
    The output pkl file contains a dictionary with:
        - 'smpl_poses': (N, 72) ndarray - 24 joints * 3 axis-angle values
        - 'smpl_trans': (N, 3) ndarray - root translation (Pelvis position)
    """
    logger = _get_logger()
    
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from visualize.utils.simplify_loc2rot import joints2smpl
    except ImportError as e:
        logger.error(f"Failed to import joints2smpl: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"Project root: {project_root}")
        logger.error(f"sys.path: {sys.path[:5]}")
        logger.error("Make sure visualize/utils/simplify_loc2rot.py is accessible")
        return False
    except Exception as e:
        logger.error(f"Failed to import joints2smpl: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    try:
        num_frames = joints.shape[0]
        logger.info(f"Converting {num_frames} frames of joint positions to SMPL format using SMPLify")
        
        j2s = joints2smpl(
            num_frames=num_frames,
            device_id=device_id,
            cuda=cuda,
            num_smplify_iters=num_smplify_iters
        )
        
        joints_tensor = torch.from_numpy(joints).float()
        if cuda and torch.cuda.is_available():
            joints_tensor = joints_tensor.to(f"cuda:{device_id}")
        
        logger.info(f"Running SMPLify optimization for all {num_frames} frames (this may take a while...)")
        
        keypoints_3d = torch.Tensor(joints).to(j2s.device).float()
        
        if init_params is None:
            pred_betas = j2s.init_mean_shape
            pred_pose = j2s.init_mean_pose
            pred_cam_t = j2s.cam_trans_zero
        else:
            pred_betas = init_params['betas'].to(j2s.device) if isinstance(init_params['betas'], torch.Tensor) else torch.from_numpy(init_params['betas']).to(j2s.device)
            pred_pose = init_params['pose'].to(j2s.device) if isinstance(init_params['pose'], torch.Tensor) else torch.from_numpy(init_params['pose']).to(j2s.device)
            pred_cam_t = init_params['cam'].to(j2s.device) if isinstance(init_params['cam'], torch.Tensor) else torch.from_numpy(init_params['cam']).to(j2s.device)
        
        confidence_input = torch.ones(j2s.num_joints).to(j2s.device)
        
        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = j2s.smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input,
        )
        
        new_opt_pose_np = new_opt_pose.cpu().numpy()
        
        logger.info(f"SMPLify output pose shape: {new_opt_pose_np.shape}")
        
        if new_opt_pose_np.shape == (num_frames, 72):
            smpl_poses_output = new_opt_pose_np
        elif len(new_opt_pose_np.shape) == 2 and new_opt_pose_np.shape[1] == 72:
            smpl_poses_output = new_opt_pose_np
        else:
            logger.error(f"Unexpected pose shape from SMPLify: {new_opt_pose_np.shape}, expected ({num_frames}, 72)")
            return False
        
        smpl_trans_output = joints[:, 0].copy()
        
        output_dict = {
            'smpl_poses': smpl_poses_output,
            'smpl_trans': smpl_trans_output
        }
        
        os.makedirs(os.path.dirname(output_pkl_path) if os.path.dirname(output_pkl_path) else '.', exist_ok=True)
        
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(output_dict, f)
        
        logger.info(f"Saved SMPL format pkl: {output_pkl_path}")
        logger.info(f"  - smpl_poses: {smpl_poses_output.shape} (should be ({len(smpl_trans_output)}, 72))")
        logger.info(f"  - smpl_trans: {smpl_trans_output.shape} (should be ({len(smpl_trans_output)}, 3))")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting joints to SMPL format: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
