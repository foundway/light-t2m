"""
Convert SMPL format pkl file to FBX using SMPL-to-FBX approach.
Based on https://github.com/softcat477/SMPL-to-FBX
"""

import os
import numpy as np
import pickle
from typing import Optional
from scipy.spatial.transform import Rotation

log = None

def _get_logger():
    global log
    if log is None:
        from src.utils import RankedLogger
        log = RankedLogger(__name__, rank_zero_only=True)
    return log


# SMPL joint order from SMPL-to-FBX repository
SMPL_JOINT_ORDER = [
    'Pelvis',      # 0
    'L hip',       # 1
    'R hip',       # 2
    'Spine1',      # 3
    'L_Knee',      # 4
    'R_Knee',      # 5
    'Spine2',      # 6
    'L_Ankle',     # 7
    'R_Ankle',     # 8
    'Spine3',      # 9
    'L_Foot',      # 10
    'R_Foot',      # 11
    'Neck',        # 12
    'L_Collar',    # 13
    'R_Collar',    # 14
    'Head',        # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',     # 18
    'R_Elbow',     # 19
    'L_Wrist',     # 20
    'R_Wrist',     # 21
    'L_Hand',      # 22
    'R_Hand',      # 23
]

# SMPL parent indices (from SMPL kinematic tree)
SMPL_PARENT_INDICES = [
    None,  # 0: Pelvis (root)
    0,     # 1: L hip
    0,     # 2: R hip
    0,     # 3: Spine1
    1,     # 4: L_Knee
    2,     # 5: R_Knee
    3,     # 6: Spine2
    4,     # 7: L_Ankle
    5,     # 8: R_Ankle
    6,     # 9: Spine3
    7,     # 10: L_Foot
    8,     # 11: R_Foot
    9,     # 12: Neck
    9,     # 13: L_Collar
    9,     # 14: R_Collar
    12,    # 15: Head
    13,    # 16: L_Shoulder
    14,    # 17: R_Shoulder
    16,    # 18: L_Elbow
    17,    # 19: R_Elbow
    18,    # 20: L_Wrist
    19,    # 21: R_Wrist
    20,    # 22: L_Hand
    21,    # 23: R_Hand
]


def smpl_pkl_to_fbx(
    pkl_path: str,
    output_path: str,
    base_fbx_path: str,
    fps: float = 20.0
) -> bool:
    """Convert SMPL format pkl file to FBX.
    
    Args:
        pkl_path: Path to input pkl file with 'smpl_poses' and 'smpl_trans'
        output_path: Path to output FBX file
        base_fbx_path: Path to base SMPL FBX file (template)
        fps: Frames per second for animation
        
    Returns:
        True if successful, False otherwise
    """
    logger = _get_logger()
    
    try:
        from FbxCommon import (
            FbxManager, FbxExporter, FbxImporter, FbxScene, FbxNode,
            FbxSkeleton, FbxAnimStack, FbxAnimLayer, FbxAnimCurveNode,
            FbxTime, FbxAnimCurve, FbxDouble3
        )
    except ImportError:
        try:
            import fbx
            from fbx import (
                FbxManager, FbxExporter, FbxImporter, FbxScene, FbxNode,
                FbxSkeleton, FbxAnimStack, FbxAnimLayer, FbxAnimCurveNode,
                FbxTime, FbxAnimCurve, FbxDouble3
            )
        except ImportError:
            logger.error("FBX SDK not available")
            return False
    
    try:
        with open(pkl_path, 'rb') as f:
            smpl_data = pickle.load(f)
        
        smpl_poses = smpl_data['smpl_poses']
        smpl_trans = smpl_data['smpl_trans']
        
        num_frames = smpl_poses.shape[0]
        logger.info(f"Loading SMPL pkl: {num_frames} frames, poses shape: {smpl_poses.shape}, trans shape: {smpl_trans.shape}")
        
        if smpl_poses.shape[1] != 72:
            logger.error(f"Expected smpl_poses shape (N, 72), got {smpl_poses.shape}")
            return False
        
        if smpl_trans.shape[1] != 3:
            logger.error(f"Expected smpl_trans shape (N, 3), got {smpl_trans.shape}")
            return False
        
        manager = FbxManager.Create()
        io_settings = manager.GetIOSettings()
        
        importer = FbxImporter.Create(manager, "")
        if not importer.Initialize(base_fbx_path, -1, io_settings):
            logger.error(f"Failed to initialize FBX importer for {base_fbx_path}")
            manager.Destroy()
            return False
        
        scene = FbxScene.Create(manager, "Scene")
        importer.Import(scene)
        importer.Destroy()
        
        anim_stack = FbxAnimStack.Create(scene, "Take001")
        anim_layer = FbxAnimLayer.Create(anim_stack, "BaseLayer")
        anim_stack.AddMember(anim_layer)
        
        root_node = scene.GetRootNode()
        
        smpl_to_fbx_name_map = {
            0: ['pelvis', 'root', 'hip', 'midhip'],
            1: ['lhip', 'lefthip', 'left hip'],
            2: ['rhip', 'righthip', 'right hip'],
            3: ['spine1', 'spine'],
            4: ['lknee', 'leftknee', 'left knee'],
            5: ['rknee', 'rightknee', 'right knee'],
            6: ['spine2'],
            7: ['lankle', 'leftankle', 'left ankle'],
            8: ['rankle', 'rightankle', 'right ankle'],
            9: ['spine3'],
            10: ['lfoot', 'leftfoot', 'left foot'],
            11: ['rfoot', 'rightfoot', 'right foot'],
            12: ['neck'],
            13: ['lcollar', 'leftcollar', 'left collar'],
            14: ['rcollar', 'rightcollar', 'right collar'],
            15: ['head'],
            16: ['lshoulder', 'leftshoulder', 'left shoulder'],
            17: ['rshoulder', 'rightshoulder', 'right shoulder'],
            18: ['lelbow', 'leftelbow', 'left elbow'],
            19: ['relbow', 'rightelbow', 'right elbow'],
            20: ['lwrist', 'leftwrist', 'left wrist'],
            21: ['rwrist', 'rightwrist', 'right wrist'],
            22: ['lhand', 'lefthand', 'left hand'],
            23: ['rhand', 'righthand', 'right hand'],
        }
        
        def normalize_name(name):
            return name.lower().replace('f_avg_', '').replace('_', '').replace('-', '').replace(' ', '')
        
        def find_skeleton_nodes(root, nodes_dict, depth=0):
            if depth > 10:
                return
            for i in range(root.GetChildCount()):
                child = root.GetChild(i)
                node_name_norm = normalize_name(child.GetName())
                skeleton = child.GetSkeleton()
                if skeleton or 'joint' in node_name_norm or 'bone' in node_name_norm:
                    for joint_idx, name_variants in smpl_to_fbx_name_map.items():
                        for variant in name_variants:
                            variant_norm = normalize_name(variant)
                            if variant_norm in node_name_norm or node_name_norm in variant_norm:
                                if joint_idx not in nodes_dict:
                                    nodes_dict[joint_idx] = child
                                    logger.debug(f"Mapped FBX node '{child.GetName()}' to SMPL joint {joint_idx} ({SMPL_JOINT_ORDER[joint_idx]})")
                                break
                find_skeleton_nodes(child, nodes_dict, depth + 1)
        
        smpl_nodes = {}
        find_skeleton_nodes(root_node, smpl_nodes)
        
        if len(smpl_nodes) == 0:
            logger.error("No SMPL nodes found in FBX file")
            manager.Destroy()
            return False
        
        logger.info(f"Found {len(smpl_nodes)} SMPL nodes in FBX file: {sorted(smpl_nodes.keys())}")
        
        rotation_orders = {}
        fbx_to_scipy_order_map = {
            0: 'XYZ', 1: 'XZY', 2: 'YZX', 3: 'YXZ', 4: 'ZXY', 5: 'ZYX'
        }
        
        for joint_idx, node in smpl_nodes.items():
            try:
                rotation_order = node.GetRotationOrder(0)
                rotation_orders[joint_idx] = fbx_to_scipy_order_map.get(rotation_order, 'XYZ')
            except (TypeError, AttributeError):
                rotation_orders[joint_idx] = 'XYZ'
        
        trans_curve_xs = {}
        trans_curve_ys = {}
        trans_curve_zs = {}
        rot_curve_xs = {}
        rot_curve_ys = {}
        rot_curve_zs = {}
        
        for joint_idx, node in smpl_nodes.items():
            if joint_idx == 0:
                lcl_translation = node.LclTranslation
                trans_curve_xs[joint_idx] = lcl_translation.GetCurve(anim_layer, "X", True)
                trans_curve_ys[joint_idx] = lcl_translation.GetCurve(anim_layer, "Y", True)
                trans_curve_zs[joint_idx] = lcl_translation.GetCurve(anim_layer, "Z", True)
            
            lcl_rotation = node.LclRotation
            rot_curve_xs[joint_idx] = lcl_rotation.GetCurve(anim_layer, "X", True)
            rot_curve_ys[joint_idx] = lcl_rotation.GetCurve(anim_layer, "Y", True)
            rot_curve_zs[joint_idx] = lcl_rotation.GetCurve(anim_layer, "Z", True)
        
        for frame_idx in range(num_frames):
            time = FbxTime()
            time.SetSecondDouble(frame_idx / fps)
            
            pose_frame = smpl_poses[frame_idx].reshape(24, 3)
            trans_frame = smpl_trans[frame_idx]
            
            for joint_idx in range(24):
                if joint_idx not in smpl_nodes:
                    continue
                
                node = smpl_nodes[joint_idx]
                
                if joint_idx == 0:
                    if trans_curve_xs.get(joint_idx) and trans_curve_ys.get(joint_idx) and trans_curve_zs.get(joint_idx):
                        result_x = trans_curve_xs[joint_idx].KeyAdd(time)
                        result_y = trans_curve_ys[joint_idx].KeyAdd(time)
                        result_z = trans_curve_zs[joint_idx].KeyAdd(time)
                        
                        key_index_x = int(result_x[0]) if isinstance(result_x, tuple) else int(result_x)
                        key_index_y = int(result_y[0]) if isinstance(result_y, tuple) else int(result_y)
                        key_index_z = int(result_z[0]) if isinstance(result_z, tuple) else int(result_z)
                        
                        trans_curve_xs[joint_idx].KeySet(key_index_x, time, float(trans_frame[0]))
                        trans_curve_ys[joint_idx].KeySet(key_index_y, time, float(trans_frame[1]))
                        trans_curve_zs[joint_idx].KeySet(key_index_z, time, float(trans_frame[2]))
                
                if rot_curve_xs.get(joint_idx) and rot_curve_ys.get(joint_idx) and rot_curve_zs.get(joint_idx):
                    rot_vec = pose_frame[joint_idx]
                    
                    rot_order = rotation_orders.get(joint_idx, 'XYZ')
                    
                    rot_obj = Rotation.from_rotvec(rot_vec)
                    euler_deg = np.rad2deg(rot_obj.as_euler(rot_order, degrees=False))
                    
                    result_x = rot_curve_xs[joint_idx].KeyAdd(time)
                    result_y = rot_curve_ys[joint_idx].KeyAdd(time)
                    result_z = rot_curve_zs[joint_idx].KeyAdd(time)
                    
                    key_index_x = int(result_x[0]) if isinstance(result_x, tuple) else int(result_x)
                    key_index_y = int(result_y[0]) if isinstance(result_y, tuple) else int(result_y)
                    key_index_z = int(result_z[0]) if isinstance(result_z, tuple) else int(result_z)
                    
                    rot_curve_xs[joint_idx].KeySet(key_index_x, time, float(euler_deg[0]))
                    rot_curve_ys[joint_idx].KeySet(key_index_y, time, float(euler_deg[1]))
                    rot_curve_zs[joint_idx].KeySet(key_index_z, time, float(euler_deg[2]))
        
        exporter = FbxExporter.Create(manager, "")
        if not exporter.Initialize(output_path, -1, io_settings):
            logger.error(f"Failed to initialize FBX exporter for {output_path}")
            manager.Destroy()
            return False
        
        exporter.Export(scene)
        exporter.Destroy()
        manager.Destroy()
        
        logger.info(f"Exported SMPL pkl to FBX: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting SMPL pkl to FBX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
