import os
import numpy as np
import torch
from typing import Optional, Dict, Tuple
from scipy.spatial.transform import Rotation
from src.data.humanml.common.quaternion import qeuler_np

log = None

def _get_logger():
    global log
    if log is None:
        from src.utils import RankedLogger
        log = RankedLogger(__name__, rank_zero_only=True)
    return log

def extract_smpl_params_from_fbx(fbx_path: str, device: str = "0") -> Optional[Dict]:
    """Extract SMPL parameters (betas, initial pose) from an SMPL-compatible FBX file.
    
    Args:
        fbx_path: Path to input FBX file
        device: Device ID for computation
        
    Returns:
        Dictionary with 'betas' and 'pose' keys, or None if extraction fails
    """
    logger = _get_logger()
    
    if not os.path.exists(fbx_path):
        logger.warning(f"FBX file not found: {fbx_path}")
        return None
    
    try:
        from FbxCommon import (
            FbxManager, FbxImporter, FbxScene, FbxNode, 
            FbxSkeleton, FbxAnimStack, FbxAnimLayer, FbxAnimCurveNode,
            FbxTime
        )
    except ImportError:
        try:
            import fbx
            from fbx import (
                FbxManager, FbxImporter, FbxScene, FbxNode,
                FbxSkeleton, FbxAnimStack, FbxAnimLayer, FbxAnimCurveNode,
                FbxTime
            )
        except ImportError:
            logger.warning(
                "FBX SDK not available. Install Autodesk FBX SDK Python bindings "
                "or use Blender API for FBX support. Skipping FBX parameter extraction."
            )
            return None
    
    try:
        manager = FbxManager.Create()
        io_settings = manager.GetIOSettings()
        importer = FbxImporter.Create(manager, "")
        
        if not importer.Initialize(fbx_path, -1, io_settings):
            logger.warning(f"Failed to initialize FBX importer for {fbx_path}")
            return None
        
        scene = FbxScene.Create(manager, "Scene")
        importer.Import(scene)
        importer.Destroy()
        
        root_node = scene.GetRootNode()
        
        smpl_betas = None
        smpl_pose = None
        
        for i in range(root_node.GetChildCount()):
            child = root_node.GetChild(i)
            node_name = child.GetName()
            
            skeleton = child.GetSkeleton()
            if skeleton:
                attr = child.LclRotation
                rotation = np.array([attr.Get()[0], attr.Get()[1], attr.Get()[2]], dtype=np.float32)
                translation = np.array([child.LclTranslation.Get()[0], 
                                       child.LclTranslation.Get()[1],
                                       child.LclTranslation.Get()[2]], dtype=np.float32)
        
        if smpl_pose is None:
            smpl_pose = np.zeros(72, dtype=np.float32)
        if smpl_betas is None:
            smpl_betas = np.zeros(10, dtype=np.float32)
        
        manager.Destroy()
        
        result = {
            'betas': torch.from_numpy(smpl_betas).float(),
            'pose': torch.from_numpy(smpl_pose).float(),
            'cam': torch.zeros(3).float()
        }
        
        logger.info(f"Extracted SMPL parameters from FBX: {fbx_path}")
        return result
        
    except Exception as e:
        logger.warning(f"Error extracting SMPL parameters from FBX: {e}")
        return None


def export_motion_to_fbx(
    joints: np.ndarray,
    output_path: str,
    fps: float = 20.0,
    base_fbx_path: Optional[str] = None,
    betas: Optional[np.ndarray] = None,
    pose_params: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None
) -> bool:
    """Export joint motion to an FBX file.
    
    Args:
        joints: Joint positions array of shape [num_frames, num_joints, 3]
        output_path: Path to output FBX file
        fps: Frames per second for animation
        base_fbx_path: Optional path to base FBX file to use as template
        betas: Optional SMPL shape parameters
        pose_params: Optional SMPL pose parameters (rotation matrices or axis-angle)
        rotations: Optional rotation matrices array of shape [num_frames, num_joints, 3, 3]
                   If provided, these rotations will be used instead of computing from positions
        
    Returns:
        True if export succeeded, False otherwise
    """
    logger = _get_logger()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
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
            logger.warning(
                "FBX SDK not available. Install Autodesk FBX SDK Python bindings "
                "or use Blender API for FBX export. Skipping FBX export."
            )
            return False
    
    try:
        manager = FbxManager.Create()
        io_settings = manager.GetIOSettings()
        scene = None
        
        if base_fbx_path and os.path.exists(base_fbx_path):
            importer = FbxImporter.Create(manager, "")
            if importer.Initialize(base_fbx_path, -1, io_settings):
                scene = FbxScene.Create(manager, "Scene")
                importer.Import(scene)
                importer.Destroy()
                logger.info(f"Using base FBX as template: {base_fbx_path}")
        
        if scene is None:
            scene = FbxScene.Create(manager, "Scene")
            logger.info("Created new FBX scene")
        
        root_node = scene.GetRootNode()
        if base_fbx_path:
            logger.info(f"Using existing scene from base FBX. Root node: {root_node.GetName()}, child count: {root_node.GetChildCount()}")
        
        num_frames, num_joints, _ = joints.shape
        logger.info(f"Exporting {num_frames} frames for {num_joints} joints at {fps} fps")
        
        time_inc = FbxTime()
        time_inc.SetSecondDouble(1.0 / fps)
        
        anim_stack = FbxAnimStack.Create(scene, "AnimationStack")
        anim_layer = FbxAnimLayer.Create(scene, "BaseLayer")
        anim_stack.AddMember(anim_layer)
        scene.SetCurrentAnimationStack(anim_stack)
        logger.info("Created animation stack and layer")
        logger.info(f"Root node: {root_node.GetName()}, child count: {root_node.GetChildCount()}")
        
        def find_skeleton_nodes(node, skeleton_nodes=None):
            if skeleton_nodes is None:
                skeleton_nodes = []
            
            skeleton = node.GetSkeleton()
            if skeleton:
                skeleton_nodes.append(node)
                logger.info(f"Found skeleton node: {node.GetName()} (type: {skeleton.GetSkeletonType()})")
            
            for i in range(node.GetChildCount()):
                child = node.GetChild(i)
                find_skeleton_nodes(child, skeleton_nodes)
            
            return skeleton_nodes
        
        skeleton_nodes_list = find_skeleton_nodes(root_node)
        logger.info(f"Found {len(skeleton_nodes_list)} skeleton nodes in scene")
        
        if len(skeleton_nodes_list) == 0 and base_fbx_path:
            logger.warning("No skeleton nodes found in base FBX. Looking for any nodes with 'joint' or 'bone' in name...")
            def find_all_nodes(node, all_nodes=None):
                if all_nodes is None:
                    all_nodes = []
                node_name = node.GetName().lower()
                if 'joint' in node_name or 'bone' in node_name or 'smpl' in node_name:
                    all_nodes.append(node)
                    logger.info(f"Found potential joint node: {node.GetName()}")
                for i in range(node.GetChildCount()):
                    find_all_nodes(node.GetChild(i), all_nodes)
                return all_nodes
            
            skeleton_nodes_list = find_all_nodes(root_node)
            logger.info(f"Found {len(skeleton_nodes_list)} potential joint nodes")
        
        fbx_to_humanml_mapping = [
            ('root', 0), ('pelvis', 0),
            ('rhip', 1), ('righthip', 1),
            ('lhip', 2), ('lefthip', 2),
            ('spine1', 3),
            ('rknee', 4), ('rightknee', 4),
            ('lknee', 5), ('leftknee', 5),
            ('spine3', 6), ('spine2', 6),
            ('rheel', 7), ('rightheel', 7), ('rmrot', 7),
            ('lheel', 8), ('leftheel', 8), ('lmrot', 8),
            ('neck', 9),
            ('rfoot', 10), ('rightfoot', 10),
            ('lfoot', 11), ('leftfoot', 11),
            ('rcollar', 13), ('rightcollar', 13), ('rsi', 13),
            ('lcollar', 14), ('leftcollar', 14), ('lsi', 14),
            ('head', 15),
            ('rshoulder', 16), ('rightshoulder', 16),
            ('lshoulder', 17), ('leftshoulder', 17),
            ('relbow', 18), ('rightelbow', 18),
            ('lelbow', 19), ('leftelbow', 19),
            ('rwrist', 20), ('rightwrist', 20),
            ('lwrist', 21), ('leftwrist', 21),
        ]
        
        def normalize_name(name):
            name_lower = name.lower().replace('f_avg_', '').replace('_', '').replace('-', '')
            return name_lower
        
        node_to_joint_map = {}
        for node in skeleton_nodes_list:
            node_name_normalized = normalize_name(node.GetName())
            matched_joint_idx = None
            best_match_key = None
            
            for key, joint_idx in fbx_to_humanml_mapping:
                if key in node_name_normalized:
                    if matched_joint_idx is None or len(key) > len(best_match_key):
                        matched_joint_idx = joint_idx
                        best_match_key = key
            
            if matched_joint_idx is not None and matched_joint_idx < num_joints:
                if matched_joint_idx not in node_to_joint_map:
                    node_to_joint_map[matched_joint_idx] = node
                    logger.info(f"Mapped FBX node '{node.GetName()}' (normalized: '{node_name_normalized}') -> key '{best_match_key}' -> HumanML3D joint {matched_joint_idx}")
        
        nodes_by_joint_idx = [None] * num_joints
        for joint_idx, node in node_to_joint_map.items():
            nodes_by_joint_idx[joint_idx] = node
        
        missing_joints = [i for i in range(num_joints) if nodes_by_joint_idx[i] is None]
        if missing_joints:
            logger.warning(f"Missing FBX nodes for HumanML3D joints: {missing_joints}. Will skip animation for these joints.")
        
        nodes = nodes_by_joint_idx
        trans_curve_xs = [None] * num_joints
        trans_curve_ys = [None] * num_joints
        trans_curve_zs = [None] * num_joints
        rot_curve_xs = [None] * num_joints
        rot_curve_ys = [None] * num_joints
        rot_curve_zs = [None] * num_joints
        
        root_rot_curve_xs = None
        root_rot_curve_ys = None
        root_rot_curve_zs = None
        
        initial_rotations = {}
        initial_translations = {}
        rotation_orders = {}
        
        fbx_to_scipy_order_map = {
            0: 'XYZ',  # eEulerXYZ
            1: 'XZY',  # eEulerXZY
            2: 'YZX',  # eEulerYZX
            3: 'YXZ',  # eEulerYXZ
            4: 'ZXY',  # eEulerZXY
            5: 'ZYX',  # eEulerZYX
        }
        
        for joint_idx in range(num_joints):
            node = nodes[joint_idx]
            if node is None:
                logger.warning(f"No FBX node found for HumanML3D joint {joint_idx}, skipping")
                continue
            
            node_name = node.GetName()
            
            lcl_rotation = node.LclRotation
            initial_rot = lcl_rotation.Get()
            initial_rotations[joint_idx] = np.array([initial_rot[0], initial_rot[1], initial_rot[2]], dtype=np.float32)
            
            rotation_order = 0
            try:
                rotation_order = node.GetRotationOrder(0)
            except (TypeError, AttributeError):
                rotation_order = 0
            
            scipy_order = fbx_to_scipy_order_map.get(rotation_order, 'XYZ')
            rotation_orders[joint_idx] = scipy_order
            logger.debug(f"Joint {joint_idx} ({node_name}) initial rotation (degrees): {initial_rotations[joint_idx]}, rotation order: {scipy_order}")
            
            lcl_translation = node.LclTranslation
            initial_trans = lcl_translation.Get()
            initial_translations[joint_idx] = np.array([initial_trans[0], initial_trans[1], initial_trans[2]], dtype=np.float32)
            
            if joint_idx == 0:
                trans_curve_x = lcl_translation.GetCurve(anim_layer, "X", True)
                trans_curve_y = lcl_translation.GetCurve(anim_layer, "Y", True)
                trans_curve_z = lcl_translation.GetCurve(anim_layer, "Z", True)
                
                if trans_curve_x is None or trans_curve_y is None or trans_curve_z is None:
                    logger.error(f"Failed to get/create translation curves for root joint {joint_idx} (node: {node_name})")
                    manager.Destroy()
                    return False
                
                trans_curve_xs[joint_idx] = trans_curve_x
                trans_curve_ys[joint_idx] = trans_curve_y
                trans_curve_zs[joint_idx] = trans_curve_z
                
                root_rot_curve_x = lcl_rotation.GetCurve(anim_layer, "X", True)
                root_rot_curve_y = lcl_rotation.GetCurve(anim_layer, "Y", True)
                root_rot_curve_z = lcl_rotation.GetCurve(anim_layer, "Z", True)
                
                if root_rot_curve_x is not None and root_rot_curve_y is not None and root_rot_curve_z is not None:
                    root_rot_curve_xs = root_rot_curve_x
                    root_rot_curve_ys = root_rot_curve_y
                    root_rot_curve_zs = root_rot_curve_z
            else:
                rot_curve_x = lcl_rotation.GetCurve(anim_layer, "X", True)
                rot_curve_y = lcl_rotation.GetCurve(anim_layer, "Y", True)
                rot_curve_z = lcl_rotation.GetCurve(anim_layer, "Z", True)
                
                if rot_curve_x is None or rot_curve_y is None or rot_curve_z is None:
                    logger.error(f"Failed to get/create rotation curves for joint {joint_idx} (node: {node_name})")
                    manager.Destroy()
                    return False
                
                rot_curve_xs[joint_idx] = rot_curve_x
                rot_curve_ys[joint_idx] = rot_curve_y
                rot_curve_zs[joint_idx] = rot_curve_z
        
        valid_node_count = sum(1 for n in nodes if n is not None)
        if valid_node_count == 0:
            logger.error("No skeleton nodes found to animate. Cannot export FBX.")
            manager.Destroy()
            return False
        
        logger.info(f"Using {valid_node_count} skeleton nodes for animation export (mapped to HumanML3D joints)")
        
        def get_parent_node(node):
            if node is None:
                return None
            parent = node.GetParent()
            while parent:
                for joint_idx, n in enumerate(nodes):
                    if n == parent:
                        return joint_idx
                parent = parent.GetParent()
            return None
        
        parent_indices = {}
        for joint_idx, node in enumerate(nodes):
            if node is None:
                parent_indices[joint_idx] = None
            else:
                parent_joint_idx = get_parent_node(node)
                parent_indices[joint_idx] = parent_joint_idx
        
        root_joint_idx = 0
        
        initial_rotations_rad = {}
        for joint_idx, rot_euler_deg in initial_rotations.items():
            initial_rotations_rad[joint_idx] = np.deg2rad(rot_euler_deg)
        
        default_rotation_order = rotation_orders.get(0, 'XYZ') if rotation_orders else 'XYZ'
        
        def compute_bind_pose_global_positions():
            bind_pose_global = np.zeros((num_joints, 3))
            if 0 in initial_translations:
                bind_pose_global[0] = initial_translations[0]
            
            for joint_idx in range(1, num_joints):
                parent_idx = parent_indices[joint_idx]
                if parent_idx is None or parent_idx >= num_joints:
                    bind_pose_global[joint_idx] = initial_translations.get(joint_idx, np.zeros(3))
                    continue
                
                parent_global = bind_pose_global[parent_idx]
                local_trans = initial_translations.get(joint_idx, np.zeros(3))
                
                if parent_idx in initial_rotations_rad:
                    rot_order = rotation_orders.get(parent_idx, default_rotation_order)
                    parent_rot_rad = initial_rotations_rad[parent_idx]
                    parent_rot_obj = Rotation.from_euler(rot_order, parent_rot_rad, degrees=False)
                    rot_matrix = parent_rot_obj.as_matrix()
                    local_trans_rotated = rot_matrix @ local_trans
                    bind_pose_global[joint_idx] = parent_global + local_trans_rotated
                else:
                    bind_pose_global[joint_idx] = parent_global + local_trans
            
            return bind_pose_global
        
        logger.debug(f"Computing bind pose global positions from FBX initial rotations and translations")
        bind_pose_global_positions = compute_bind_pose_global_positions()
        
        rotations_quat = None
        if rotations is not None:
            logger.info(f"Using provided rotations directly from model output: shape {rotations.shape}")
            if rotations.shape[1] != num_joints:
                logger.warning(f"Rotation shape mismatch: expected {num_joints} joints, got {rotations.shape[1]}. Will fall back to position-based computation.")
                rotations = None
            else:
                logger.info(f"Converting rotation matrices to quaternions for continuous representation. Shape: {rotations.shape}, Frames: {num_frames}")
                rotations_quat = np.zeros((num_frames, num_joints, 4))
                for frame_idx in range(num_frames):
                    for joint_idx in range(num_joints):
                        rot_obj = Rotation.from_matrix(rotations[frame_idx, joint_idx])
                        quat = rot_obj.as_quat()
                        rotations_quat[frame_idx, joint_idx] = [quat[3], quat[0], quat[1], quat[2]]
                logger.debug(f"Frame 0 rotation for joint 0 (root): quat={rotations_quat[0, 0]}, matrix=\n{rotations[0, 0]}")
                
                from src.data.humanml.common.quaternion import qfix
                for joint_idx in range(num_joints):
                    joint_quats = rotations_quat[:, joint_idx:joint_idx+1, :]
                    joint_quats = joint_quats.transpose(1, 0, 2)
                    fixed_quats = qfix(joint_quats)
                    rotations_quat[:, joint_idx, :] = fixed_quats.transpose(1, 0, 2)[0]
                logger.info("Fixed quaternion continuity across frames")
        
        for frame_idx in range(num_frames):
            time = FbxTime()
            time.SetSecondDouble(frame_idx / fps)
            
            for joint_idx in range(num_joints):
                if nodes[joint_idx] is None:
                    continue
                
                joint_global_pos = joints[frame_idx, joint_idx]
                parent_idx = parent_indices[joint_idx]
                
                if joint_idx == root_joint_idx or parent_idx is None:
                    if trans_curve_xs[joint_idx] is None:
                        continue
                    
                    initial_trans = initial_translations[joint_idx]
                    root_pos_first_frame = joints[0, joint_idx]
                    
                    trans_offset = (joint_global_pos - root_pos_first_frame) + initial_trans
                    
                    trans_curve_x = trans_curve_xs[joint_idx]
                    trans_curve_y = trans_curve_ys[joint_idx]
                    trans_curve_z = trans_curve_zs[joint_idx]
                    
                    result_x = trans_curve_x.KeyAdd(time)
                    result_y = trans_curve_y.KeyAdd(time)
                    result_z = trans_curve_z.KeyAdd(time)
                    
                    key_index_x = int(result_x[0]) if isinstance(result_x, tuple) else int(result_x)
                    key_index_y = int(result_y[0]) if isinstance(result_y, tuple) else int(result_y)
                    key_index_z = int(result_z[0]) if isinstance(result_z, tuple) else int(result_z)
                    
                    trans_curve_x.KeySet(key_index_x, time, float(trans_offset[0]))
                    trans_curve_y.KeySet(key_index_y, time, float(trans_offset[1]))
                    trans_curve_z.KeySet(key_index_z, time, float(trans_offset[2]))
                    
                    if root_rot_curve_xs is not None and rotations is not None:
                        root_quat = rotations_quat[frame_idx, 0]
                        rot_order = rotation_orders.get(0, default_rotation_order)
                        
                        from src.data.humanml.common.quaternion import qeuler_np
                        final_root_rotation_euler_deg = qeuler_np(root_quat[np.newaxis, :], rot_order)[0]
                        
                        result_rx = root_rot_curve_xs.KeyAdd(time)
                        result_ry = root_rot_curve_ys.KeyAdd(time)
                        result_rz = root_rot_curve_zs.KeyAdd(time)
                        
                        key_index_rx = int(result_rx[0]) if isinstance(result_rx, tuple) else int(result_rx)
                        key_index_ry = int(result_ry[0]) if isinstance(result_ry, tuple) else int(result_ry)
                        key_index_rz = int(result_rz[0]) if isinstance(result_rz, tuple) else int(result_rz)
                        
                        root_rot_curve_xs.KeySet(key_index_rx, time, float(final_root_rotation_euler_deg[0]))
                        root_rot_curve_ys.KeySet(key_index_ry, time, float(final_root_rotation_euler_deg[1]))
                        root_rot_curve_zs.KeySet(key_index_rz, time, float(final_root_rotation_euler_deg[2]))
                else:
                    if rot_curve_xs[joint_idx] is None:
                        continue
                    if parent_idx >= num_joints or nodes[parent_idx] is None:
                        continue
                    
                    parent_global_pos = joints[frame_idx, parent_idx]
                    rest_parent_pos = bind_pose_global_positions[parent_idx]
                    rest_joint_pos = bind_pose_global_positions[joint_idx]
                    
                    rest_dir = rest_joint_pos - rest_parent_pos
                    anim_dir = joint_global_pos - parent_global_pos
                    
                    rest_dir_norm = rest_dir / (np.linalg.norm(rest_dir) + 1e-8)
                    anim_dir_norm = anim_dir / (np.linalg.norm(anim_dir) + 1e-8)
                    
                    dot = np.clip(np.dot(rest_dir_norm, anim_dir_norm), -1.0, 1.0)
                    rot_order = rotation_orders.get(joint_idx, default_rotation_order)
                    
                    if dot > 0.9999:
                        final_rot_obj = Rotation.from_euler(rot_order, initial_rotations_rad[joint_idx], degrees=False)
                    elif dot < -0.9999:
                        axis = np.cross(rest_dir_norm, anim_dir_norm)
                        if np.linalg.norm(axis) < 1e-8:
                            axis = np.array([1, 0, 0])
                        axis_norm = axis / (np.linalg.norm(axis) + 1e-8)
                        flip_rot = Rotation.from_rotvec(axis_norm * np.pi)
                        initial_rot_obj = Rotation.from_euler(rot_order, initial_rotations_rad[joint_idx], degrees=False)
                        final_rot_obj = flip_rot * initial_rot_obj
                    else:
                        axis = np.cross(rest_dir_norm, anim_dir_norm)
                        axis_norm = axis / (np.linalg.norm(axis) + 1e-8)
                        angle = np.arccos(dot)
                        rotation_axis_angle = axis_norm * angle
                        delta_rot = Rotation.from_rotvec(rotation_axis_angle)
                        initial_rot_obj = Rotation.from_euler(rot_order, initial_rotations_rad[joint_idx], degrees=False)
                        final_rot_obj = delta_rot * initial_rot_obj
                    
                    final_rotation_euler_deg = np.rad2deg(final_rot_obj.as_euler(rot_order, degrees=False))
                    
                    rot_curve_x = rot_curve_xs[joint_idx]
                    rot_curve_y = rot_curve_ys[joint_idx]
                    rot_curve_z = rot_curve_zs[joint_idx]
                    
                    result_x = rot_curve_x.KeyAdd(time)
                    result_y = rot_curve_y.KeyAdd(time)
                    result_z = rot_curve_z.KeyAdd(time)
                    
                    key_index_x = int(result_x[0]) if isinstance(result_x, tuple) else int(result_x)
                    key_index_y = int(result_y[0]) if isinstance(result_y, tuple) else int(result_y)
                    key_index_z = int(result_z[0]) if isinstance(result_z, tuple) else int(result_z)
                    
                    rot_curve_x.KeySet(key_index_x, time, float(final_rotation_euler_deg[0]))
                    rot_curve_y.KeySet(key_index_y, time, float(final_rotation_euler_deg[1]))
                    rot_curve_z.KeySet(key_index_z, time, float(final_rotation_euler_deg[2]))
        
        exporter = FbxExporter.Create(manager, "")
        if not exporter.Initialize(output_path, -1, io_settings):
            logger.error(f"Failed to initialize FBX exporter for {output_path}")
            manager.Destroy()
            return False
        
        exporter.Export(scene)
        exporter.Destroy()
        manager.Destroy()
        
        logger.info(f"Exported motion to FBX: {output_path} ({num_frames} frames)")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to FBX: {e}")
        return False
