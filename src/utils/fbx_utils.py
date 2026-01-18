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


def export_humanml3d_to_fbx(
    joints: np.ndarray,
    rotations: np.ndarray,
    output_path: str,
    fps: float = 20.0
) -> bool:
    """Export HumanML3D joint motion directly to FBX without SMPLify.
    
    Creates a clean FBX skeleton with 22 HumanML3D joints and box geometry for visualization.
    
    Args:
        joints: Joint positions array of shape [num_frames, 22, 3]
        rotations: Rotation matrices array of shape [num_frames, 22, 3, 3]
        output_path: Path to output FBX file
        fps: Frames per second for animation
        
    Returns:
        True if export succeeded, False otherwise
    """
    logger = _get_logger()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Remove existing file to ensure overwrite (similar to np.save behavior)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    try:
        from FbxCommon import (
            FbxManager, FbxExporter, FbxScene, FbxNode,
            FbxSkeleton, FbxAnimStack, FbxAnimLayer,
            FbxTime, FbxDouble3, FbxVector4, FbxMesh
        )
    except ImportError:
        try:
            import fbx
            from fbx import (
                FbxManager, FbxExporter, FbxScene, FbxNode,
                FbxSkeleton, FbxAnimStack, FbxAnimLayer,
                FbxTime, FbxDouble3, FbxVector4, FbxMesh
            )
        except ImportError:
            logger.error("FBX SDK not available. Install Autodesk FBX SDK Python bindings.")
            return False
    
    try:
        num_frames, num_joints, _ = joints.shape
        assert num_joints == 22, f"Expected 22 HumanML3D joints, got {num_joints}"
        
        # Verify rotations shape
        if rotations.shape != (num_frames, num_joints, 3, 3):
            logger.warning(f"Unexpected rotations shape: {rotations.shape}, expected ({num_frames}, {num_joints}, 3, 3)")
            # Try to reshape if possible
            if rotations.size == num_frames * num_joints * 9:
                rotations = rotations.reshape(num_frames, num_joints, 3, 3)
            else:
                logger.error(f"Cannot reshape rotations: size {rotations.size} != {num_frames * num_joints * 9}")
                return False
        
        humanml3d_joint_names = [
            "root", "RH", "LH", "BP", "RK", "LK", "BT", "RMrot", "LMrot",
            "BLN", "RF", "LF", "BMN", "RSI", "LSI", "BUN", "RS", "LS",
            "RE", "LE", "RW", "LW",
        ]
        
        manager = FbxManager.Create()
        io_settings = manager.GetIOSettings()
        scene = FbxScene.Create(manager, "Scene")
        root_node = scene.GetRootNode()
        
        # Create skeleton nodes
        skeleton_nodes = [None] * num_joints
        for joint_idx in range(num_joints):
            joint_name = humanml3d_joint_names[joint_idx]
            node = FbxNode.Create(manager, joint_name)
            skeleton = FbxSkeleton.Create(manager, joint_name + "_Skeleton")
            
            try:
                if joint_idx == 0:
                    skeleton_type = FbxSkeleton.EType.eRoot
                else:
                    skeleton_type = FbxSkeleton.EType.eLimbNode
                skeleton.SetSkeletonType(skeleton_type)
            except (AttributeError, TypeError):
                # Fallback: try without setting skeleton type (it's optional)
                pass
            
            skeleton.LimbLength.Set(joints[0, joint_idx, 2])
            node.SetNodeAttribute(skeleton)
            root_node.AddChild(node)
            skeleton_nodes[joint_idx] = node
        
        for joint_idx in range(num_joints):
            world_pos = joints[0, joint_idx]
            skeleton_nodes[joint_idx].LclTranslation.Set(FbxDouble3(world_pos[0], world_pos[1], world_pos[2]))
            skeleton_nodes[joint_idx].LclRotation.Set(FbxDouble3(0, 0, 0))
        
        half_size = 0.05
        for joint_idx in range(num_joints):
            mesh = FbxMesh.Create(manager, f"Box_{humanml3d_joint_names[joint_idx]}")
            vertices = [
                np.array([-half_size, -half_size, -half_size]), np.array([half_size, -half_size, -half_size]),
                np.array([half_size, half_size, -half_size]), np.array([-half_size, half_size, -half_size]),
                np.array([-half_size, -half_size, half_size]), np.array([half_size, -half_size, half_size]),
                np.array([half_size, half_size, half_size]), np.array([-half_size, half_size, half_size]),
            ]
            
            # Initialize control points
            mesh.InitControlPoints(len(vertices))
            for i_vert, vert in enumerate(vertices):
                mesh.SetControlPointAt(FbxVector4(vert[0], vert[1], vert[2]), i_vert)
            
            for face in [[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
                         [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                         [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]:
                mesh.BeginPolygon()
                for vid in face:
                    mesh.AddPolygon(vid)
                mesh.EndPolygon()
            
            box_node = FbxNode.Create(manager, f"Box_{humanml3d_joint_names[joint_idx]}")
            box_node.SetNodeAttribute(mesh)
            box_node.LclTranslation.Set(FbxDouble3(0, 0, 0))
            box_node.LclRotation.Set(FbxDouble3(0, 0, 0))
            skeleton_nodes[joint_idx].AddChild(box_node)
        
        anim_stack = FbxAnimStack.Create(scene, "Take001")
        anim_layer = FbxAnimLayer.Create(anim_stack, "BaseLayer")
        anim_stack.AddMember(anim_layer)
        
        if len(rotations.shape) == 3 and rotations.shape[2] == 9:
            rotations = rotations.reshape(num_frames, num_joints, 3, 3)
        elif rotations.shape != (num_frames, num_joints, 3, 3):
            logger.error(f"Invalid rotations shape: {rotations.shape}")
            return False
        
        rot_objs = Rotation.from_matrix(rotations.reshape(-1, 3, 3))
        euler_deg = np.rad2deg(rot_objs.as_euler('xyz', degrees=False)).reshape(num_frames, num_joints, 3)
        
        def get_key_index(result):
            return int(result[0]) if isinstance(result, tuple) else int(result)
        
        for frame_idx in range(num_frames):
            time = FbxTime()
            time.SetSecondDouble(frame_idx / fps)
            
            for joint_idx in range(num_joints):
                node = skeleton_nodes[joint_idx]
                world_pos = joints[frame_idx, joint_idx]
                euler = euler_deg[frame_idx, joint_idx]
                
                for prop, values, channels in [
                    (node.LclTranslation, world_pos, ["X", "Y", "Z"]),
                    (node.LclRotation, euler, ["X", "Y", "Z"])
                ]:
                    curves = [prop.GetCurve(anim_layer, ch, True) for ch in channels]
                    key_indices = [get_key_index(curve.KeyAdd(time)) for curve in curves]
                    for curve, key_idx, val in zip(curves, key_indices, values):
                        curve.KeySet(key_idx, time, float(val))
        
        # Export
        exporter = FbxExporter.Create(manager, "")
        if not exporter.Initialize(output_path, -1, io_settings):
            logger.error(f"Failed to initialize FBX exporter for {output_path}")
            manager.Destroy()
            return False
        
        exporter.Export(scene)
        exporter.Destroy()
        manager.Destroy()
        
        logger.info(f"Exported HumanML3D motion to FBX: {output_path} ({num_frames} frames)")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting HumanML3D to FBX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
