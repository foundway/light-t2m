from typing import Any, Dict, Tuple

import hydra
import rootutils
import numpy as np
import torch
import lightning.pytorch as L
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

import os
from os.path import join as pjoin

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

from src.data.humanml.scripts.motion_process import recover_from_ric
from src.data.humanml.common.quaternion import cont6d_to_matrix_np
from src.utils.fbx_utils import export_humanml3d_to_fbx


@torch.no_grad()
def generation(model, cfg):
    mean = np.load(pjoin(cfg.data_dir, "Mean.npy"))
    std = np.load(pjoin(cfg.data_dir, "Std.npy"))
    
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    if cfg.device != "cpu":
        model.to(f"cuda:{cfg.device}")
    model.eval()

    mean = torch.from_numpy(mean).to(model.device)
    std = torch.from_numpy(std).to(model.device)

    save_path = pjoin(cfg.save_path, "gen_joints")
    os.makedirs(save_path, exist_ok=True)
    
    log.info(f"Using text prompt: '{cfg.text}' (type: {type(cfg.text)})")
    texts = [cfg.text] * cfg.repeats
    log.info(f"Texts list: {texts[:2]}... (total: {len(texts)} repeats)")
    motion_length = cfg.length
    
    motion = torch.zeros([1, motion_length, 263], device=model.device).repeat([len(texts), 1, 1])
    lens = torch.tensor(motion_length, dtype=torch.long, device=model.device).repeat(len(texts))
    gen_motions = model.sample_motion(motion, lens, texts)
    gen_motions = gen_motions * std + mean
    gen_joints = recover_from_ric(gen_motions, 22).cpu().numpy()
    
    log.info("Extracting rotations from model output")
    from src.data.humanml.scripts.motion_process import recover_root_rot_pos
    from src.data.humanml.common.quaternion import quaternion_to_cont6d
    
    r_rot_quat, _ = recover_root_rot_pos(gen_motions)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    
    start_indx = 1 + 2 + 1 + 21 * 3
    cont6d_params = torch.cat([r_rot_cont6d, gen_motions[..., start_indx:start_indx + 21 * 6]], dim=-1)
    cont6d_params = cont6d_params.view(-1, 22, 6)
    
    rot_matrices = cont6d_to_matrix_np(cont6d_params.cpu().numpy())
    num_samples, num_frames = len(texts), gen_joints[0].shape[0]
    gen_rotations = rot_matrices.reshape(num_samples, num_frames, 22, 3, 3)
    
    name = cfg.sample_name
    for i in range(len(texts)):
        np.save(pjoin(save_path, name + f"_{i}.npy"), gen_joints[i])    

    fbx_save_path = pjoin(cfg.save_path, "gen_fbx")
    os.makedirs(fbx_save_path, exist_ok=True)
    
    for i in range(len(texts)):
        output_fbx_path = pjoin(fbx_save_path, name + f"_{i}.fbx")
        
        log.info(f"Exporting HumanML3D motion to FBX for sample {i}...")
        success_fbx = export_humanml3d_to_fbx(
            joints=gen_joints[i],
            rotations=gen_rotations[i],
            output_path=output_fbx_path,
            fps=cfg.get("fps", 20.0)
        )
        
        if success_fbx:
            log.info(f"Exported FBX: {output_fbx_path}")
        else:
            log.warning(f"Failed to export FBX: {output_fbx_path}")



@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    torch.set_float32_matmul_precision('high')

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)


    if cfg.ckpt_path is None or cfg.ckpt_path == "none":
        log.error("No checkpoint path provided!")
        return {}, None
    
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    for key in list(state_dict.keys()):
        if 'orig_mod.' in key:
            state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)


    num_parameters = sum([x.numel() for x in model.denoiser.parameters() if x.requires_grad])
    log.info("Total parameters: %.3fM" % (num_parameters / 1000_000))

    log.info("Starting generation!")

    generation(model, cfg)

    log.info("Done!")
    return {}, None



@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_motion.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
