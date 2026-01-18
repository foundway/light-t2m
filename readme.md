# Light-T2M Fork

This is a fork of [Light-T2M](https://github.com/qinghuannn/light-t2m) with fixes for CUDA 13.1 compatibility and modern PyTorch versions. For the original README and documentation, see the [official repository](https://github.com/qinghuannn/light-t2m).

## Quick Start

```bash
python src/sample_motion.py text="A person walks and turns right."
```
Outputs:
- `./visual_datas/gen_joints/gen_motion_{X}.npy`
- `./visual_datas/gen_fbx/gen_motion_{X}.fbx`

## Key Changes & Fixes

### CUDA 13.1 & PyTorch 2.9+ Compatibility

1. **PyTorch CUDA Version Check**: Modified `torch/utils/cpp_extension.py` to allow CUDA 13.1 with PyTorch compiled for CUDA 12.1 (changes `RuntimeError` to `UserWarning`)

2. **Mamba Build Fixes** (`mamba/setup.py`):
   - Removed unsupported GPU architectures for CUDA 13.1 (compute_53, compute_60, compute_62, compute_70, compute_72)
   - Added support for RTX 5080 (compute_120, sm_120)
   - Replaced deprecated CUB APIs in `mamba/csrc/selective_scan/reverse_scan.cuh`:
     - `cub::LaneId()` → `threadIdx.x & 31`
     - `cub::CTA_SYNC()` → `__syncthreads()`

3. **PyTorch 2.9+ Compatibility**: Added `weights_only=False` to all `torch.load()` calls in:
   - `src/sample_motion.py`
   - `src/gen_motion.py`
   - `src/eval.py`
   - `src/models/evaluator/T2M/evaluator.py`
   - `src/transforms/rots2joints/base.py`
   - `src/transforms/rots2rfeats/base.py`
   - `src/transforms/joints2jfeats/base.py`

4. **Optional T2MEvaluator**: Made T2MEvaluator initialization conditional in `src/models/light_final.py` to prevent errors when `deps/t2m_guo` directory is missing

5. **Visualization Fix**: Added automatic creation of `mesh_dir` in `visualize/blend_render.py`

## Installation Notes

### CUDA 13.1 Setup

```bash
# Set CUDA_HOME before building mamba/causal-conv1d
export CUDA_HOME=/usr/local/cuda

# Install causal-conv1d with no build isolation
pip install --no-build-isolation causal-conv1d==1.3.0.post1

# Install mamba with no build isolation
cd mamba && pip install --no-build-isolation -e .
```

**Note**: You may need to patch PyTorch's `cpp_extension.py` to allow CUDA version mismatch. See fix #1 above.

### PyTorch Installation

Tested with PyTorch 2.9.1+cu130 (CUDA 13.0) on CUDA 13.1:

```bash
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Configuration Changes

In [DDPM_ori.yaml](./configs/model/noise_scheduler/DDPM_ori.yaml), `prediction_type` is set to `sample` as opposed to the original `epsilon` to simplify commandline arguement override.

## Citation

Please cite the original Light-T2M paper:

```bibtex
@inproceedings{light-t2m,
  title={Light-T2M: A Lightweight and Fast Model for Text-to-motion Generation},
  author={Zeng, Ling-An and Huang, Guohong and Wu, Gaojie and Zheng, Wei-Shi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## License

This fork maintains the same [MIT License](https://github.com/qinghuannn/light-t2m) as the original project.