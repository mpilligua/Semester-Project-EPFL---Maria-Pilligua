# VGGT: Visual Geometry Grounded Transformer - Inference Code

This is a cleaned inference-only version of VGGT (Visual Geometry Grounded Transformer) for 3D vision tasks including camera pose estimation, depth prediction, and 3D point tracking.

**Original Paper**: [VGGT: Visual Geometry Grounded Transformer (CVPR 2025)](https://jytime.github.io/data/VGGT_CVPR25.pdf)  
**Original Repository**: [facebookresearch/vggt](https://github.com/facebookresearch/vggt)

## Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/mpilligua/Semester-Project-EPFL---Maria-Pilligua.git
cd Semester-Project-EPFL---Maria-Pilligua

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.is_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load pretrained weights
# Weights will be automatically downloaded from Hugging Face on first run
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess images
image_paths = ["path/to/image1.png", "path/to/image2.png", "path/to/image3.png"]
images = load_and_preprocess_images(image_paths).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Get predictions
        predictions = model(images)
```

### Output Dictionary

The model returns a dictionary with the following keys:

- **pose_enc**: Camera pose encoding [B, S, 9]
- **depth**: Depth maps [B, S, H, W, 1]
- **depth_conf**: Depth confidence scores [B, S, H, W]
- **world_points**: 3D world coordinates [B, S, H, W, 3]
- **world_points_conf**: 3D points confidence scores [B, S, H, W]
- **images**: Original input images [B, S, 3, H, W]

### Point Tracking (Optional)

```python
query_points = torch.tensor([[100, 150], [200, 250]])  # [N, 2] pixel coordinates

predictions = model(images, query_points=query_points)

# Additional outputs:
# predictions["track"]: Point trajectories [B, S, N, 2]
# predictions["vis"]: Visibility scores [B, S, N]
# predictions["conf"]: Confidence scores [B, S, N]
```

## Architecture Overview

VGGT consists of:

1. **Aggregator**: Alternating frame and global attention transformer
   - Processes sequences of images
   - Frame attention: processes individual frames
   - Global attention: aggregates information across frames
   
2. **Heads**: Task-specific prediction heads
   - **CameraHead**: Predicts camera intrinsics and extrinsics (4 iterations)
   - **DPTHead**: Dense prediction head for depth and 3D points
   - **TrackHead**: Point tracking across frames

3. **Layers**: Core transformer components
   - Multi-head self-attention with RoPE (Rotary Position Embeddings)
   - MLP feedforward networks
   - LayerNorm with layer scaling
   - Patch embedding from vision transformer backbone

## Model Features

- ✅ Feed-forward inference (no test-time optimization)
- ✅ Supports 1 to 100+ images per sequence
- ✅ Outputs camera poses, depth maps, and 3D point clouds
- ✅ Point tracking across frame sequences
- ✅ Mixed precision (bfloat16/float16) support
- ✅ Detailed debug logging for layer-by-layer analysis

## Debug Logging

This version includes comprehensive logging to understand the data flow through the network:

```
[VGGT.forward] INPUT IMAGES
  Shape: torch.Size([1, 3, 3, 518, 518]), dtype: torch.float16
  Range: [0.0000, 1.0000]

[Aggregator.forward] BEGIN
  Batch=1, Sequence=3, Channels=3, Height=518, Width=518
  [Step 1] Normalizing images
    Normalized range: [-2.1179, 2.3699]
  [Step 2] Patch embedding
    Patch tokens shape: torch.Size([3, 1369, 1024])
  ...
[CameraHead.forward] INPUT
  Extracting features...
  [CameraHead.trunk_fn] Iterative refinement with 4 iterations
...
```

To see this logging during inference, simply run your script and check the console output.

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
- **CPU**: Fallback available but significantly slower
- **Memory**: ~12GB for typical 3-5 image sequences
- **Python**: 3.8+

## Known Limitations

- Training code is not included (inference-only)
- Requires Hugging Face model weights to download automatically
- Individual transformer block implementations use gradient checkpointing in training mode

## Citation

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

This code follows the original VGGT license. See repository for full details.

## References

- [VGGT Paper](https://jytime.github.io/data/VGGT_CVPR25.pdf)
- [Project Page](https://vgg-t.github.io/)
- [Original Repository](https://github.com/facebookresearch/vggt)
