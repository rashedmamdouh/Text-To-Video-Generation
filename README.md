# Text-to-Video Generation using Pretrained Diffusion Model

This project utilizes a pretrained diffusion model to generate a video based on a given textual prompt. The diffusion model generates high-quality video frames that match the provided prompt.

## Model Details

Developed by: ModelScope
Model type: Diffusion-based text-to-video generation model
Language(s): English
License: CC-BY-NC-ND
Resources for more information: ModelScope GitHub Repository, Summary.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use this text-to-video generation model, follow these steps:

1. Install Python dependencies:

```bash
pip install diffusers
```

2. Clone the repository and download the pretrained model weights.

## Usage

To generate a video from a textual prompt, follow the example code provided:

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Check if CUDA is available
if torch.cuda.is_available():
    # Use GPU
    device = torch.device("cuda")
    print("GPU available. Using GPU for computation.")
else:
    # Use CPU as fallback
    device = torch.device("cpu")
    print("GPU not available. Using CPU for computation.")

# Load the diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16, device=device)
pipe.enable_model_cpu_offload()

# Memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

# Define the textual prompt
prompt = "Spiderman is surfing"

# Generate video frames
video_frames = pipe(prompt, num_frames=40).frames[0]

# Export video
video_path = export_to_video(video_frames, fps=10, output_video_path="vid2-export.mp4")
```

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on GitHub.

## License

License:
cc-by-nc-4.0
