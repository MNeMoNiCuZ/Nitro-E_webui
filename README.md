---
license: mit
pipeline_tag: text-to-image
library_name: diffusers
---
# Nitro-E Webui



# AMD Nitro-E


![image/png](https://huggingface.co/amd/Nitro-E/resolve/main/assets/teaser.png)

## Introduction
Nitro-E is a family of text-to-image diffusion models focused on highly efficient training. With just 304M parameters, Nitro-E is designed to be resource-friendly for both training and inference. For training, it only takes 1.5 days on a single node with 8 AMD Instinct™ MI300X GPUs. On the inference side, Nitro-E delivers a throughput of 18.8 samples per second (batch size 32, 512px images) a single AMD Instinct MI300X GPU. The distilled version can further increase the throughput to 39.3 samples per second. The release consists of:

* [Nitro-E-512px](https://huggingface.co/amd/Nitro-E/blob/main/Nitro-E-512px.safetensors): a EMMDiT-based 20-steps model train from scratch.
* [Nitro-E-512px-dist](https://huggingface.co/amd/Nitro-E/blob/main/Nitro-E-512px-dist.safetensors): a EMMDiT-based model distilled from Nitro-E-512px.
* [Nitro-E-512px-GRPO](https://huggingface.co/amd/Nitro-E/tree/main/ckpt_grpo_512px): a post-training model fine-tuned from Nitro-E-512px using Group Relative Policy Optimization (GRPO) strategy. 

⚡️ [Open-source code](https://github.com/AMD-AGI/Nitro-E)! 
⚡️ [technical blog](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html)! 


## Details

* **Model architecture**: We propose Efficient Multimodal Diffusion Transformer (E-MMDiT), an efficient and lightweight multimodal diffusion model with only 304M
parameters for fast image synthesis requiring low training resources. Our design philosophy centers on token reduction as the computational
cost scales significantly with the token count. We adopt a highly compressive visual tokenizer to produce a more compact representation and propose a novel multi-path compression
module for further compression of tokens. To enhance our design, we introduce Position Reinforcement, which strengthens positional information to maintain spatial coherence,
and Alternating Subregion Attention (ASA), which performs attention within subregions to further reduce computational cost. In addition, we propose AdaLN-affine, an
efficient lightweight module for computing modulation parameters in transformer blocks. See our technical blog post for more details.
* **Dataset**: Our models were trained on a dataset of ~25M images consisting of both real and synthetic data sources that are openly available on the internet. We make use of the following datasets for training: [Segment-Anything-1B](https://ai.meta.com/datasets/segment-anything/), [JourneyDB](https://journeydb.github.io/), [DiffusionDB](https://github.com/poloclub/diffusiondb) and [DataComp](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) as prompt of the generated data.
* **Training cost**: The Nitro-E-512px model requires only 1.5 days of training from scratch on a single node with 8 AMD Instinct™ MI300X GPUs.


## Quickstart
* **Image generation with 20 steps**:
```python
import torch
from core.tools.inference_pipe import init_pipe

device = torch.device('cuda:0')
dtype = torch.bfloat16
repo_name = "amd/Nitro-E"

resolution = 512
ckpt_name = 'Nitro-E-512px.safetensors'

# for 1024px model
# resolution = 1024
# ckpt_name = 'Nitro-E-1024px.safetensors'

use_grpo = True

if use_grpo: 
    pipe = init_pipe(device, dtype, resolution, repo_name=repo_name, ckpt_name=ckpt_name, ckpt_path_grpo='ckpt_grpo_512px')
else:
    pipe = init_pipe(device, dtype, resolution, repo_name=repo_name, ckpt_name=ckpt_name)
prompt = 'A hot air balloon in the shape of a heart grand canyon'
images = pipe(prompt=prompt, width=resolution, height=resolution, num_inference_steps=20, guidance_scale=4.5).images
```

* **Image generation with 4 steps**:
```python
import torch
from core.tools.inference_pipe import init_pipe

device = torch.device('cuda:0')
dtype = torch.bfloat16
resolution = 512
repo_name = "amd/Nitro-E"
ckpt_name = 'Nitro-E-512px-dist.safetensors'

pipe = init_pipe(device, dtype, resolution, repo_name=repo_name, ckpt_name=ckpt_name)
prompt = 'A hot air balloon in the shape of a heart grand canyon'

images = pipe(prompt=prompt, width=resolution, height=resolution, num_inference_steps=4, guidance_scale=0).images
```


## License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

This project is licensed under the [MIT License](https://mit-license.org/).
