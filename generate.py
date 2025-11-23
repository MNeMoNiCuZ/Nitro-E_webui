
import torch
import argparse
from core.tools.inference_pipe import init_pipe
import os
from datetime import datetime

class Generator:
    def __init__(self, mode, resolution, device, no_grpo):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.repo_name = "amd/Nitro-E"
        self.mode = mode
        self.resolution = resolution
        self.no_grpo = no_grpo
        self.pipe = self._load_model()

    def _load_model(self):
        print("Loading model...")
        if self.mode == '20_steps':
            if self.resolution == 512:
                ckpt_name = 'Nitro-E-512px.safetensors'
                ckpt_path_grpo = 'ckpt_grpo_512px'
            else: # 1024
                ckpt_name = 'Nitro-E-1024px.safetensors'
                ckpt_path_grpo = None
                if not self.no_grpo:
                    print("Warning: GRPO is not specified for 1024px in the README. Disabling GRPO.")
                    self.no_grpo = True
            use_grpo = not self.no_grpo
        else: # 4_steps
            if self.resolution == 1024:
                print("Warning: 4-step mode is only supported for 512px resolution. Forcing 512px.")
                self.resolution = 512
            ckpt_name = 'Nitro-E-512px-dist.safetensors'
            ckpt_path_grpo = None
            use_grpo = False

        if use_grpo and ckpt_path_grpo:
            pipe = init_pipe(self.device, self.dtype, self.resolution, repo_name=self.repo_name, ckpt_name=ckpt_name, ckpt_path_grpo=ckpt_path_grpo)
        else:
            pipe = init_pipe(self.device, self.dtype, self.resolution, repo_name=self.repo_name, ckpt_name=ckpt_name)
        print("Model loaded.")
        return pipe

    def generate(self, prompt, negative_prompt, num_inference_steps, guidance_scale, num_images_per_prompt, output_path):
        if self.mode == '20_steps':
            steps = num_inference_steps if num_inference_steps is not None else 20
            scale = guidance_scale if guidance_scale is not None else 4.5
        else: # 4_steps
            steps = num_inference_steps if num_inference_steps is not None else 4
            scale = guidance_scale if guidance_scale is not None else 0

        print(f"Generating image for prompt: '{prompt}'")
        images = self.pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            width=self.resolution, 
            height=self.resolution, 
            num_inference_steps=steps, 
            guidance_scale=scale,
            num_images_per_prompt=num_images_per_prompt
        ).images

        image_paths = []
        if images and len(images) > 0:
            now = datetime.now()
            date_folder = now.strftime("%Y-%m-%d")
            output_dir = os.path.join(output_path, date_folder)
            os.makedirs(output_dir, exist_ok=True)

            for i, image in enumerate(images):
                timestamp = now.strftime("%Y-%m-%d - %H.%M.%S")
                filename = f"{timestamp}.{now.microsecond // 1000:03d}_{i+1}.png"
                image_path = os.path.join(output_dir, filename)
                image.save(image_path)
                image_paths.append(image_path)
        else:
            print("Image generation failed.")
        
        return image_paths

def main():
    parser = argparse.ArgumentParser(description='Generate an image using the Nitro-E model.')
    parser.add_argument('--mode', type=str, choices=['20_steps', '4_steps'], required=True, help='Inference mode.')
    parser.add_argument('--prompt', type=str, required=True, help='The text prompt for image generation.')
    parser.add_argument('--output_path', type=str, default='output', help='Path to save the generated image.')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the model on (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--resolution', type=int, default=512, choices=[512, 1024], help='Image resolution.')
    parser.add_argument('--no_grpo', action='store_true', help='Do not use GRPO for post-training fine-tuning (only for 20-step mode).')
    parser.add_argument('--num_inference_steps', type=int, help='Number of inference steps.')
    parser.add_argument('--guidance_scale', type=float, help='Guidance scale.')
    parser.add_argument('--negative_prompt', type=str, default=None, help='The negative prompt.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images to generate per prompt (batch size).')

    args = parser.parse_args()

    generator = Generator(args.mode, args.resolution, args.device, args.no_grpo)
    image_paths = generator.generate(args.prompt, args.negative_prompt, args.num_inference_steps, args.guidance_scale, args.num_images_per_prompt, args.output_path)
    
    for path in image_paths:
        print(path)

if __name__ == '__main__':
    main()
