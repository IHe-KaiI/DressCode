import torch
import diffusers
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import os

class gen_texture():
	def __init__(self):
		sd_device = "cuda"

		self.vae_diffuse = AutoencoderKL.from_pretrained(
			"./nn/material_gen/refine_vae", subfolder="vae_checkpoint_diffuse", revision="fp16", local_files_only=True, torch_dtype=torch.float16).half().to(sd_device)

		self.vae_normal = AutoencoderKL.from_pretrained(
			"./nn/material_gen/refine_vae", subfolder="vae_checkpoint_normal", revision="fp16", local_files_only=True, torch_dtype=torch.float16).half().to(sd_device)

		self.vae_roughness = AutoencoderKL.from_pretrained(
			"./nn/material_gen/refine_vae", subfolder="vae_checkpoint_roughness", revision="fp16", local_files_only=True, torch_dtype=torch.float16).half().to(sd_device)


		self.invpipe = StableDiffusionPipeline.from_pretrained("./nn/material_gen", torch_dtype=torch.float16, safety_checker=None, vae=self.vae_diffuse)
		self.invpipe = self.invpipe.to(sd_device)

		def patch_conv(module):
			if isinstance(module, torch.nn.Conv2d):
				module.padding_mode="circular"

		self.invpipe.unet.apply(patch_conv)
		self.invpipe.vae.apply(patch_conv)
		self.vae_diffuse.apply(patch_conv)
		self.vae_normal.apply(patch_conv)
		self.vae_roughness.apply(patch_conv)

	def run(self, prompt, out_folder):

		with torch.no_grad():
			
			latents = self.invpipe([prompt], 512, 512, output_type = "latent", return_dict=True)[0]

			pt = self.vae_diffuse.decode(latents / self.vae_diffuse.config.scaling_factor, return_dict=False)[0]
			diffuse = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
			diffuse.save(os.path.join(out_folder, f"texture_diffuse.png"))

			pt = self.vae_normal.decode(latents / self.vae_normal.config.scaling_factor, return_dict=False)[0]
			normal = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
			normal.save(os.path.join(out_folder, f"texture_normal.png"))

			pt = self.vae_roughness.decode(latents / self.vae_roughness.config.scaling_factor, return_dict=False)[0]
			roughness = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
			roughness.save(os.path.join(out_folder, f"texture_roughness.png"))
			

if __name__ == "__main__":
	Gen = gen_texture()

	Gen.run("Deep grey fabric", "./")