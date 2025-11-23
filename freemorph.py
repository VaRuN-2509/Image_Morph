import os
from pathlib import Path

import torch
import json
import tqdm
from dataloader import Dataloader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from aid_attention import (
    OuterConvergedAttnProcessor_SDPA,
    OuterConvergedAttnProcessor_SDPA2,
    OuterInterpolatedAttnProcessor_SDPA,
)
from aid_utils import (
    fourier_filter,
    generate_beta_tensor,
    linear_interpolation,
    load_im_from_path,
    spherical_interpolation,
)

import argparse, sys, os, math, re

@torch.no_grad()
def aid_inversion(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncon: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale=7,
):
    warmup_step = int(steps * 0.3)
    warmup_step2 = int(steps * 0.6)
    iter_latent = latent.clone()
    for i, t in enumerate(timesteps):
        if i > warmup_step and i < warmup_step2:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA())
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA())
        elif i > warmup_step2:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA2(is_fused=False))
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(OuterConvergedAttnProcessor_SDPA2(is_fused=False))
        else:
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn1"):
                    m.set_processor(AttnProcessor2_0())
            for m_name, m in unet.named_modules():
                if m_name.endswith("attn2"):
                    m.set_processor(AttnProcessor2_0())
        for _ in range(6):
            noise_pred_cond = unet(iter_latent, t, encoder_hidden_states=text_input_con).sample
            iter_latent = invert_scheduler.step(sample=latent, model_output=noise_pred_cond, timestep=t).prev_sample
        latent = iter_latent.clone()
    return latent


@torch.no_grad()
def aid_forward(
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    text_input_con: torch.Tensor,
    text_input_uncond: torch.Tensor,
    coef_self_attn: torch.Tensor,
    coef_cross_attn: torch.Tensor,
    guidance_scale: float = 3,
):
    attn_processor_dict = {
        "original": {
            "self_attn": AttnProcessor2_0(),
            "cross_attn": AttnProcessor2_0(),
        },
    }
    warmup_step1 = int(len(timesteps) * 0.2)
    warmup_step2 = int(len(timesteps) * 0.6)
    for i, t in enumerate(tqdm(timesteps, desc="Interpolating attention processors", ncols=100)):
        if i < warmup_step1:
            interpolate_attn_proc = {
                "self_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_self_attn),
                "cross_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=False, t=coef_cross_attn),
            }
        elif i < warmup_step2:
            interpolate_attn_proc = {
                "self_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_self_attn),
                "cross_attn": OuterInterpolatedAttnProcessor_SDPA(is_fused=True, t=coef_cross_attn),
            }
        else:
            interpolate_attn_proc = {
                "self_attn": AttnProcessor2_0(),
                "cross_attn": AttnProcessor2_0(),
            }
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(interpolate_attn_proc["self_attn"])
            if m_name.endswith("attn2"):
                m.set_processor(interpolate_attn_proc["cross_attn"])
        noise_pred_cond = unet(latent, t, encoder_hidden_states=text_input_con).sample
        for m_name, m in unet.named_modules():
            if m_name.endswith("attn1"):
                m.set_processor(attn_processor_dict["original"]["self_attn"])
            if m_name.endswith("attn2"):
                m.set_processor(attn_processor_dict["original"]["cross_attn"])
        noise_pred_uncond = unet(latent, t, encoder_hidden_states=text_input_uncond).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = forward_scheduler.step(sample=latent, model_output=noise_pred, timestep=t).prev_sample
    return latent


if __name__ == "__main__":
  
    
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)
    model_name = "runwayml/stable-diffusion-v1-5"
    image_resolution = 768
    dtype_weight = torch.float16
    steps = 15
    edit_strength = 0.8
    guidance_scale = 7.5
    accelerater = Accelerator()
    this_gpu = accelerater.local_process_index
    # class_name = args.class_name
    # exp_name = args.class_name
    # save_dir = f"./eval_results/{exp_name}/freemorph"
    
   
    device = accelerater.device
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device, dtype_weight)
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        use_fast=False,
        chat_template=None,
        trust_remote_code=False,
        tokenizer_config_file=None,
        skip_special_tokens=False
    )
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device, dtype_weight)
    forward_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    forward_scheduler.set_timesteps(steps)
    invert_scheduler = DDIMInverseScheduler.from_pretrained(model_name, subfolder="scheduler")
    invert_scheduler.set_timesteps(steps)
    injection_noise = torch.randn((1, 4, 96, 96), device=device, dtype=dtype_weight)

    input_dir = Path("/home/shifu/workspace/varun/split_edits_by_instruction")
    json_output = []
    
    for json_file in input_dir.rglob("*.json"):
        if json_file.is_file() and json_file.suffix == ".json":   # optional filter
            file_name = json_file.stem
            save_dir = f"./eval_results/freemorph/{file_name}"
            json_path = f"output_log/edit_{file_name}"

        print("Processing file:", json_file)
        dataloader = Dataloader(str(json_file))
        print(json_file)

        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
                print(f"Skipping directory — already processed ({save_dir})")
                continue
        
        dataloader = Dataloader(f"{json_file}")
        
        for i,all_images in enumerate(dataloader.read()):
            exp_id = i
            out_dir = f"{save_dir}/{exp_id}"


            os.makedirs(out_dir, exist_ok=True)
            print(f"Processing entry {exp_id}...")
            image_0 = load_im_from_path(all_images['input_image_url'])
            image_1 = load_im_from_path(all_images['output_image_url'])
            vae_dtype = next(vae.parameters()).dtype
            vae_device = next(vae.parameters()).device

            image_0 = image_0.to(device=vae_device, dtype=vae_dtype)
            image_1 = image_1.to(device=vae_device, dtype=vae_dtype)

            prompts = all_images['prompts']
            json_output.append(
                {
                    "exp_id": i,
                    "edit_prompt": all_images['edit_prompt'],
                    "image_paths": [
                        all_images['input_image_url'],
                        all_images['output_image_url'],
                    ],
                    "prompts": prompts,
                }
            )
            exp_id = i
            latent_x_list = []
            text_input_con_list = []
            text_input_uncond_list = []
            original_images = []
            for img_idx in range(2):
                image = image_0 if img_idx == 0 else image_1
                original_images.append(image)
                input_ids = tokenizer(
                    [prompts[img_idx], ""],
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                ).input_ids.to(device)
                text_input = text_encoder(input_ids).last_hidden_state.to(device, dtype_weight)
                text_input_con, text_input_uncond = text_input.chunk(2, dim=0)
                text_input_con_list.append(text_input_con)
                text_input_uncond_list.append(text_input_uncond)
                latent_x = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                latent_x_list.append(latent_x)

            interpolation_size = 8
            coef_attn = generate_beta_tensor(interpolation_size, alpha=20, beta=20)
            latent_x = spherical_interpolation(latent_x_list[0], latent_x_list[1], interpolation_size).squeeze(0)
            invert_timesteps = invert_scheduler.timesteps - 1
            invert_timesteps = invert_timesteps[: int(steps * edit_strength)]
            text_input_con_all = linear_interpolation(
                text_input_con_list[0], text_input_con_list[1], interpolation_size
            ).squeeze(0)
            text_input_uncond_all = linear_interpolation(
                text_input_uncond_list[0], text_input_uncond_list[1], interpolation_size
            ).squeeze(0)
            reverted_latent = aid_inversion(
                latent=latent_x,
                timesteps=invert_timesteps,
                text_input_con=text_input_con_all,
                text_input_uncon=text_input_uncond_all,
                coef_cross_attn=coef_attn,
                coef_self_attn=coef_attn,
            )
            thresholds = [48] + [44] + [42] * 2 + [42] + [42] * 2 + [44] + [48]
            new_input_latents_list = []
            for k in range(1, 8 - 1):
                new_input_latent = fourier_filter(
                    x=reverted_latent[k].unsqueeze(0),
                    y=injection_noise.clone(),
                    threshold=int(thresholds[k]),
                )
                new_input_latents_list.append(new_input_latent.clone())
            latent = torch.cat(
                [reverted_latent[0].unsqueeze(0)] + new_input_latents_list + [reverted_latent[-1].unsqueeze(0)]
            )
            forward_timesteps = forward_scheduler.timesteps - 1
            forward_timesteps = forward_timesteps[steps - int(steps * edit_strength) :]
            coef_attn = generate_beta_tensor(interpolation_size, alpha=20, beta=20)
            morphing_latent = aid_forward(
                forward_timesteps,
                latent=latent,
                text_input_con=text_input_con_all,
                text_input_uncond=text_input_uncond_all,
                coef_cross_attn=coef_attn,
                coef_self_attn=coef_attn,
                guidance_scale=guidance_scale,
            )
            images = []
            for x in morphing_latent:
                x = vae.decode(x.unsqueeze(0) / vae.config.scaling_factor).sample
                x = (x / 2 + 0.5).clamp(0, 1)
                x = x.detach().to(torch.float).cpu()
                images.append(x)
            images[0] = (original_images[0] / 2 + 0.5).detach().to(torch.float).cpu()
            images[-1] = (original_images[1] / 2 + 0.5).detach().to(torch.float).cpu()
            out_dir = f"{save_dir}/{exp_id}"
            os.makedirs(out_dir, exist_ok=True)  # ✅ ensures directory exists
            save_image(torch.cat(images), f"{out_dir}/{exp_id}.png")

            os.makedirs(json_path, exist_ok=True)
            with open(f"{json_path}/data_loaded.json", "w") as f:
                for item in json_output:
                    f.write(json.dumps(item) + "\n")

            for i, img in enumerate(images):
                save_path = os.path.join(out_dir, f"{exp_id}_s{i}.png")
                save_image(img, save_path)
                print(f"✅ Saved: {save_path}")
