#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from math import sqrt
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm
import argparse
import numpy as np
import random
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from utils import *
from model import Text2Img, CPfreeText2Img

def parse_args():
    parser = argparse.ArgumentParser(description="evaluation script")
    parser.add_argument("--prompt", type=str, default='a black and white zebra standing next to a star')
    parser.add_argument("--mode", type=str, default='cpfree',
                        help='sampling mode: cpfree, model_1, model_2')
    parser.add_argument("--path_model_1", type=str, default='./ckpts/model-q1', 
                        help='path to model q1 (has access to copyright image)')
    parser.add_argument("--path_model_2", type=str, default='./ckpts/model-q2', 
                        help='path to safe model q2 (no access to copyright image)')
    parser.add_argument("--target_image_path", type=str, default='./target_image.jpg', 
                        help='the target copyright-protected image')
    parser.add_argument("--clip_path", type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument("--sscd_path", type=str, default='./ckpts/sscd_disc_mixup.torchscript.pt')

    parser.add_argument("--num_sample_steps", type=int, default=50, help='sample diffusion steps')
    parser.add_argument("--num_images", type=int, default=1, help='the number of generated images for each batch')
    parser.add_argument("--height", type=int, default=512, help='the height of generated images')
    parser.add_argument("--width", type=int, default=512, help='the width of generated images')
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_times", type=int, default=100, help='how many times of sampling')

    parser.add_argument("--ac", type=float, default=.965, help='acceptance probability')
    parser.add_argument("--sim_thres", type=float, default=0.4, help='similarity threshold')
    parser.add_argument("--k", type=float, default=20000.0, help='threshold k')

    parser.add_argument("--save_path", type=str, default='./sample_results/')
    parser.add_argument("--img_name", type=str, default='anti-naf')
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def cpfree_sample(prompt, targets):
    eval_model = CPfreeText2Img(args)
    rho_list = []
    sim_list = {'l1':[],
                'l2':[],
                'clip':[],
                'sscd':[]
                }

    half_times = args.sample_times // 2
    for i in tqdm(range(args.sample_times)):
        choice_q = 0 if i < half_times else 1  # half sample from q1 and half sample from q2
        latents, logprobrat_max, _, _ = eval_model.sample_halfq(choice_q, prompt)
        output_imgs = eval_model.decode_latents(latents)
        for img in range(output_imgs.size(0)):

            idx = i * args.num_images + img
            image_path = os.path.join(img_save_path, img_name + f'_sample_{idx}.png')
            save_image(
                output_imgs[img],
                image_path,
                nrow=1
            )
            with torch.no_grad():
                pre_image = Image.open(image_path).convert('RGB')
                pre_image = train_transforms(pre_image).to("cuda")

                pre_clip_latent = get_clip_feature(clip_processor, clip_model, pre_image)
                pre_sscd_latent = get_sscd_feature(sscd_model, pre_image)
                
                l1_score = calculate_image_sim('l1', targets['image'], pre_image)
                l2_score = calculate_image_sim('l2', targets['image'], pre_image)
                clip_score = calculate_image_sim('cossim', targets['clip'], pre_clip_latent)
                sscd_score = calculate_image_sim('cossim', targets['sscd'].to('cuda'), pre_sscd_latent.to('cuda'))

            sim_list['l1'].append(l1_score.item())
            sim_list['l2'].append(l2_score.item())
            sim_list['clip'].append(clip_score.item())
            sim_list['sscd'].append(sscd_score.item())
        rho_list.append(logprobrat_max.cpu())
        print('Iteration {}, sample from q_{}, rho = {}, l1 = {:.4f}, l2 = {:.4f}, clip_score = {:.4f}, sscd_score = {:.4f}'.format(
                i, choice_q+1, logprobrat_max.cpu(),sim_list['l1'][-1],sim_list['l2'][-1],sim_list['clip'][-1],sim_list['sscd'][-1]))
    rho_list = torch.cat(rho_list, dim=0).numpy()
    sim_score_l1 = np.array(sim_list['l1'])
    sim_score_l2 = np.array(sim_list['l2'])
    sim_score_clip = np.array(sim_list['clip'])
    sim_score_sscd = np.array(sim_list['sscd'])
    np.save(os.path.join(save_path, f'{args.img_name}_rho.npy'), rho_list)
    np.save(os.path.join(save_path, f'{args.img_name}_l1.npy'), sim_score_l1)
    np.save(os.path.join(save_path, f'{args.img_name}_l2.npy'), sim_score_l2)
    np.save(os.path.join(save_path, f'{args.img_name}_clip_score.npy'), sim_score_clip)
    np.save(os.path.join(save_path, f'{args.img_name}_sscd_score.npy'), sim_score_sscd)
    
    plot(rho_list, os.path.join(save_path, f'{args.img_name}_rho.png'))
    plot(sim_score_l1, os.path.join(save_path, f'{args.img_name}_l1.png'))
    plot(sim_score_l2, os.path.join(save_path, f'{args.img_name}_l2.png'))
    plot(sim_score_clip, os.path.join(save_path, f'{args.img_name}_clip_score.png'))
    plot(sim_score_sscd, os.path.join(save_path, f'{args.img_name}_sscd_score.png'))

    rho_list.sort()
    return rho_list[int(len(rho_list)*(args.ac))]


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = os.path.join(args.save_path, args.img_name, args.mode)
    img_save_path = os.path.join(save_path, 'images')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    img_name = args.img_name

    # Load model components from pretrained model path
    args.path_model_p = args.path_model_1
    if args.mode == 'cpfree':
        args.unet = UNet2DConditionModel.from_pretrained(args.path_model_p, subfolder="unet", torch_dtype=torch.float16).to("cuda")
        args.unet_1 = UNet2DConditionModel.from_pretrained(args.path_model_1, subfolder="unet", torch_dtype=torch.float16).to("cuda")
        args.unet_2 = UNet2DConditionModel.from_pretrained(args.path_model_2, subfolder="unet", torch_dtype=torch.float16).to("cuda")
        component_path = args.path_model_p
    elif args.mode == 'model_1':
        args.unet = UNet2DConditionModel.from_pretrained(args.path_model_1, subfolder="unet", torch_dtype=torch.float16).to("cuda")
        args.unet_1 = args.unet_2 = args.unet
        component_path = args.path_model_1
    elif args.mode == 'model_2':
        args.unet = UNet2DConditionModel.from_pretrained(args.path_model_2, subfolder="unet", torch_dtype=torch.float16).to("cuda")
        args.unet_1 = args.unet_2 = args.unet
        component_path = args.path_model_2
    else:
        raise ValueError('Please select mode from cpfree, model_1, model_2, model_p')
    args.vae = AutoencoderKL.from_pretrained(component_path, subfolder="vae", torch_dtype=torch.float16).to("cuda")
    args.text_encoder = CLIPTextModel.from_pretrained(component_path, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
    args.tokenizer = CLIPTokenizer.from_pretrained(component_path, subfolder="tokenizer", torch_dtype=torch.float16)
    args.scheduler = DDPMScheduler.from_pretrained(component_path, subfolder="scheduler", torch_dtype=torch.float16)
 
    clip_model = CLIPModel.from_pretrained(args.clip_path).to('cuda')
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    sscd_model = torch.jit.load(args.sscd_path)
    

    img = Image.open(args.target_image_path).convert('RGB')
    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    target_image = train_transforms(img).to("cuda")
    target_clip_latent = get_clip_feature(target_image)
    target_sscd_latent = get_sscd_feature(target_image)

    targets = {'image': target_image,
               'clip': target_clip_latent,
               'sscd': target_sscd_latent,
    }

    if args.mode == 'cpfree':
        k = cpfree_sample(args.prompt, targets)
        print(k)
    else:
        eval_model = Text2Img(args)
        for i in tqdm(range(args.sample_times)):
            latents = eval_model.sample(args.prompt)
            output_imgs = eval_model.decode_latents(latents)
            for img in range(output_imgs.size(0)):
                idx = i * args.num_images + img
                image_path = os.path.join(img_save_path, img_name + f'_sample_{idx}.png')
                save_image(
                    output_imgs[img],
                    image_path,
                    nrow=1
                )

    