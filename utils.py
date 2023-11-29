#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
import torch
import random
import numpy as np
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def set_logger(log_name):
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_sscd_feature(sscd_model, image):
    return sscd_model(image.cpu().unsqueeze(0))[0, :]

def get_clip_feature(clip_processor, clip_model, image):
    image = (image + 1.) / 2.
    image = clip_processor(text=None, images=image,return_tensors="pt")["pixel_values"].to('cuda')
    return clip_model.get_image_features(image)


def get_features(vision_model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = vision_model(**inputs)
    pooled_output = outputs.pooler_output  # pooled CLS states
    return pooled_output


def calculate_image_sim(mode, target_latents, latents):
    if 'cossim' in mode:
        target_latents = target_latents.reshape(1,-1)
        latents = latents.reshape(1,-1)

        target_latents = F.normalize(target_latents, dim=1)
        latents = F.normalize(latents, dim=1)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos(target_latents.half(), latents.half())
    elif 'l1' in mode:
        distance = torch.nn.L1Loss()
        sim = distance(latents.view(1,-1), target_latents.half().view(1,-1))
    elif 'l2' in mode:
        distance = torch.nn.MSELoss()
        sim = distance(latents.view(1,-1), target_latents.half().view(1,-1))
    elif 'sscd' in mode:
        model = torch.jit.load("./sscd_disc_mixup.torchscript.pt")
        target_latents = model(target_latents.cpu().unsqueeze(0))[0, :]
        target_latents = F.normalize(target_latents, dim=0)
        latents = model(latents.unsqueeze(0))[0, :]
        latents = F.normalize(latents, dim=0)
        cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        sim = cosine_similarity(target_latents.view(1,-1).to('cuda'), latents.view(1,-1).to('cuda'))
    else:
        sim = torch.zeros(1)
    return sim



def get_char_table():
    char_table=['·','~','!','@','#','$','%','^','&','*','(',')','=','-','*','+','.','<','>','?',',','\'',';',':','|','\\','/']
    for i in range(ord('A'),ord('Z')+1):
        char_table.append(chr(i))
    for i in range(0,10):
        char_table.append(str(i))
    return char_table


def latents_to_pil(vae, latents):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    if image.ndim == 3:
        images = image[None, ...]
    images = (image * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def plot(x, name, num_bins=20):
    l = len(x) // 2
    fig, ax = plt.subplots()
    ax.hist(x[:l], num_bins, density=True, color='red',alpha=0.5)  # samples from q1
    ax.hist(x[l:], num_bins, density=True, color='green',alpha=0.5)  # samples from q2
    fig.tight_layout()
    plt.savefig(name)

