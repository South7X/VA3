#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import random
import copy
from math import sqrt
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from utils import *
from model import GradOptimize, CPfreeText2Img
from torchvision import transforms
import torch.nn.functional as F
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)
import wandb
import matplotlib.pyplot as plt
wandb.init(project='anti_naf', allow_val_change=True)

def parse_args():
    parser = argparse.ArgumentParser(description="evaluation script")
    # For attack
    parser.add_argument("--clip_p", type=float, default=0.01, 
                        help='clip bound for loss p')
    parser.add_argument("--max_q_weight", type=float, default=0.95, 
                        help='loss weigt for loss q')
    parser.add_argument("--init_prompt", type=str, default='')
    parser.add_argument("--prefix_prompt_temp", type=str, default='', 
                        help='prefix text prompt template')
    parser.add_argument("--suffix_prompt_temp", type=str, default='', 
                        help='suffix text prompt template')
    parser.add_argument("--ban_words", type=str, default='',
                        help='constrained words that are not allowed to be projected, '
                             'seperate by comma, e.g. pokemon,pokémon')
    parser.add_argument("--target_image_path", type=str, default='./target_image.jpg',
                        help='the target copyright-protected image')
    parser.add_argument("--len_prompt", type=int, default=8, 
                        help='the length of added prompt tokens for attack')
    parser.add_argument("--optimizer_class", type=str, default='adagrad',
                        help='optimizer choosing from adamw, adam, adagrad')
    parser.add_argument("--lr", type=float, default=1e-2, help='learning rate')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step_per_epoch", type=int, default=5000)

    # For text-to-image
    parser.add_argument("--path_model_1", type=str, default='./ckpts/model-q1', 
                        help='path to model q1 (has access to copyright image)')
    parser.add_argument("--path_model_2", type=str, default='./ckpts/model-q2', 
                        help='path to safe model q2 (no access to copyright image)')
    parser.add_argument("--num_sample_steps", type=int, default=50, help='sample diffusion steps')
    parser.add_argument("--num_images", type=int, default=16, help='the number of generated images')
    parser.add_argument("--height", type=int, default=512, help='the height of generated images')
    parser.add_argument("--width", type=int, default=512, help='the width of generated images')
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    # General
    parser.add_argument("--sscd_path", type=str, default='./ckpts/sscd_disc_mixup.torchscript.pt')
    parser.add_argument("--log_name", type=str, default='test')
    parser.add_argument("--log_path", type=str, default='./attack_logs')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", default=False, action='store_true')
    args = parser.parse_args()
    return args


def initialize_prompt(tokenizer, token_embedding):

    # add init prompt setting
    if args.init_prompt != '':
        init_prompt_ids = tokenizer.encode(args.init_prompt, add_special_tokens=False)
        prompt_ids = torch.tensor([init_prompt_ids]).to(args.device)
        prompt_len = prompt_ids.size(1)
    else:
        prompt_len = args.len_prompt

        prompt_ids = torch.randint(len(tokenizer.encoder), (1, prompt_len)).to(args.device)
    prompt_embs = token_embedding(prompt_ids).detach()
    prompt_embs.requires_grad = True

    dummy_ids = [tokenizer.bos_token_id]
    if args.prefix_prompt_temp != '':
        dummy_ids += tokenizer.encode(args.prefix_prompt_temp, add_special_tokens=False)
    dummy_ids += [-1] * prompt_len
    if args.suffix_prompt_temp != '':
        dummy_ids += tokenizer.encode(args.suffix_prompt_temp, add_special_tokens=False)
    dummy_ids += [tokenizer.eos_token_id]
    dummy_ids += [tokenizer.pad_token_id] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * 1).to(args.device)

    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    return prompt_embs, dummy_embeds, dummy_ids


def ban_words(ban_words_list, original_token_embedding):
    if ban_words_list[0] == '':
        return original_token_embedding

    blocklist = []
    blocklist_words = []

    for curr_w in ban_words_list:
        blocklist += args.tokenizer.encode(curr_w)
        blocklist_words.append(curr_w)

    for curr_w in list(args.tokenizer.encoder.keys()):
        for ban_words in ban_words_list:
            if ban_words in curr_w:
                blocklist.append(args.tokenizer.encoder[curr_w])
                blocklist_words.append(curr_w)
    blocklist = list(set(blocklist))

    token_embedding = copy.deepcopy(original_token_embedding)
    if blocklist is not None:
        with torch.no_grad():
            token_embedding.weight[blocklist] = 0
            
    logger.info("blocked words:\n{}".format(blocklist_words))
    return token_embedding


def projection(prompt_embs, token_embedding):
    with torch.no_grad():
        batch_size, seq_len, emb_dim = prompt_embs.shape
        prompt_embs = prompt_embs.reshape((-1,emb_dim))
        embedding_matrix = token_embedding.weight
        vocab_size = embedding_matrix.size(0)

        prompt_embs = F.normalize(prompt_embs, eps=1e-6)
        embedding_matrix = F.normalize(embedding_matrix, eps=1e-6)

        hits = semantic_search(prompt_embs, embedding_matrix,
                                query_chunk_size=prompt_embs.shape[0],
                                top_k=1,
                                score_function=dot_score)
        projected_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=prompt_embs.device)
        projected_indices = projected_indices.reshape((batch_size, seq_len))

        projected_embeds = token_embedding(projected_indices)
    return projected_embeds, projected_indices


def decode_text(input_ids, tokenizer):
    input_ids = input_ids.detach().cpu().numpy()
    text = []
    for input_ids_i in input_ids:
        text.append(tokenizer.decode(input_ids_i))
    return text


def complete_prompt(prompt):
    if args.prefix_prompt_temp is not None:
        prompt = ' '.join([args.prefix_prompt_temp, prompt])
    if args.suffix_prompt_temp is not None:
        prompt = ' '.join([prompt, args.suffix_prompt_temp])
    return prompt


def optimize_prompt_loop(model, target_image):
    token_embedding = args.text_encoder.get_input_embeddings()
    allowed_token_embedding = ban_words(args.ban_words_list, token_embedding)
    prompt_embs, dummy_embeds, dummy_ids = initialize_prompt(args.tokenizer, token_embedding)
    p_bs, p_len, p_dim = prompt_embs.shape
    input_optimizer = args.optimizer([prompt_embs], lr=args.lr, eps=1e-6) 

    best_sim_score = torch.inf
    best_text = ""
    best_step = 0
    best_prompt_no = 0
    train_loss = .0
    train_sim_loss = .0
    train_q1_loss = .0
    train_q2_loss = .0
    prompt_no = 0
    prompt_scores = {}
    prompt_cnts = {}
    epoch_texts = []


    for epoch in range(args.epoch):
        for step in tqdm(range(args.step_per_epoch)):
            projected_embeds, projected_indices = projection(prompt_embs, allowed_token_embedding)

            tmp_embeds = prompt_embs.detach().clone()
            tmp_embeds.data = projected_embeds.data
            tmp_embeds.requires_grad = True

            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)

            model_output = model.sample(padded_embeds, dummy_ids, target_features=target_image)

            q1_loss = F.mse_loss(model_output['q1_pred'], model_output['target'], reduction="mean")
            q2_loss = F.mse_loss(model_output['q2_pred'], model_output['target'], reduction="mean")
            
            
            p_loss = q1_loss
            
            train_sim_loss += p_loss.item() / args.gradient_accumulation_steps
            train_q1_loss += q1_loss.item() / args.gradient_accumulation_steps
            train_q2_loss += q2_loss.item() / args.gradient_accumulation_steps

            loss = torch.clamp(p_loss, min=args.clip_p) * args.max_q_weight + (1. - args.max_q_weight) * torch.max(q1_loss, q2_loss)

            train_loss += loss.item() / args.gradient_accumulation_steps

            grad, = torch.autograd.grad(loss, [tmp_embeds])
            if prompt_embs.grad is None:
                prompt_embs.grad = grad
            else:
                prompt_embs.grad += grad

            if step % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                input_optimizer.step()
                input_optimizer.zero_grad()

                curr_lr = input_optimizer.param_groups[0]["lr"]
                text = decode_text(projected_indices, args.tokenizer)[0]
                prompt_no += 1

                if text in prompt_scores.keys():
                    prompt_scores[text] += train_loss
                    prompt_cnts[text] += 1
                else:
                    prompt_scores[text] = train_loss
                    prompt_cnts[text] = 1

                logger.info('prompt no.{}, text: {}, lr: {}, loss: {}, p_loss: {}, q1_loss: {}, q2_loss: {}'.format(
                    prompt_no, text, curr_lr, train_loss, train_sim_loss, train_q1_loss, train_q2_loss))
                wandb.log({'train_loss':train_loss}, step=prompt_no)
                wandb.log({'p_loss':train_sim_loss}, step=prompt_no)
                wandb.log({'q1_loss':train_q1_loss}, step=prompt_no)
                wandb.log({'q2_loss':train_q2_loss}, step=prompt_no)

                if train_loss < best_sim_score:
                    best_sim_score = train_loss
                    best_prompt_no = prompt_no
                    best_step = step + args.step_per_epoch * epoch
                    best_text = text
                    logger.info('New best at prompt no.: {}, mean loss: {}, text: {}'.format(prompt_no, best_sim_score, best_text))

                train_loss = .0
                train_sim_loss = .0
                train_q1_loss = .0
                train_q2_loss = .0

        epoch_texts.append(complete_prompt(text))

    logger.info('End of optimization, \nAt step {}, prompt no. {} / {}, sim score {}, best_text: \n{}'.format(
        best_step, best_prompt_no, args.optimize_prompts, best_sim_score, best_text))

    return epoch_texts, complete_prompt(best_text)


def cpfree(prompt, target_image, prompt_no, image_name):
    sscd_model = torch.jit.load(args.sscd_path)
    target_sscd_latent = get_sscd_feature(sscd_model, target_image.to('cuda').float())

    eval_model = CPfreeText2Img(args)
    k_list = []
    sim_list = []

    save_path = os.path.join(args.eval_save_path, prompt_no)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for q in [0,1]:  # sample from q1 or q2
        latents, logprobrat_max, _, _ = eval_model.sample_halfq(q, prompt)
        output_imgs = eval_model.decode_latents(latents)
        sim_scores = []
        for img in range(output_imgs.size(0)):
            img_path = os.path.join(save_path, image_name + f'_sample_q{q+1}_{img}.png')
            save_image(
                    output_imgs[img],
                    img_path,
                    nrow=1,
                )
            with torch.no_grad():
                pre_image = Image.open(img_path).convert('RGB')
                pre_image = train_transforms(pre_image).to("cuda")

                pre_sscd_latent = get_sscd_feature(sscd_model,pre_image)
                sscd_score = calculate_image_sim('cossim', target_sscd_latent.to('cuda'), pre_sscd_latent.to('cuda'))
                sim_scores.append(sscd_score.item())
        sim_list += sim_scores
        k_list.append(logprobrat_max.cpu())
        logging.info('Sample from q{}, logprobrat = \n{} \nsscd_cossim = \n{}'.format(
            q+1, logprobrat_max, np.around(sim_scores, 4), 4))
        del latents, logprobrat_max, output_imgs
    sim_list = np.array(sim_list)
    k_list = torch.cat(k_list, dim=0).numpy()

    np.save(os.path.join(save_path, f'{image_name}_logprobrat.npy'), k_list)
    np.save(os.path.join(save_path, f'{image_name}_sscd_cossim.npy'), sim_list)

    k_list[np.isinf(k_list)] = torch.finfo(torch.float16).max

    def plot(x, name, num_bins=20):
        fig, ax = plt.subplots()
        ax.hist(x[:args.num_images], num_bins, density=True, color='red',alpha=0.5)  # samples from q1
        ax.hist(x[args.num_images:], num_bins, density=True, color='green',alpha=0.5)  # samples from q2
        fig.tight_layout()
        plt.savefig(name)

    plot(k_list, os.path.join(save_path, f'{image_name}_logprobrat_max.png'))
    plot(sim_list, os.path.join(save_path, f'{image_name}_sscd_cossim.png'))


def main():
    logging.info('Num of optimized prompts: {}, optimized steps: {}'.format(args.optimize_prompts, args.optimize_steps))

    # load models
    args.unet_1 = UNet2DConditionModel.from_pretrained(args.path_model_1, subfolder="unet", torch_dtype=torch.float16).to("cuda")
    args.unet_2 = UNet2DConditionModel.from_pretrained(args.path_model_2, subfolder="unet", torch_dtype=torch.float16).to("cuda")
    args.unet = args.unet_1
    args.vae = AutoencoderKL.from_pretrained(args.path_model_1, subfolder="vae", torch_dtype=torch.float16).to("cuda")
    args.text_encoder = CLIPTextModel.from_pretrained(args.path_model_1, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
    args.tokenizer = CLIPTokenizer.from_pretrained(args.path_model_1, subfolder="tokenizer", torch_dtype=torch.float16)
    args.scheduler = DDPMScheduler.from_pretrained(args.path_model_1, subfolder="scheduler", torch_dtype=torch.float16)
    args.feature_extractor = CLIPImageProcessor.from_pretrained(args.path_model_1, subfolder="feature_extractor")

    args.ban_words_list = args.ban_words.split(',')

    img = Image.open(args.target_image_path)
    
    target_image = train_transforms(img).half()

    model = GradOptimize(args)

    epoch_texts, best_prompt = optimize_prompt_loop(model, target_image)

    with open(os.path.join(args.log_path, args.log_name, 'prompts.txt'), 'w') as f:
        for i in range(args.epoch):
            step = args.step_per_epoch * (i + 1)
            prompt = epoch_texts[i]
            f.write(f'{i}\t'+prompt+'\n')
            logging.info('Evaluation on prompt of step {}: {}'.format(step, prompt))
            cpfree(prompt, target_image, f'step{step}', args.log_name+f'_step{step}')
        logging.info('Evaluation on best prompt {}'.format(best_prompt))
        cpfree(best_prompt, target_image, 'best', args.log_name+'_best')
        f.write(best_prompt+'\n')
    f.close()


if __name__ == '__main__':
    args = parse_args()
    args.eval_save_path = os.path.join(args.log_path, args.log_name, 'cpfree_results')
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)

    logger = set_logger(os.path.join(args.log_path, args.log_name, 'logger.log'))
    logger.info('Params:\n{}'.format(args))
    wandb.config.update(args)
    set_random_seed(args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizers = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'adagrad': torch.optim.Adagrad,
    }
    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    args.optimizer = optimizers[args.optimizer_class]
    args.optimize_steps = args.epoch * args.step_per_epoch
    args.optimize_prompts = args.optimize_steps / args.gradient_accumulation_steps

    main()
