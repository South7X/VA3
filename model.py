#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import torch
from tqdm import tqdm


class Text2Img:
    def __init__(self, args):
        self.num_images = args.num_images
        self.height = args.height
        self.width = args.width
        self.num_sample_steps = args.num_sample_steps
        self.guidance_scale = args.guidance_scale

        self.unet = args.unet
        self.unet_1 = args.unet_1
        self.unet_2 = args.unet_2
        self.vae = args.vae
        self.text_encoder = args.text_encoder
        self.tokenizer = args.tokenizer
        self.scheduler = args.scheduler

        self.device = args.device

    def text_enc(self, prompt, max_len=None):
        if max_len is None:
            max_len = self.tokenizer.model_max_length
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_len,
                                     truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device))[0].half()
        # prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, self.num_images, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * self.num_images, seq_len, -1)
        return prompt_embeds

    def get_init_latents(self):
        num_channels_latents = self.unet.config.in_channels
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        shape = (self.num_images, num_channels_latents, self.height // vae_scale_factor, self.width // vae_scale_factor)
        latents = torch.randn(shape, device=self.device).half()
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def scheduler_step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor):
        beta_start = self.scheduler.config.beta_start
        beta_end = self.scheduler.config.beta_end
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        t = timestep
        prev_t = timestep - num_train_timesteps // self.num_sample_steps

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32,device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        one = torch.tensor(1.0,device=self.device)
        # 1. compute alphas, betas
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 4. Compute predicted previous sample µ_t
        pred_prev_sample_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # variance type: fixed_small_log
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        log_variance = torch.log(torch.clamp(variance, min=1e-20))

        return pred_prev_sample_mean, log_variance
    
    def denoise_loop(self, latents, prompt_embeds):
        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)  # do classifier free guidance
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            with torch.no_grad():
                noise_pred_uncond, noise_pred_text = self.unet(latent_model_input, t,
                                                        encoder_hidden_states=prompt_embeds).sample.chunk(2)
            # perform guidance
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            mean_p, log_variance = self.scheduler_step(noise_pred, t, latents)
            variance = 0
            if t > 0:
                variance_noise = torch.randn(noise_pred.shape, device=noise_pred.device, dtype=noise_pred.dtype)
                variance = torch.exp(0.5 * log_variance) * variance_noise
            latents = mean_p + variance

        return latents

    def sample(self, prompt):
        text = self.text_enc(prompt)
        uncond = self.text_enc([""], max_len=text.shape[1])
        emb = torch.cat([uncond, text])
        self.scheduler.set_timesteps(self.num_sample_steps, device=self.device)
        latents = self.get_init_latents()
        return self.denoise_loop(latents, emb)
    


class CPfreeText2Img(Text2Img):
    def __init__(self, args):
        super().__init__(args)

    def cpfree_denoise_loop_halfq(self, choice_q, latents, prompt_embeds):
        logprobrat1 = torch.zeros([latents.shape[0], ], dtype=torch.float16, device=latents.device)
        logprobrat2 = torch.zeros([latents.shape[0], ], dtype=torch.float16, device=latents.device)
        logprob1_list = []
        logprob2_list = []

        for i,t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)  # do classifier free guidance
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            with torch.no_grad():
                if choice_q == 0:  
                    u_p, t_p = self.unet_1(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2)
                else:
                    u_p, t_p = self.unet_2(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2)
                u_1, t_1 = self.unet_1(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2)
                u_2, t_2 = self.unet_2(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2)
            # perform guidance, get predicted epsilon
            pred_p = u_p + self.guidance_scale * (t_p - u_p)
            pred_1 = u_1 + self.guidance_scale * (t_1 - u_1)
            pred_2 = u_2 + self.guidance_scale * (t_2 - u_2)
            # perform scheduler step to get mean and variance
            mean_p, log_variance = self.scheduler_step(pred_p, t, latents)
            mean1, _ = self.scheduler_step(pred_1, t, latents)
            mean2, _ = self.scheduler_step(pred_2, t, latents)
            # Add noise
            noise = 0
            variance = torch.exp(0.5 * log_variance)
            if t > 0:
                variance_noise = torch.randn(pred_p.shape, device=pred_p.device, dtype=pred_p.dtype)
                noise = variance * variance_noise
            # compute x_t_p
            latents = mean_p + noise

            if t > 0:
                dists1 = torch.distributions.Normal(mean1, variance.to(mean1.device))
                logprob1 = dists1.log_prob(latents).sum(dim=[1, 2, 3]) / 10.
                logprob1_list.append(logprob1)
                logprobrat1 -= logprob1

                dists2 = torch.distributions.Normal(mean2, variance.to(mean2.device))
                logprob2 = dists2.log_prob(latents).sum(dim=[1, 2, 3]) / 10.
                logprob2_list.append(logprob2)
                logprobrat2 -= logprob2

                dists = torch.distributions.Normal(mean_p, variance.to(mean_p.device))
                p_prob = dists.log_prob(latents).sum(dim=[1, 2, 3]) / 10.
                logprobrat1 += p_prob
                logprobrat2 += p_prob

        logprobrat_max = torch.maximum(logprobrat1, logprobrat2)

        # latents = latents[logprobrat_max < k]
        return latents, logprobrat_max, logprob1_list, logprob2_list


    def cpfree_denoise_loop(self, latents, prompt_embeds):
        """
        Sampling process of the cpfree method with three models
        Implementation from paper "Provable Copyright Protection for Generative Models"
        """

        logprobrat1 = torch.zeros([latents.shape[0], ], dtype=torch.float16, device=latents.device)
        logprobrat2 = torch.zeros([latents.shape[0], ], dtype=torch.float16, device=latents.device)
        logprob1_list = []
        logprob2_list = []

        for i,t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)  # do classifier free guidance
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            with torch.no_grad():
                u_p, t_p = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2)
                u_1, t_1 = self.unet_1(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2)
                u_2, t_2 = self.unet_2(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2)
            # perform guidance, get predicted epsilon
            pred_p = u_p + self.guidance_scale * (t_p - u_p)
            pred_1 = u_1 + self.guidance_scale * (t_1 - u_1)
            pred_2 = u_2 + self.guidance_scale * (t_2 - u_2)
            # perform scheduler step to get mean and variance
            mean_p, log_variance = self.scheduler_step(pred_p, t, latents)
            mean1, _ = self.scheduler_step(pred_1, t, latents)
            mean2, _ = self.scheduler_step(pred_2, t, latents)
            # Add noise
            noise = 0
            variance = torch.exp(0.5 * log_variance)
            if t > 0:
                variance_noise = torch.randn(pred_p.shape, device=pred_p.device, dtype=pred_p.dtype)
                noise = variance * variance_noise
            # compute x_t_p
            latents = mean_p + noise

            if t > 0:
                dists1 = torch.distributions.Normal(mean1, variance.to(mean1.device))
                logprob1 = dists1.log_prob(latents).sum(dim=[1, 2, 3]) / 10.
                logprob1_list.append(logprob1)
                logprobrat1 -= logprob1

                dists2 = torch.distributions.Normal(mean2, variance.to(mean2.device))
                logprob2 = dists2.log_prob(latents).sum(dim=[1, 2, 3]) / 10.
                logprob2_list.append(logprob2)
                logprobrat2 -= logprob2

                dists = torch.distributions.Normal(mean_p, variance.to(mean_p.device))
                p_prob = dists.log_prob(latents).sum(dim=[1, 2, 3]) / 10.
                logprobrat1 += p_prob
                logprobrat2 += p_prob

        logprobrat_max = torch.maximum(logprobrat1, logprobrat2)

        # latents = latents[logprobrat_max < k]
        return latents, logprobrat_max, logprob1_list, logprob2_list

    def sample(self, prompt):
        text = self.text_enc(prompt)
        uncond = self.text_enc([""], max_len=text.shape[1])
        emb = torch.cat([uncond, text])
        self.scheduler.set_timesteps(self.num_sample_steps, device=self.device)
        latents = self.get_init_latents()
        return self.cpfree_denoise_loop(latents, emb)
    
    def sample_halfq(self, choice_q, prompt):
        text = self.text_enc(prompt)
        uncond = self.text_enc([""], max_len=text.shape[1])
        emb = torch.cat([uncond, text])
        self.scheduler.set_timesteps(self.num_sample_steps, device=self.device)
        latents = self.get_init_latents()
        return self.cpfree_denoise_loop_halfq(choice_q, latents, emb)


class GradOptimize(Text2Img):
    def __init__(self, args):
        super().__init__(args)

    def encode_embeddings(self, prompt_ids, prompt_embs, attention_mask=None):
        output_attentions = self.text_encoder.text_model.config.output_attentions
        output_hidden_states = (
            self.text_encoder.text_model.config.output_hidden_states
        )
        return_dict = self.text_encoder.text_model.config.use_return_dict

        hidden_states = self.text_encoder.text_model.embeddings(inputs_embeds=prompt_embs)

        bsz, seq_len = prompt_ids.shape[0], prompt_ids.shape[1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = self.text_encoder.text_model._expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_encoder.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=prompt_ids.device), prompt_ids.to(torch.int).argmax(dim=-1)
        ]

        # if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    def text_enc(self, prompt, prompt_embs=None, prompt_ids=None, max_len=None):
        if prompt_embs is not None:
            text_embs = self.encode_embeddings(prompt_ids, prompt_embs)[0].half()
        else:
            if max_len is None:
                max_len = self.tokenizer.model_max_length
            text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_len,
                                         truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids
            text_embs = self.text_encoder(text_input_ids.to(self.device))[0].half()
        return text_embs

    def sample(self, padded_embeds, dummy_ids, target_features=None):
        text_emb = self.text_enc(dummy_ids, padded_embeds, dummy_ids)
        self.scheduler.set_timesteps(self.num_sample_steps, device=self.device)

        latents_x0 = self.vae.encode(target_features.unsqueeze(0).half().to(self.device)).latent_dist.sample()
        latents_x0 = latents_x0 * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents_x0)
        target = noise

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents_x0.size(0),), device=self.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.scheduler.add_noise(latents_x0, noise, timesteps)

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
        q1_pred = self.unet_1(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample
        q2_pred = self.unet_2(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample

        return {
            'model_pred': model_pred,
            'q1_pred': q1_pred,
            'q2_pred': q2_pred,
            'target': target,
            'timestep': timesteps,
        }
    