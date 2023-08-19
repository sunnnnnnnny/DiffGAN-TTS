import os
import json
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from functools import partial
from inspect import isfunction

from .modules import Denoiser
from utils.tools import get_noise_schedule_list


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape): # a tensor([0.0000, 0.7076, 0.9916, 0.9998]) t tensor([1, 1]) x_shape torch.Size([2, 1, 80, 737])
    b, *_ = t.shape
    out = a.gather(-1, t) # out tensor([0.7076, 0.7076])
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) #


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(self, args, preprocess_config, model_config, train_config):
        super().__init__()
        self.model = args.model
        self.denoise_fn = Denoiser(preprocess_config, model_config)
        self.mel_bins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]

        betas = get_noise_schedule_list(
            schedule_mode=model_config["denoiser"]["noise_schedule_naive"],
            timesteps=model_config["denoiser"]["timesteps" if self.model == "naive" else "shallow_timesteps"],
            min_beta=model_config["denoiser"]["min_beta"],
            max_beta=model_config["denoiser"]["max_beta"],
            s=model_config["denoiser"]["s"],
        )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = train_config["loss"]["noise_loss"]

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            self.register_buffer("spec_min", torch.FloatTensor(stats["spec_min"])[None, None, :model_config["denoiser"]["keep_bins"]])
            self.register_buffer("spec_max", torch.FloatTensor(stats["spec_max"])[None, None, :model_config["denoiser"]["keep_bins"]])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t): # x_start [2,1,80,737] x_t [2,1,80,737] t [1,1]
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + # self.posterior_mean_coef1 tensor([1.0000, 0.5206, 0.0804, 0.0035])
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t # self.posterior_mean_coef2  tensor([0.0000, 0.1102, 0.0434, 0.0126])
        ) # posterior_mean [2,1,80,737]
        posterior_variance = extract(self.posterior_variance, t, x_t.shape) # posterior_variance [2,1,1,1]
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape) # posterior_log_variance_clipped [2,1,1,1]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False): # x_start [2,1,80,737] x_t [2,1,80,737] t [1,1] repeat_noise=False
        b, *_, device = *x_start.shape, x_start.device # b  2
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t) # model_mean [2,1,80,737] model_log_variance [2,1,1,1]
        noise = noise_like(x_start.shape, device, repeat_noise) # noise [2,1,80,737]
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, cond, spk_emb, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x_t.shape, x_t.device
        x_0_pred = self.denoise_fn(x_t, t, cond, spk_emb)

        if clip_denoised:
            x_0_pred.clamp_(-1., 1.)

        return self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)

    @torch.no_grad()
    def interpolate(self, x1, x2, t, cond, spk_emb, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        x = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond, spk_emb)
        x = x[:, 0].transpose(1, 2)
        return self.denorm_spec(x)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @torch.no_grad()
    def sampling(self, noise=None):
        b, *_, device = *self.cond.shape, self.cond.device
        t = self.num_timesteps
        shape = (self.cond.shape[0], 1, self.mel_bins, self.cond.shape[2])
        xs = [torch.randn(shape, device=device) if noise is None else noise]
        for i in tqdm(reversed(range(0, t)), desc="sample time step", total=t):
            x = self.p_sample(xs[-1], torch.full((b,), i, device=device, dtype=torch.long), self.cond, self.spk_emb)
            xs.append(x)
        output = [self.denorm_spec(x[:, 0].transpose(1, 2)) for x in xs]
        return output

    def diffuse_trace(self, x_start, mask):
        b, *_, device = *x_start.shape, x_start.device
        trace = [self.norm_spec(x_start).clamp_(-1., 1.) * ~mask.unsqueeze(-1)]
        for t in range(self.num_timesteps):
            t = torch.full((b,), t, device=device, dtype=torch.long)
            trace.append(
                self.diffuse_fn(x_start, t)[:, 0].transpose(1, 2) * ~mask.unsqueeze(-1)
            )
        return trace

    def diffuse_fn(self, x_start, t, noise=None): # x_start [2,737,80] t[t1=0,t2=0] noise=None
        x_start = self.norm_spec(x_start)
        x_start = x_start.transpose(1, 2)[:, None, :, :]  # [2,1,80,737]
        zero_idx = t < 0 # for items where t is -1, zero_idx = [False, False]
        t[zero_idx] = 0 # t[0,0]
        noise = default(noise, lambda: torch.randn_like(x_start)) # torch.randn_like 标准正态分布 [2,1,80,737]
        out = self.q_sample(x_start=x_start, t=t, noise=noise) # [2,1,80,737]
        out[zero_idx] = x_start[zero_idx] # set x_{-1} as the gt mel
        return out

    def forward(self, mel, cond, spk_emb, mel_mask, coarse_mel=None, clip_denoised=True): # cond [2,737,256] mel [2,737,80] spk_emb None mel_mask [2,737]
        b, *_, device = *cond.shape, cond.device  # b:2
        x_t = x_t_prev = x_t_prev_pred = t = None
        mel_mask = ~mel_mask.unsqueeze(-1) # mel_mask [2,737,1]
        cond = cond.transpose(1, 2) # [2,256,737]
        self.cond = cond.detach() # detach 分离tensor，不再具有梯度
        self.spk_emb = spk_emb.detach() if spk_emb is not None else None
        if mel is None:
            if self.model != "shallow":
                noise = None
            else:
                t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
                noise = self.diffuse_fn(coarse_mel, t) * mel_mask.unsqueeze(-1).transpose(1, -1)
            x_0_pred = self.sampling(noise=noise)[-1] * mel_mask
        else:
            mel_mask = mel_mask.unsqueeze(-1).transpose(1, -1) # [2,737,1,1] -> [2,1,1,737]
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long() # self.num_timesteps=4   t [1, 1]

            # Diffusion
            x_t = self.diffuse_fn(mel, t) * mel_mask # 正向过程 mel [2,737,80] t [1,1]  > xt
            x_t_prev = self.diffuse_fn(mel, t - 1) * mel_mask  # mel [2,737,80] t[0,0] > xt-1

            # Predict x_{start}
            x_0_pred = self.denoise_fn(x_t, t, cond, spk_emb) * mel_mask  # 反向过程 [2,1,80,737]
            if clip_denoised:
                x_0_pred.clamp_(-1., 1.) # 限幅作用

            # Sample x_{t-1} using the posterior distribution
            if self.model != "shallow":
                x_start = x_0_pred # x_start = [2,1,80,737]
            else:
                x_start = self.norm_spec(coarse_mel)
                x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            x_t_prev_pred = self.q_posterior_sample(x_start=x_start, x_t=x_t, t=t) * mel_mask # [2,1,80,737]

            x_0_pred = x_0_pred[:, 0].transpose(1, 2)
            x_t = x_t[:, 0].transpose(1, 2)
            x_t_prev = x_t_prev[:, 0].transpose(1, 2)
            x_t_prev_pred = x_t_prev_pred[:, 0].transpose(1, 2)
        return x_0_pred, x_t, x_t_prev, x_t_prev_pred, t  # [2,737_80] [2,737,80] [2,737,80] [2,737,80]

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def out2mel(self, x):
        return x
