from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,  # 特征维度
        cfm_params,  # 连续流匹配参数
        n_spks=1,  # 说话人数量
        spk_emb_dim=128,  # 说话人嵌入维度
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature  # 生成初始噪声，shape: [batch_size, n_feats, mel_timesteps]，如[1, 80, 848]
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)  # 创建时间步序列，理论上n_timesteps越大，结果越好，耗时越长
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)  # 使用欧拉求解器从噪声生成目标音频特征

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.  # 固定步长的欧拉求解器
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)  # 当前时间步Flow对时间t的倒数，就是向量场；shape不变，如[1, 80, 848]

            x = x + dt * dphi_dt  # 基于初始噪声进行欧拉更新步骤；shape不变，如[1, 80, 848]
            t = t + dt  # 更新时间
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t  # 更新步长；适合步长非固定的情况

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target，目标mel谱图
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask，目标掩码
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)  # 从先验分布中采样的随机噪声

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1  # 使用基于OT-CFM推导出的流的表达式计算出时间t时刻的流
        u = x1 - (1 - self.sigma_min) * z  # 因为是对ODE建模，目标向量场u就是流对时间t的导数，此行计算的u就是t时刻的目标向量场
        # y是流，mu_y是encoder预测的mel谱图，t是时间步，spks是说话人嵌入，基于estimator，也就是decoder预测出t时刻的向量场，然后与目标向量场u计算损失
        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
