import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from math import exp

class FrequentLoss(nn.Module):
    def __init__(self, wavelet='haar', levels=1, alpha_low=1.0, alpha_high=1.0):
        super().__init__()
        self.dwt = DWTForward(wave=wavelet, J=levels)  # 二级小波分解
        self.ssim = SSIM()  # 使用自定义SSIM模块
        self.alpha_low = alpha_low  # 低频损失权重

        self.alpha_high = alpha_high  # 高频损失权重

    def decompose(self, img):
        """执行二级小波分解，返回低频、中频、高频分量"""
        # 第一级分解
        Yl1, Yh1 = self.dwt(img)  # Yl1:低频, Yh1:高频(HL1, LH1, HH1)

        # 第二级分解
        Yl2, Yh2 = self.dwt(Yl1)  # Yl2:二级低频, Yh2:二级高频

        # 组合频段
        low_freq = Yl1  # 低频分量
        mid_freq = Yh2[0]  # 中频分量
        high_freq = Yh1[0].mean(dim=2)   # 高频分量

        return low_freq, mid_freq, high_freq

    def forward(self, high, low, target):
        assert high.dim() == 4 and low.dim() == 4 and target.dim() == 4 and target.dim() == 4, "输入必须为4D张量[B,C,H,W]"
        # 分解各图像


        t_low, t_mid, t_high = self.decompose(target)


        # 计算低频损失（融合低频 vs 红外低频）
        low_loss = 1 - self.ssim(low, t_low)

        # 计算高频损失
        high_loss = 1 - self.ssim(high, t_high)

        # 加权总损失
        total_loss = (self.alpha_low * low_loss +
                      self.alpha_high * high_loss)

        return total_loss, self.alpha_low * low_loss, self.alpha_high * high_loss

class FusLoss(nn.Module):
    def __init__(self, wavelet='haar', levels=1, alpha_low=1.0, alpha_high=1.0):
        super().__init__()
        self.dwt = DWTForward(wave=wavelet, J=levels)  # 二级小波分解
        self.ssim = SSIM()  # 使用自定义SSIM模块
        self.alpha_low = alpha_low  # 低频损失权重

        self.alpha_high = alpha_high  # 高频损失权重
        self.l1 = nn.L1Loss()

    def decompose(self, img):
        """执行二级小波分解，返回低频、中频、高频分量"""
        # 第一级分解
        Yl1, Yh1 = self.dwt(img)  # Yl1:低频, Yh1:高频(HL1, LH1, HH1)

        # 第二级分解
        Yl2, Yh2 = self.dwt(Yl1)  # Yl2:二级低频, Yh2:二级高频

        # 组合频段
        low_freq = Yl1  # 低频分量
        mid_freq = Yh2[0]  # 中频分量
        high_freq = Yh1[0].mean(dim=2)   # 高频分量

        return low_freq, mid_freq, high_freq

    def forward(self, ir, vis, fus):
        assert ir.dim() == 4 and vis.dim() == 4 and fus.dim() == 4, "输入必须为4D张量[B,C,H,W]"
        # 分解各图像


        i_low, i_mid, i_high = self.decompose(ir)
        v_low, v_mid, v_high = self.decompose(vis)
        f_low, f_mid, f_high = self.decompose(fus)



        # 计算低频损失（融合低频 vs 红外低频）
        low_loss = (1 - self.ssim(f_low, i_low)/2)+(1 - self.ssim(f_low, v_low)/2) + self.l1(f_low, i_low) + self.l1(f_low, v_low)

        # 计算高频损失
        high_loss =  2*(1 - self.ssim(f_high, i_high)/2) + (1 - self.ssim(f_high, v_high)/2) + self.l1(f_high, i_high)
        # high_loss = self.l1(f_high,v_high) + self.l1(f_high, i_high)

        # 加权总损失
        total_loss = (self.alpha_low * low_loss +
                      self.alpha_high * high_loss)

        return total_loss, self.alpha_low * low_loss, self.alpha_high * high_loss
# SSIM模块实现
class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._create_window())

    def _gaussian_weights(self):
        gauss = torch.exp(-(torch.arange(self.window_size) - self.window_size // 2) ** 2 / (2 * self.sigma ** 2))
        return gauss / gauss.sum()

    def _create_window(self):
        # 创建适用于批量处理的窗口
        _1D = self._gaussian_weights()
        _2D = _1D.unsqueeze(1) @ _1D.unsqueeze(0)
        return _2D.unsqueeze(0).unsqueeze(0)  # 形状: [1,1,H,W]

    def forward(self, img1, img2):
        # 输入形状: [B, C, H, W]
        B, C, H, W = img1.size()
        window = self.window.expand(C, 1, self.window_size, self.window_size)

        # 批量卷积计算
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=C) - mu1_mu2

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # 批量SSIM计算
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # 按样本计算均值（保持批量维度）
        return ssim_map.view(B, -1).mean(dim=1).mean()  # 先对每个样本的空间和通道平均，再对batch平均