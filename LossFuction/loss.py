import torch
from torch import nn
from LossFuction.evalute import FrequentLoss, FusLoss, SSIM
class ReconstructLoss(nn.Module):
    def __init__(self, args):
        super(ReconstructLoss, self).__init__()
        device = args.device
        self.fre_loss = FrequentLoss()
        self.fre_loss = self.fre_loss.to(device)
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.L1Loss()
        self.ssim = SSIM()
        self.ssim =self.ssim.to(device)
    def forward(self, fv, fi, rv, ri, target):
        data_VIS,data_IR = target
        feature_V_1, feature_V_2, feature_V_B, feature_V_D = fv
        feature_I_1, feature_I_2, feature_I_B, feature_I_D = fi
        img_recon_VL, img_recon_VH, img_recon_V = rv
        img_recon_IL, img_recon_IH, img_recon_I = ri
        mse_loss_B = self.L1Loss(feature_I_B, feature_V_B)
        mse_loss_D = self.L1Loss(feature_I_D, feature_V_D)

        mse_loss_VF = 5 * self.ssim(data_VIS, img_recon_V) + self.MSELoss(data_VIS, img_recon_V)
        mse_loss_IF = 5 * self.ssim(data_IR, img_recon_I) + self.MSELoss(data_IR, img_recon_I)

        f_loss_V, _, _ = self.fre_loss(img_recon_VH, img_recon_VL, data_VIS)
        f_loss_I, _, _ = self.fre_loss(img_recon_IH, img_recon_IL, data_IR)
        loss = 2 * mse_loss_VF + 2 * mse_loss_IF + torch.tanh(mse_loss_B) - 0.5 * torch.tanh(
            mse_loss_D) + 2 * f_loss_V + 2 * f_loss_I

        loss_feature = torch.tanh(mse_loss_B) - 0.5 * torch.tanh(mse_loss_D)
        loss_Recon = mse_loss_VF + mse_loss_IF + f_loss_I + f_loss_V
        return loss, loss_feature, loss_Recon

class FusionLoss(nn.Module):
    def __init__(self, args):
        super(FusionLoss, self).__init__()
        device = args.device
        self.fus_loss = FusLoss()
        self.fus_loss = self.fus_loss.to(device)
        self.fre_loss = FrequentLoss()
        self.fre_loss = self.fre_loss.to(device)
        self.fus_loss = self.fus_loss.to(device)
        self.MSELoss = nn.MSELoss()
        self.ssim = SSIM()
        self.ssim =self.ssim.to(device)
        self.ssim = self.ssim.to(device)
    def forward(self, rv, ri, fus, target):
        data_VIS,data_IR = target

        img_recon_VL, img_recon_VH, img_recon_V = rv
        img_recon_IL, img_recon_IH, img_recon_I = ri
        total_loss, L_loss, H_loss = self.fus_loss(data_IR, data_VIS, fus)
        f_loss_V, _, _ = self.fre_loss(img_recon_VH, img_recon_VL, data_VIS)
        f_loss_I, _, _ = self.fre_loss(img_recon_IH, img_recon_IL, data_IR)
        mse_loss_VF = self.ssim(data_VIS, img_recon_V) + self.MSELoss(data_VIS, img_recon_V)
        mse_loss_IF = self.ssim(data_IR, img_recon_I) + self.MSELoss(data_IR, img_recon_I)
        mse_loss = 10 * mse_loss_VF + 10 * mse_loss_IF + 10 * f_loss_V + 10 * f_loss_I
        loss = total_loss + mse_loss
        return loss, L_loss, H_loss

