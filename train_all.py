
import argparse
import torch
from FDVIF.FDIVF_Reconstruct import FD_Reconstruction
from FDVIF.FDIVF_Fusion import FD_Fusion
parser = argparse.ArgumentParser(description='FDIVF_train')
parser.add_argument('--train_data_path', type=str, default='.\\Datasets\\Train_data_FLIR\\', help='train_data_path')
parser.add_argument('--result_path', type=str, default='.\\TrainResult\\', help='device')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size')
parser.add_argument('--recon_epochs', type=int, default=20, help='recon_epochs')
parser.add_argument('--fuse_epochs', type=int, default=60, help='fuse_epochs')
parser.add_argument('--fuse_lr', type=float, default=1e-3, help='fuse_lr')
parser.add_argument('--recon_lr', type=float, default=1e-3, help='recon_lr')
parser.add_argument('--draw_loss', type=bool, default=True, help='draw_loss')
parser.add_argument('--recon_encoder_pretrain', type=str, default=None, help='recon_encoder_pretrain')
parser.add_argument('--recon_decoder_pretrain', type=str, default=None, help='recon_decoder_pretrain')
parser.add_argument('--fusion_encoder_pretrain', type=str, default="TestResult/Encoder_weight.pkl", help='fusion_encoder_pretrain')
parser.add_argument('--fusion_decoder_pretrain', type=str, default="TestResult/Decoder_weight.pkl", help='fusion_decoder_pretrain')
if __name__ == '__main__':
    args = parser.parse_args()
    fd_r = FD_Reconstruction()
    fd_r.train_reconstruction(args)
    fd_f = FD_Fusion()
    fd_f.train_Fusion(args)
