import os
import torchvision
from torchvision import transforms
from Model.BackBone import AE_Encoder, Ind_Decoder
from Model.Fusioner import Fusioner_D, Fusioner_B
from LossFuction.loss import FusionLoss
import torch.optim as optim
import torch.utils as utils
import torch
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
class FD_Fusion():
    def __init__(self):
        self.Encoder = AE_Encoder()
        self.Decoder = Ind_Decoder()
        self.F_D = Fusioner_D()
        self.F_B = Fusioner_B()


        self.transforms = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def Average_loss(self,loss, Iter_per_epoch):
        return [sum(loss[i * Iter_per_epoch:(i + 1) * Iter_per_epoch]) / Iter_per_epoch for i in
                range(int(len(loss) / Iter_per_epoch))]

    def train_Fusion(self, args):
        train_data_path = args.train_data_path

        result_path = args.result_path
        device = args.device
        batch_size = args.batch_size
        epochs_fuse =  args.fuse_epochs
        lr_fuse = args.fuse_lr
        draw_loss = args.draw_loss
        fusion_encoder_pretrain = args.fusion_encoder_pretrain
        fusion_decoder_pretrain = args.fusion_decoder_pretrain
        if fusion_encoder_pretrain is not None:
            self.Encoder.load_state_dict(torch.load(fusion_encoder_pretrain)['weight'])
            self.Decoder.load_state_dict(torch.load(fusion_decoder_pretrain)['weight'])
        Train_Image_Number = len(os.listdir(train_data_path + 'IR\\IR'))
        Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size
        root_VIS = train_data_path + 'VIS\\'
        root_IR = train_data_path + '\\IR\\'

        optimizer1 = optim.Adam(self.Encoder.parameters(), lr=lr_fuse)
        optimizer2 = optim.Adam(self.Decoder.parameters(), lr=lr_fuse)

        Data_VIS = torchvision.datasets.ImageFolder(root_VIS, transform=self.transforms)
        dataloader_VIS = utils.data.DataLoader(Data_VIS, batch_size, shuffle=False)

        Data_IR = torchvision.datasets.ImageFolder(root_IR, transform=self.transforms)
        dataloader_IR = utils.data.DataLoader(Data_IR, batch_size, shuffle=False)


        self.Encoder = self.Encoder.to(device)
        self.Decoder = self.Decoder.to(device)
        self.F_D = self.F_D.to(device)
        self.F_B = self.F_B.to(device)
        f_loss = FusionLoss(args)


        optimizer = optim.Adam(self.Decoder.parameters(), lr=lr_fuse)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs_fuse // 3, epochs_fuse // 3 * 2],
                                                          gamma=0.1)




        print('============ Recon Training Begins ===============')
        fus_loss_train = []
        fus_loss_L = []
        fus_loss_H = []
        lr_list = []

        alpha_list = []
        for iteration in range(epochs_fuse):
            self.Encoder.eval()
            self.Decoder.train()
            self.F_B.train()
            self.F_D.train()

            iter_VIS = iter(dataloader_VIS)
            iter_IR = iter(dataloader_IR)
            for step in range(Iter_per_epoch):
                data_VIS, _ = next(iter_VIS)
                data_IR, _ = next(iter_IR)

                data_VIS = data_VIS.to(device)
                data_IR = data_IR.to(device)

                optimizer.zero_grad()



                fv = self.Encoder(data_VIS)
                fi = self.Encoder(data_IR)
                feature_V_1, feature_V_2, feature_V_B, feature_V_D = fv
                feature_I_1, feature_I_2, feature_I_B, feature_I_D = fi
                F_b = self.F_B(feature_I_B, feature_V_B)
                F_d = self.F_D(feature_I_D, feature_V_D)
                F_1 = (feature_I_1 + feature_V_1)
                F_2 = (feature_I_2 + feature_V_2)
                rv = self.Decoder(feature_V_1, feature_V_2, feature_V_B, feature_V_D)
                ri = self.Decoder(feature_I_1, feature_I_2, feature_I_B, feature_I_D)
                _, _, fus = self.Decoder(F_1, F_2, F_b, F_d)
                loss, L_loss, H_loss = f_loss(rv, ri, fus, (data_VIS,data_IR))
                loss.backward()
                optimizer.step()

                los = loss.item()
                los_H = H_loss.item()
                los_L = L_loss.item()
                print('Epoch/step: %d/%d, loss: %.7f, lr: %f' % (
                    iteration + 1, step + 1, los, optimizer1.state_dict()['param_groups'][0]['lr']))

                # Save Loss
                fus_loss_train.append(los)
                fus_loss_H.append(los_H)
                fus_loss_L.append(los_L)
            scheduler.step()

            lr_list.append(optimizer1.state_dict()['param_groups'][0]['lr'])



        torch.save({'weight': self.Decoder.state_dict(), 'epoch': epochs_fuse},
                   os.path.join(result_path, 'Fusion_Decoder_weight.pkl'))
        torch.save({'weight': self.F_D.state_dict(), 'epoch': epochs_fuse},
                   os.path.join(result_path, 'Fusioner_D.pkl'))
        torch.save({'weight': self.F_B.state_dict(), 'epoch': epochs_fuse},
                   os.path.join(result_path, 'Fusioner_B.pkl'))

        if draw_loss:
            plt.figure(figsize=[12, 4])
            plt.subplot(1, 3, 1), plt.plot(self.Average_loss(fus_loss_train,Iter_per_epoch)), plt.title('Loss')
            plt.subplot(1, 3, 2), plt.plot(self.Average_loss(fus_loss_H,Iter_per_epoch)), plt.title('fus_loss_H')
            plt.subplot(1, 3, 3), plt.plot(self.Average_loss(fus_loss_L,Iter_per_epoch)), plt.title('fus_loss_L')

            plt.tight_layout()
            plt.savefig(os.path.join(result_path, 'fus_curve_per_epoch_control.png'))





