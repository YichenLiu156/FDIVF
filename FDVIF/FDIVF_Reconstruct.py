import os
import torchvision
from torchvision import transforms
from Model.BackBone import AE_Encoder, Ind_Decoder
from LossFuction.loss import ReconstructLoss
import torch.optim as optim
import torch.utils as utils
import torch
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
class FD_Reconstruction():
    def __init__(self):
        self.Encoder = AE_Encoder()
        self.Decoder = Ind_Decoder()

        self.transforms = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def average_loss(self, loss, Iter_per_epoch):
        return [sum(loss[i * Iter_per_epoch:(i + 1) * Iter_per_epoch]) / Iter_per_epoch for i in
                range(int(len(loss) / Iter_per_epoch))]

    def train_reconstruction(self, args):
        train_data_path = args.train_data_path

        result_path = args.result_path
        device = args.device
        batch_size = args.batch_size
        epochs_recon = args.recon_epochs
        lr_recon = args.recon_lr
        draw_loss = args.draw_loss
        recon_encoder_pretrain = args.recon_encoder_pretrain
        recon_decoder_pretrain = args.recon_decoder_pretrain
        if recon_encoder_pretrain is not None:
            self.Encoder.load_state_dict(torch.load(recon_encoder_pretrain)['weight'])
            self.Decoder.load_state_dict(torch.load(recon_decoder_pretrain)['weight'])
        Train_Image_Number = len(os.listdir(train_data_path + 'IR\\IR'))
        Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size
        root_VIS = train_data_path + 'VIS\\'
        root_IR = train_data_path + '\\IR\\'

        Data_VIS = torchvision.datasets.ImageFolder(root_VIS, transform=self.transforms)
        dataloader_VIS = utils.data.DataLoader(Data_VIS, batch_size, shuffle=False)

        Data_IR = torchvision.datasets.ImageFolder(root_IR, transform=self.transforms)
        dataloader_IR = utils.data.DataLoader(Data_IR, batch_size, shuffle=False)


        self.Encoder = self.Encoder.to(device)
        self.Decoder = self.Decoder.to(device)

        optimizer1 = optim.Adam(self.Encoder.parameters(), lr=lr_recon)
        optimizer2 = optim.Adam(self.Decoder.parameters(), lr=lr_recon)

        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, [epochs_recon // 3, epochs_recon // 3 * 2],
                                                          gamma=0.1)
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, [epochs_recon // 3, epochs_recon // 3 * 2],
                                                          gamma=0.1)
        r_loss = ReconstructLoss(args)

        print('============ Recon Training Begins ===============')
        loss_trains = []
        loss_recons = []
        loss_features = []
        lr_list1 = []
        lr_list2 = []

        for iteration in range(epochs_recon):
            self.Encoder.train()
            self.Decoder.train()

            iter_VIS = iter(dataloader_VIS)
            iter_IR = iter(dataloader_IR)
            for step in range(Iter_per_epoch):
                data_VIS, _ = next(iter_VIS)
                data_IR, _ = next(iter_IR)

                data_VIS = data_VIS.to(device)
                data_IR = data_IR.to(device)

                optimizer1.zero_grad()
                optimizer2.zero_grad()


                fv = self.Encoder(data_VIS)
                fi = self.Encoder(data_IR)
                feature_V_1, feature_V_2, feature_V_B, feature_V_D = fv
                feature_I_1, feature_I_2, feature_I_B, feature_I_D = fi
                rv = self.Decoder(feature_V_1, feature_V_2, feature_V_B, feature_V_D)
                ri = self.Decoder(feature_I_1, feature_I_2, feature_I_B, feature_I_D)
                loss, loss_feature, loss_Recon = r_loss(fv, fi, rv, ri, (data_VIS,data_IR))
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                los = loss.item()
                los_feature = loss_feature.item()
                los_Recon = loss_Recon.item()
                print('Epoch/step: %d/%d, loss: %.7f, lr: %f' % (
                    iteration + 1, step + 1, los, optimizer1.state_dict()['param_groups'][0]['lr']))

                # Save Loss
                loss_trains.append(los)
                loss_features.append(los_feature)
                loss_recons.append(los_Recon)
            scheduler1.step()
            scheduler2.step()
            lr_list1.append(optimizer1.state_dict()['param_groups'][0]['lr'])
            lr_list2.append(optimizer2.state_dict()['param_groups'][0]['lr'])

        torch.save({'weight': self.Encoder.state_dict(), 'epoch': epochs_recon},
                   os.path.join(result_path, 'Encoder_weight.pkl'))
        torch.save({'weight': self.Decoder.state_dict(), 'epoch': epochs_recon},
                   os.path.join(result_path, 'Decoder_weight.pkl'))


        if draw_loss:
            plt.figure(figsize=[12, 4])
            plt.subplot(1, 3, 1), plt.plot(self.average_loss(loss_trains, Iter_per_epoch)), plt.title('Loss')
            plt.subplot(1, 3, 2), plt.plot(self.average_loss(loss_features, Iter_per_epoch)), plt.title('loss_feature')
            plt.subplot(1, 3, 3), plt.plot(self.average_loss(loss_recons, Iter_per_epoch)), plt.title('loss_recon')

            plt.tight_layout()
            plt.savefig(os.path.join(result_path, 'recon_curve_per_epoch_control.png'))





