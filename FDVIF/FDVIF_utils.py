import numpy as np
import torch
import os

from FDVIF.FDIVF_Fusion import FD_Fusion
from Model.BackBone import AE_Encoder, Ind_Decoder
from Model.Fusioner import Fusioner_D, Fusioner_B
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import cv2
class FD_Test():
    def __init__(self):

        self.Encoder = AE_Encoder()
        self.Decoder = Ind_Decoder()
        self.F_D = Fusioner_D()
        self.F_B = Fusioner_B()

    def output_img(self, x):
        return x.cpu().detach().numpy()[0, 0, :, :]

    def test_FDVIF(self, args):
        encoder_weights = args.encoder_weights
        decoder_weights = args.decoder_weights
        D_weights = args.D_weights
        B_weights = args.B_weights
        device = args.device
        test_data_path = args.test_data_path
        save_path = args.save_path
        self.Encoder.load_state_dict(torch.load(encoder_weights)['weight'])
        self.Decoder.load_state_dict(torch.load(decoder_weights)['weight'])
        self.F_D.load_state_dict(torch.load(D_weights)['weight'])
        self.F_B.load_state_dict(torch.load(B_weights)['weight'])

        self.Encoder = self.Encoder.to(device)
        self.Decoder = self.Decoder.to(device)
        self.F_D = self.F_D.to(device)
        self.F_B = self.F_B.to(device)

        self. Encoder.eval()
        self.Decoder.eval()
        self.F_B.eval()
        self.F_D.eval()

        Test_Image_Number = len(os.listdir(test_data_path))
        for i in range(int(Test_Image_Number / 2)):
            Test_IR = Image.open(test_data_path + '\IR' + str(i + 1) + '.bmp')  # infrared image
            Test_Vis = Image.open(test_data_path + '\VIS' + str(i + 1) + '.bmp')  # visible image
            img_test1 = np.array(Test_IR, dtype='float32') / 255  # 将其转换为一个矩阵
            img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))

            img_test2 = np.array(Test_Vis, dtype='float32') / 255  # 将其转换为一个矩阵
            img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))

            img_test1 = img_test1.cuda()
            img_test2 = img_test2.cuda()
            with torch.no_grad():
                F_i1, F_i2, F_ib, F_id = self.Encoder(img_test1)
                F_v1, F_v2, F_vb, F_vd = self.Encoder(img_test2)
                F_b = self.F_B(F_ib, F_vb)
                F_d = self.F_D(F_id, F_vd)
                F_1 = (F_i1 + F_v1)
                F_2 = (F_i2 + F_v2)
                _, _, fus = self.Decoder(F_1, F_2, F_b, F_d)
            fus_img = self.output_img(fus)
            cv2.imwrite(save_path + str(i + 1) + '.jpg', fus_img * 225)





