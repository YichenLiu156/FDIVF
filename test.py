
import argparse
from FDVIF.FDVIF_utils import FD_Test

parser = argparse.ArgumentParser(description='FDIVF_test')
parser.add_argument('--encoder_weights', type=str, default="Weights/Encoder_weight.pkl", help='encoder_weights')
parser.add_argument('--decoder_weights', type=str, default="Weights/Fusion_Decoder_weight.pkl", help='decoder_weights')
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--B_weights', type=str, default="Weights/Fusioner_B.pkl", help='B_weights')
parser.add_argument('--D_weights', type=str, default="Weights/Fusioner_D.pkl", help='D_weights')
parser.add_argument('--test_data_path', type=str, default="Datasets/Test_data_TNO", help='test_data_path')
parser.add_argument('--save_path', type=str, default="TestResult/", help='save_path')
if __name__ == '__main__':
    args = parser.parse_args()
    fd_test = FD_Test()
    fd_test.test_FDVIF(args)

