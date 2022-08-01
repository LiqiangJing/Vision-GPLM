import argparse
from argparse import Namespace
def get_args():
    parser = argparse.ArgumentParser(description='Style Transfer.')
    parser.add_argument('--use_gpu', default=True, type=bool, help="Use CUDA on the device.")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)

    parser.add_argument('--pretrained_model', type=str, default='./bart-base')
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--Epoch', type=int, default=200)

    parser.add_argument('--max_length_src', type=int, default=50)
    parser.add_argument('--max_length_tgt', type=int, default=50)

    parser.add_argument('--save_path', type=str, default='./save/')

    parser.add_argument('--load', type=str, default=None)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--log_file', type=str, default='log')
    parser.add_argument('--model_name', type=str, default='multi')

    args = parser.parse_args()
    return args
