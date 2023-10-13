from Train_Gen.train import train
from Train_Gen.transfer_test import advtest
#from Train_Gen.density_compute import advtest
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='ESMA')

parser.add_argument('--state', type=str, default='train_model', choices=['pretrain_embedding', 'train_model', 'advtest'],
                    help='Mode for model training or evaluation')
parser.add_argument('--Source_Model', type=str, default='ResNet50',
                    help='Source Model')
parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=25, help='Batch size')
parser.add_argument('--channel', type=int, default=32, help='Channel value')
parser.add_argument('--channel_mult', nargs='+', type=int, default=[1, 2, 3, 4], 
                    help='List of channel multipliers')
parser.add_argument('--num_res_blocks', type=int, default=1, help='Number of residual blocks')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for model training')
parser.add_argument('--Generator_save_dir', type=str, default='./ESMA_Checkpoints/', 
                    help='Directory to save checkpoints')
parser.add_argument('--training_load_weight', type=str, default='ckpt_pretrained_ResNet50_.pt', 
                    help='Weight file for training')
parser.add_argument('--test_load_weight', type=str, default='ckpt_299_ResNet50_.pt', 
                    help='Weight file for testing')
parser.add_argument('--q', type=int, default=2, help='q for screening')
args = parser.parse_args()


def main():
    if args.state == 'pretrain_embedding':
        ckpt = None
        epoch = 15000
    elif args.state == 'train_model':
        ckpt = args.training_load_weight
        epoch = args.epoch
    else:
        ckpt = args.test_load_weight
        epoch = args.epoch
    modelConfig = {
        "state": args.state,
        "Source_Model": args.Source_Model,
        "epoch": epoch,
        "batch_size": args.batch_size,
        "channel": args.channel,
        "channel_mult": args.channel_mult,
        "num_res_blocks": args.num_res_blocks,
        "lr": args.lr,
        "device": args.device,
        "training_load_weight": ckpt,
        "test_load_weight": args.test_load_weight,
        "Generator_save_dir": args.Generator_save_dir,
        "q": args.q
    }
    
    if modelConfig["state"] == "train_model" or modelConfig["state"] == "pretrain_embedding":
        
        train(modelConfig)
    elif modelConfig["state"] == "advtest":
        
        advtest(modelConfig)
        
if __name__ == '__main__':
    
    main()
    
    
