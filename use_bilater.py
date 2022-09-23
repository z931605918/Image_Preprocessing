import os
import argparse
from torchsummary import summary
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DnCNN
from dataset_PIV import prepare_data, Images_Dataset_folder, Images_Dataset_folder_Use ,Images_Dataset_folder_Use_timeresolved
from utils import *
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p, MSELoss,precies_recall,correlation
import torch.nn.functional as F
from visdom import Visdom
from Models import Network_both,Network_both_Res
import time
import torchvision.transforms as transforms
import torchvision
import re

import matplotlib.pyplot as plt
#from psnr_ssim_loss import psnr, SSIM
parser = argparse.ArgumentParser(description="Bilateral_CNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepar e_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--milestone", type=int, default=300, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
#parser.add_argument('--or_resume', default='D:\desktop\粒子质量增强\Bilateral-CNN\logname\Bilateral_out205.tar')  #在此加载基础模型
parser.add_argument('--or_resume', default='D:\desktop\粒子质量增强\Bilateral-CNN\logname_second\\Bilateral_res_330.tar',)  #在此加载基础模型
 #                   help='whether to reset moving mean / other hyperparameters')
parser.add_argument("--input_data_folder",type=str, default= 'H:\图像增强数据集\第四届挑战赛\前50',   #原始粒子图片
                    help=' original PIV image with low quality')
parser.add_argument('--save_folder',type=str, default='H:\图像增强数据集\\training_set4\low_light_back_gauss_input',  #加载输出目录
                    help='输出目录')
parser.add_argument("--label_data_folder",type=str, default= 'H:\image_enhancement\\training_set3\标签',
                     help=' optimized PIV image with high quality,输入目录')
time_resovled=False
opt = parser.parse_args()

def test():
    print('Loading dataset ...\n')
    test_transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
               # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    dataset_test = Images_Dataset_folder(opt.input_data_folder, opt.label_data_folder,transformI=test_transform,transformM=test_transform)
    # dataset_val = Images_Dataset_folder(opt.input_data_folder,opt.label_data_folder )
    loader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=16, shuffle=False,
                              drop_last=False, pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_test)))
    # Build model
    torch.cuda.set_device(0)
    model=Network_both_Res()
    # Move to GPU
    device_ids = [0]
    save_data = torch.load(opt.or_resume)
    state_dict = save_data['model_state_dict']
    model = model.cuda()  # nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    total_MSE_loss = 0.0
    total_psnr = 0.0
    total_SSIM = 0.0
    SSIM_loss=SSIM()
    with torch.no_grad():
        for i,(data,label) in  enumerate(loader_test):
            img_train1 = data[0].cuda()
            img_train2=data[1].cuda()
            img_label1=label[0].cuda()
            img_label2=label[1].cuda()
            out1, out2 = model(img_train1, img_train2)
            mse_loss = 0.5 * (MSELoss(out1, img_label1) + MSELoss(out2, img_label2))
            total_MSE_loss += mse_loss.detach().cpu().numpy()
            psnr_loss=(psnr(out1.cpu().detach().numpy(),img_label1.cpu().detach().numpy())+psnr(out2.cpu().detach().numpy(),img_label2.cpu().detach().numpy()))/2
            ssim_loss=(SSIM_loss(out1.cpu().detach(),img_label1.cpu().detach())+SSIM_loss(out2.cpu().detach(),img_label2.cpu().detach()))/2
            total_psnr+=psnr_loss
            total_SSIM+=ssim_loss
            print(i)
        mean_MSE_loss = total_MSE_loss / (len(loader_test))
        mean_pnsr = total_psnr / (len(loader_test))
        mean_SSIM = total_SSIM / (len(loader_test))
        print('mean_RMSE ={}, mean_pnsr= {}, mean_SSIM = {}'.format( mean_MSE_loss, mean_pnsr, mean_SSIM))
def Use():
    with torch.no_grad():
        model = Network_both_Res()
        #model = Network_both()  # nn.DataParallel(net, device_ids=device_ids).cuda()
        print('Loading dataset ...\n')
        if time_resovled:
            dataset_test=Images_Dataset_folder_Use_timeresolved(opt.input_data_folder)
        else:
            dataset_test = Images_Dataset_folder_Use(opt.input_data_folder)
        # dataset_val = Images_Dataset_folder(opt.input_data_folder,opt.label_data_folder )
        loader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=1, shuffle=False,
                                  drop_last=False, pin_memory=True)
        save_names =os.listdir( opt.input_data_folder)
        save_folder=opt.save_folder
        save_dirs=[]
        for i in save_names:
            save_dirs.append(os.path.join(save_folder,i))
        print("# of training samples: %d\n" % int(len(dataset_test)))
        # Build model
        torch.cuda.set_device(0)
        # net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
        # net.apply(weights_init_kaiming)

        # Move to GPU
        device_ids = [0]
        save_data = torch.load(opt.or_resume)
        state_dict = save_data['model_state_dict']

        model.load_state_dict(state_dict)
        model=model.cuda()
        model = model.eval()
        #summary(model, input_size=[(3, 256, 256),(3, 256, 256)], batch_size=2, device='cuda')
        t1=time.time()
        for batch_idx,(img1,img2) in enumerate(loader_test):
            img1=img1.cuda()
            img2=img2.cuda()
            out1,out2 = model(img1,img2)
        t2=time.time()
        print(t2-t1)
            # out1=out1.cpu()#/torch.max(out1)
            # out2=out2.cpu()#/torch.max(out2)
            # print(torch.max(out1))
            # out1[out1<0]=0
            # out1[out1>1]=1
            # out2[out2<0]=0
            # out2[out2>1]=1
            # out_img1=torchvision.transforms.ToPILImage()(out1[0])
            # out_img2=torchvision.transforms.ToPILImage()(out2[0])
            # out_img1.save(save_dirs[2*batch_idx])
            # out_img2.save(save_dirs[2*batch_idx+1])
            # print('process  %d / %d'%(batch_idx, len(loader_test)))


           #  plt.imshow(out1[0][0],cmap='gray',vmin=0,vmax=1)
           # # plt.savefig()
           #  plt.show()

   # print(t2-t1)
def main():
    Use()
if __name__=='__main__':
    main()

