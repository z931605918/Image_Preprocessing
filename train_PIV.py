import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader

from dataset_PIV import prepare_data, Images_Dataset_folder,Images_Dataset_folder_3channle
from utils import *
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p, MSELoss,precies_recall,correlation,Correlation_loss

from visdom import Visdom
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Models_New2 import Network_both_Res

import matplotlib.pyplot as plt
#python -m visdom.server
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepar e_data or not')
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=300, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--input_data_folder",type=str, default= 'H:\image_enhancement\Testing_set4\low_gauss_back_light_2',   #输入文件夹
                    help=' original PIV image with low quality')
parser.add_argument("--label_data_folder",type=str, default= 'H:\image_enhancement\Testing_set4\high',   #标签文件夹
        help=' optimized PIV image with high quality')
parser.add_argument('--or_resume', default='D:\desktop\粒子质量增强\Bilateral-CNN\logname_third\\Bilateral_res_340.tar',  #在此加载模型
                    help='whether to reset moving mean / other hyperparameters')
show=False
opt = parser.parse_args()
start_epoch=340
save_dir='D:\desktop\粒子质量增强\Bilateral-CNN\logname_third'
def show_out(out1):
    out_show = out1[0][0] * 255
    out_show = out_show.detach().cpu().numpy()
    out_show[out_show > 255] = 255
    out_show[out_show < 0] = 0
    return out_show
def main():
    # Load dataset
    print('Loading dataset ...\n')
    #dataset_train = Images_Dataset_folder(opt.input_data_folder,opt.label_data_folder )
    dataset_train = Images_Dataset_folder(opt.input_data_folder, opt.label_data_folder)

    #dataset_val = Images_Dataset_folder(opt.input_data_folder,opt.label_data_folder )
    loader_train = DataLoader(dataset=dataset_train, batch_size=opt.batchSize, shuffle=True,drop_last=True,pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    torch.cuda.set_device(0)
    # net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    # net.apply(weights_init_kaiming)
    #criterion = nn.MSELoss(size_average=False)
    criterion=Correlation_loss()
    MSE_loss_class = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = Network_both_Res()#nn.DataParallel(net, device_ids=device_ids).cuda()
    model.apply(weights_init_kaiming)
    if opt.or_resume is not None :
        save_data=torch.load(opt.or_resume)
        state_dict=save_data['model_state_dict']
        model.load_state_dict(state_dict)
        # start_mse=save_data['train_MSE_loss'][-1]
        # start_precise=save_data['train_precies'][-1]
        # start_recall=save_data['train_recall'][-1]
        # start_cor=save_data['train_cor'][-1]
        optimizer_state_dict=save_data['optimizer_state_dict']
        #optimizer_state_dict=optimizer_state_dict.cuda()
    else:
        start_mse=0
        start_precise=0
        start_recall=0
        start_cor=0
        optimizer_state_dict=None
    # Optimizer

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer.load_state_dict(optimizer_state_dict)
    # for state in optimizer.state.values():
    #     for k,v in state.items():
    #         if torch.is_tensor(v):
    #             state[k]=v.cuda()




#    optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.5*opt.lr)
    #scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8, patience=30, threshold=1e-5,min_lr=5e-7)
    # training
    #writer = SummaryWriter(opt.outf)
    step = 0
    train_MSE_loss=[]
    train_cross_loss=[]
    train_precies=[]
    train_recall=[]
    train_multi_loss=[]
    train_cor=[]
    viz_1 = Visdom()

    #viz_1.line([[50*start_mse,start_cor, start_precise]], [start_epoch], win='train',opts=dict(title='DnCNN', legend=['50*mse', 'correlation','precies']))
    for epoch in range(start_epoch+1,opt.epochs):
        sum_multi_loss=0.0
        crossen_loss = 0.0
        total_MSE_loss = 0.0
        sum_precies = 0.0
        sum_recall = 0.0
        sum_cor=0.0

        #
        current_lr=scheduler.optimizer.param_groups[0]['lr']
        # train
        for i,(data,label) in enumerate(loader_train):
            # training step
            t1=time.time()
            model.cuda()
            model.train()
            optimizer.zero_grad()
            img_train1 = data[0].cuda()
            img_train2=data[1].cuda()
            img_label1=label[0].cuda()
            img_label2=label[1].cuda()
            out1,out2,both = model(img_train1,img_train2)
            #out1=model(img_train1)
            if show:
                out=out1[0,0].detach().cpu().numpy()
                label=img_label1[0,0].detach().cpu().numpy()
                inputs=img_train1[0,0].detach().cpu().numpy()
                plt.subplot(1,3,1)
                plt.imshow(out,vmin=0,vmax=1,cmap='gray')
                plt.subplot(1,3,2)
                plt.imshow(inputs,vmin=0,vmax=1,cmap='gray')
                plt.subplot(1,3,3)
                plt.imshow(label,vmin=0,vmax=1,cmap='gray')
                #plt.colorbar(shrink=0.8)
                plt.show()
            # img_label1=Variable(img_label1,requires_grad=True)
            # img_label2=Variable(img_label2,requires_grad=True)
            loss1 = criterion(out1, img_label1)
            loss2= criterion(out2,img_label2)
            loss=0.5*(loss1+loss2)
            #loss.backward()
            calc_loss_total=(calc_loss(out1,img_label1,bce_weight=0.5))#+calc_loss(out2,img_label2,bce_weight=0.5))
           # calc_loss_total.backward()
            #opt_loss=0.5*loss+0.5*calc_loss_total
            #opt_loss.backward()
            mse_loss=0.5*(MSELoss(out1,img_label1)+MSELoss(out2,img_label2))
            #mse_loss=MSELoss(out1,img_label1)
            total_MSE_loss +=mse_loss.detach().cpu().numpy()
            crossen_loss+=calc_loss_total.detach().cpu().numpy()
            # pre_0_1=torch.sigmoid(out_train*255)
            # label_0_1=torch.sigmoid(label*255)
            precies1, recall1= precies_recall(out1,img_label1)
            precies2, recall2 = precies_recall(out2, img_label2)
            precies=0.5*(precies1+precies2)
            recall=0.5*(recall1+recall2)
            sum_precies += precies
            sum_recall += recall
            cor1=correlation(out1,img_label1).detach().cpu().numpy()
            cor2=correlation(out2,img_label2).detach().cpu().numpy()
            cor_loss=1-0.5*(cor1+cor2)

            backward_loss = mse_loss# + 5*cor_loss
            backward_loss.backward()
            sum_cor+=cor_loss
            optimizer.step()
            # results
            step += 1
            t2=time.time()
            # viz_1.image(img=img_train1[0][0], win='input', opts=dict(title='input'))
            # viz_1.image(img=img_label1[0][0], win='label', opts=dict(title='label'))
            # viz_1.image(img=out1[0], win='out', opts=dict(title='out'))
            #
            # out1[out1>1]=1
            # a


            if i%10==0:
                print('epoch=%d ,index=%d/%d cor_loss=%.5f, mse=%.5f, learning rate*10000=%f, per_group_cost %.2fs' % (
                    epoch,
                    i,
                    len(loader_train),
                    cor_loss,
                    mse_loss.detach().cpu().numpy(),
                    current_lr*1e5,
                    t2-t1
                ))

                viz_1.image(img=show_out(out1), win='out1', opts=dict(title='out1',vmax=255,vmin=0))
                viz_1.image(img=show_out(img_label1), win='label1', opts=dict(title='label1',vmax=255,vmin=0))
                viz_1.image(img=show_out(img_train1), win='input1', opts=dict(title='input1',vmax=255,vmin=0))
                viz_1.image(img=show_out(abs(out1-img_label1)), win='out-label1', opts=dict(title='out-label1',vmax=100,vmin=0))
                viz_1.image(img=show_out(out2), win='out2', opts=dict(title='out2',vmax=255,vmin=0))
                viz_1.image(img=show_out(img_label2), win='label2', opts=dict(title='label2',vmax=255,vmin=0))
                viz_1.image(img=show_out(img_train2), win='input2', opts=dict(title='input2',vmax=255,vmin=0))
                viz_1.image(img=show_out(abs(out2-img_label2)), win='out-label2', opts=dict(title='out-label2',vmax=100,vmin=0))
                viz_1.image(img=show_out(both), win='both',
                            opts=dict(title='both', vmax=100, vmin=0))




        if show:
            data = out1[0, 0].detach().cpu().numpy()
            label = img_label1[0, 0].detach().cpu().numpy()
            plt.subplot(1, 2, 1)
            plt.imshow(data, vmin=0, vmax=1, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(label, vmin=0, vmax=1, cmap='gray')
            plt.colorbar()
            plt.show()
        scheduler.step(epoch)
        ## the end of each epoch
        mean_MSE_loss=total_MSE_loss/(len(loader_train))
        mean_cross_loss=crossen_loss/(len(loader_train))
        mean_cor=sum_cor/(len(loader_train))
        train_MSE_loss.append(mean_MSE_loss)
        train_cross_loss.append(mean_cross_loss)
        train_cor.append(mean_cor)
        torch.cuda.empty_cache()

        print('epoch=%d mean_cor_loss=%.5f, mean_rmse=%.5f'% (
            epoch,
            mean_cor,
            mean_MSE_loss,
        ))
        if (epoch) % 1 == 0:
            model.eval()
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch,
                          'train_MSE_loss':train_MSE_loss,
                          'train_cross_loss':train_cross_loss,
                          'train_cor':train_cor
                          }
            path_checkpoint = save_dir+'\\Bilateral_res_%03d.tar' % epoch
            torch.save(checkpoint, path_checkpoint)
            print(epoch)
        viz_1.line([[ mean_cor,5*mean_MSE_loss,current_lr*1000]], [epoch],
                    win='train_Bilater_res_low', update='append',opts=dict(title='Bila_res3', legend=[ 'corelation','5*mse','lr*1000']))






if __name__ == "__main__":
    main()
