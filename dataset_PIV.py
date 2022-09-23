import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import torchvision
import PIL.Image as Image
#from utils import data_augmentation
from read_write_flo import read_flow

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

def load_data(path):
    all_image_pathes = os.listdir(path)
    image_vel_pathes = []
    for i in all_image_pathes:
        image_vel_path = os.path.join(path, i)
        image_vel_pathes.append(image_vel_path)
    image_pathes = image_vel_pathes
    image_pathes = list(filter(lambda x: x.endswith('tif') or x.endswith('bmp') or x.endswith('jpg') or x.endswith('png')   , image_pathes))

    img_group=[]

    for i in range(int(len(image_pathes)/2)):
        img_mini_group = []
        for j in range(2):
            img_mini_group.append(image_pathes[j+2*i])
        img_group.append(img_mini_group)

    return img_group


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir, transformI=None, transformM=None):
        img_group=load_data(images_dir)
        img_group.sort()
        label_group=load_data(labels_dir)
        label_group.sort()
        self.img_group=img_group
        self.label_group=label_group

        self.transformI = transformI
        self.transformM = transformM
        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation((-90, 90)),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
               # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation((-90, 90)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])
    def __len__(self):
        return len(self.img_group)
    def __getitem__(self, i):
        # print(self.labels[i])
        for j in range(1):
            if self.tx is not None:
                if self.tx is not None:
                    seed = random.randint(0, 100)
                    if seed>=50:
                        img1 = Image.open(self.img_group[i][j]).convert('RGB')
                        img2 = Image.open(self.img_group[i][j + 1]).convert('RGB')
                        label1=Image.open(self.label_group[i][j]).convert('RGB')
                        label2=Image.open(self.label_group[i][j+1]).convert('RGB')
                    if seed<50:
                        img1 = Image.open(self.img_group[i][j+1]).convert('RGB')
                        img2 = Image.open(self.img_group[i][j]).convert('RGB')
                        label1=Image.open(self.label_group[i][j+1]).convert('RGB')
                        label2=Image.open(self.label_group[i][j]).convert('RGB')

                    random.seed(seed)
                    torch.manual_seed(seed)
                    img1=self.tx(img1)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    img2=self.tx(img2)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    label1=self.lx(label1)[0].view(1,img1.shape[1],img1.shape[2])
                    random.seed(seed)
                    torch.manual_seed(seed)
                    label2=self.lx(label2)[0].view(1,img1.shape[1],img1.shape[2])
                    img1=torch.cat([img1,img1,img1],0)
                    img2 = torch.cat([img2, img2, img2], 0)

        return [img1,img2],[label1,label2]

class Images_Dataset_folder_3channle(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir, transformI=None, transformM=None):
        img_group=load_data(images_dir)
        img_group.sort()
        label_group=load_data(labels_dir)
        label_group.sort()
        self.img_group=img_group
        self.label_group=label_group

        self.transformI = transformI
        self.transformM = transformM
        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation((-90, 90)),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                #torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
               # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation((-90, 90)),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])
    def __len__(self):
        return len(self.img_group)
    def __getitem__(self, i):
        # print(self.labels[i])
        for j in range(1):
            if self.tx is not None:
                if self.tx is not None:
                    seed = random.randint(0, 100)
                    if seed>=50:
                        img1 = Image.open(self.img_group[i][j]).convert('RGB')
                        img2 = Image.open(self.img_group[i][j + 1]).convert('RGB')
                        label1=Image.open(self.label_group[i][j]).convert('RGB')
                        label2=Image.open(self.label_group[i][j+1]).convert('RGB')
                    if seed<50:
                        img1 = Image.open(self.img_group[i][j+1]).convert('RGB')
                        img2 = Image.open(self.img_group[i][j]).convert('RGB')
                        label1=Image.open(self.label_group[i][j+1]).convert('RGB')
                        label2=Image.open(self.label_group[i][j]).convert('RGB')

                    random.seed(seed)
                    torch.manual_seed(seed)
                    img1=self.tx(img1)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    img2=self.tx(img2)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    label1=self.lx(label1)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    label2=self.lx(label2)


        return [img1,img2],[label1,label2]

class Images_Dataset_folder_Use(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""
    def __init__(self, images_dir, transformM=None):
        self.img_group=load_data(images_dir)
        self.transformM = transformM
        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])
    def __len__(self):
        return len(self.img_group)
    def __getitem__(self, i):
        # print(self.labels[i])
        for j in range(1):
            if self.lx is not None:
                if self.lx is not None:
                    img1 = Image.open(self.img_group[i][j]).convert('RGB')
                    img2 = Image.open(self.img_group[i][j + 1]).convert('RGB')

                    np.random.seed(i)
                    img1=self.lx(img1)
                    img2=self.lx(img2)

                    #img1=torch.cat([img1,img1,img1],0)
                    #img2 = torch.cat([img2, img2, img2], 0)


        return img1,img2

class Images_Dataset_folder_Use_timeresolved(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""
    def __init__(self, images_dir, transformM=None):
        all_image_pathes = os.listdir(images_dir)
        image_vel_pathes = []
        for i in all_image_pathes:
            if i.endswith('.bmp'):
                image_vel_path = os.path.join(images_dir, i)
                image_vel_pathes.append(image_vel_path)
            elif i.endswith('tif'):
                image_vel_path = os.path.join(images_dir, i)
                image_vel_pathes.append(image_vel_path)
        image_pathes = image_vel_pathes
        self.img_group=image_pathes
        self.transformM = transformM
        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])
    def __len__(self):
        return int(len(self.img_group)/2)
    def __getitem__(self, i):
        # print(self.labels[i])
        if self.lx is not None:

            img1 = Image.open(self.img_group[2*i]).convert('RGB')
            img2 = Image.open(self.img_group[2*i+1]).convert('RGB')

            np.random.seed(i)
            img1=self.lx(img1)
            img2=self.lx(img2)

                #img1=torch.cat([img1,img1,img1],0)
                #img2 = torch.cat([img2, img2, img2], 0)


        return img1,img2
