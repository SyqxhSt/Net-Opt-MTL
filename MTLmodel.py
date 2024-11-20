import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from create_dataset import *
from utils import *

from VMSANet import *

from imtl import IMTL
import itertools
import os


# By configuring the argpars module, parameters can be passed in the terminal command line in order
parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--method', default='pcgrad', type=str, help='which optimization algorithm to use')

opt = parser.parse_args()


# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VMSANet_MTAN = VMSANet().to(device)

'''
# imtl algorithm
imt = IMTL('gradient').to(device)
paraments = itertools.chain(VMSANet_MTAN.parameters(), imt.parameters())
optimizer = torch.optim.Adam(paraments, lr=1e-3)        # When using the imtl algorithm, it is necessary to adjust the VMS ANet_maTAN. parameters() parameter to paraments
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
'''

optimizer = optim.Adam(VMSANet_MTAN.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # Adjust the learning rate in the optimizer

'''
optimizer = torch.optim.Adam(VMSANet_MTAN.parameters(), lr=1e-3)        # multi_task_trainer_qt
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)        # multi_task_trainer_qt
'''

print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(VMSANet_MTAN),count_parameters(VMSANet_MTAN) / 24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset
dataset_path = opt.dataroot  # '--dataroot', default='nyuv2' Default is 'nyuv2'
if opt.apply_augmentation:  # '--apply_augmentation', action='store_true'，action='store_true’ If we configure this parameter on the command line, it will be True; If not configured, it defaults to False
    mtl_train_set = Datasets_MTL(root=dataset_path, train=True, augmentation=True)
    print('Applying data augmentation on Datasets_MTL.')
else:
    mtl_train_set = Datasets_MTL(root=dataset_path, train=True)
    print('Standard training strategy without data augmentation.')  # No data augmentation

mtl_test_set = Datasets_MTL(root=dataset_path, train=False)

batch_size = 8
train_loader = torch.utils.data.DataLoader(
    dataset = mtl_train_set,
    batch_size = batch_size,
    shuffle = True)

test_loader = torch.utils.data.DataLoader(
    dataset = mtl_test_set,
    batch_size = batch_size,
    shuffle = False)


# Common algorithms and gradvac_ (imtl), imtl
# Train and evaluate multi-task network
multi_task_trainer(train_loader,test_loader,VMSANet_MTAN,device,optimizer,scheduler,opt,200)

# multi_task_trainer_AdaCorr(train_loader,test_loader,VMSANet_MTAN,device,optimizer,scheduler,opt,200)

# Other algorithms pcgrad、cagrad、......
# multi_task_trainer_qt(train_loader,test_loader,VMSANet_MTAN,device,optimizer,scheduler,opt,200)
