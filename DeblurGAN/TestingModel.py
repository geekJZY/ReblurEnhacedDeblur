
# coding: utf-8

# In[7]:


import time
import os
import torch
import sys
from options.test_options import TestOptions
from data.test_dataset import testDataSet

sys.argv += ["--dataroot" ,"/scratch/user/jiangziyu/test/" ,"--model" ,"test" ,"--dataset_mode" ,"single" ,"--learn_residual" ,"--resize_or_crop" ,"False"]

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_set = testDataSet()
data_set.initialize(opt)
data_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))


# In[8]:


from models.test_model_cal import TestModel
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from skimage.measure import compare_psnr, compare_ssim
from PIL import Image

# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

model = TestModel()
model.initialize(opt)

for i, data in enumerate(data_loader):
    if i >= opt.how_many:
        break
    counter = i+1
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    avgPSNR += compare_psnr(visuals['real_A'],visuals['fake_B'])
    avgSSIM += compare_ssim(visuals['real_A'],visuals['fake_B'],multichannel = True)
    print('process image... %s' % str(i))
    if(counter % 50 == 0):
        print('PSNR = %f, SSIM = %f' % (avgPSNR/counter, avgSSIM/counter))

avgPSNR /= counter
avgSSIM /= counter
print('PSNR = %f, SSIM = %f' % (avgPSNR, avgSSIM))

