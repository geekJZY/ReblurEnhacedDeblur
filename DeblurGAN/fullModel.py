
# In[1]:


import time
import sys
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from skimage.measure import compare_psnr


# ## Import dataloader and show it

# In[2]:

sys.argv += ['--dataroot', '/scratch/user/jiangziyu/GOPRO_for_reblurGAN/',
             '--learn_residual', '--resize_or_crop', 'scale_width','--fineSize', '256']

opt = TrainOptions().parse()


# In[3]:


#get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from data.aligned_dataset import AlignedDataset

dataset = AlignedDataset()
dataset.initialize(opt)


# In[4]:


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


# ## define model and load pretrained weights

# In[5]:


import os
import torch
from torch.autograd import Variable
from collections import OrderedDict
from models import networks

def load_network(network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
        network.load_state_dict(torch.load(save_path))
def save_network(network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda()

netG_deblur = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.gpu_ids, False,
                                      opt.learn_residual)
netG_blur = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.gpu_ids, False,
                                      opt.learn_residual)

load_network(netG_deblur, 'deblur_G', opt.which_epoch)
load_network(netG_blur, 'blur_G', opt.which_epoch)
print('------- Networks deblur_G initialized ---------')
networks.print_network(netG_deblur)
print('-----------------------------------------------')


# ### Freeze layers

# In[6]:


def fine_tune_existing_layers(model,num_layers_frozen=19):

    ct=0
    for child in list(list(netG_deblur.children())[0].children())[0]:
        ct+=1
        if ct<num_layers_frozen:
            for param in child.parameters():
                param.requires_grad=False


    print("Total number of layers are:",ct,",number of layers frozen are:", num_layers_frozen)
    return model

netG_frozen_deblur= fine_tune_existing_layers(netG_deblur, num_layers_frozen=21)
netG_frozen_blur= fine_tune_existing_layers(netG_blur, num_layers_frozen=21)


# ### Net training parameters

# In[8]:


num_epoch=100
batch_size=8
num_workers=2
learning_rate=0.0002
transforms=None       #make data augmentation. For now using only the transforms defined above
results_file_path="./results/experiment_name/full_model_results/"

# ### Cycle consistency loss

# In[9]:


import itertools
import util.util as util
import numpy as np

"""Quote from the paper about the loss function: For all the experiments, we set λ = 10 in Equation 3.
We use the Adam solver [24] with a batch size of 1"""

cycle_consistency_criterion= torch.nn.L1Loss()

#criterion= forward_cycle_consistency_criterion+backward_cycle_consistency_criterion()

#lambda_cycle is irrelevant for the moment as we use only cycle consistency loss as of now

optimizer = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netG_frozen_deblur.parameters()),
filter(lambda p: p.requires_grad, netG_frozen_blur.parameters())), lr=learning_rate)
###Get the data. dataloader already defined above
# dataloader = DataLoader(dataset, batch_size=batch_size,
#                         shuffle=True, num_workers=num_workers)


# ### Training

# In[10]:

# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)

# def load_checkpoint(resume_file_path):
#     if os.path.isfile(resume_file_path):
#         print("=> loading checkpoint '{}'".format(resume_file_path))
#         checkpoint = torch.load(resume_file_path)
#         start_epoch = checkpoint['epoch']
#         #best_prec1 = checkpoint['best_prec1']
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {})"
#               .format(resume_file_path, checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))

def transformToImg(image):
    image_numpy = image.data.cpu().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy

def model_type_gpu(blur_net, deblur_net):
    num_gpus= torch.cuda.device_count()

    if num_gpus>1:
        print("more than 1 GPU detected...")
        netDeblur=torch.nn.DataParallel(deblur_net)
        netBlur=torch.nn.DataParallel(blur_net)

    elif num_gpus==1:
        print("A GPU detected...")
        netDeblur=deblur_net.cuda()
        netBlur=blur_net.cuda()

    else:
        pass

model_type_gpu(netG_frozen_deblur,netG_frozen_blur)      ##make the correct definition for the model
loss_full=[]
psnr_full_deblur=[]
psnr_full_blur=[]


for epoch in range(num_epoch):
    loss_epoch=0.0
    psnr_epoch_blur=0.0
    psnr_epoch_deblur=0.0
    cnt=0
    for i, data in enumerate(dataset):
        
        cnt+=1
        images=Variable(data['B']).cuda()
        labels=Variable(data['A']).cuda()

        optimizer.zero_grad()

        #forward loss part
        deblur_model_outputs_f = netG_frozen_deblur.forward(images)
        blur_model_outputs_f = netG_frozen_blur.forward(deblur_model_outputs_f)
        loss = cycle_consistency_criterion(blur_model_outputs_f, images)

        #backward loss part
        # blur_model_outputs_b= netBlur.forward(labels)
        # deblur_model_outputs_b= netDeblur.forward(blur_model_outputs_b)
        # loss_b= backward_cycle_consistency_criterion(labels, deblur_model_outputs_b)
        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            Visual_img = transformToImg(images[0,:,:,:])
            Visual_deblur_out = transformToImg(deblur_model_outputs_f[0,:,:,:])
            Visual_reblur_out = transformToImg(blur_model_outputs_f[0,:,:,:])
            Visual_label = transformToImg(labels[0,:,:,:])

            psnr_deblur_PSNR= compare_psnr(Visual_label, Visual_deblur_out)
            psnr_blur_PSNR= compare_psnr(Visual_img, Visual_reblur_out)
            
            print("epoch %d itr %d"% (epoch, i+1))
            print("The psnr of the deblur network is: %f"% psnr_deblur_PSNR)
            print("The psnr of the blur network is: %f"% psnr_blur_PSNR)
            print("loss is %f"% loss.data[0])
            
            print("epoch %d itr %d"% (epoch,i+1), file=open("outputFullModel.txt", "a"))
            print("The psnr of the deblur network is: %f"% psnr_deblur_PSNR, file=open("outputFullModel.txt", "a"))
            print("The psnr of the blur network is: %f"% psnr_blur_PSNR, file=open("outputFullModel.txt", "a"))
            print("loss is %f"% loss.data[0], file=open("outputFullModel.txt", "a"))
           
        
        
    if epoch%5 ==0:    ##save deblur once every 10 epochs
        save_network(netG_deblur, 'deblur_G', opt.which_epoch)
        save_network(netG_blur, 'blur_G', opt.which_epoch)
        save_network(netG_deblur, 'deblur_G', epoch)
        save_network(netG_blur, 'blur_G', epoch)
        