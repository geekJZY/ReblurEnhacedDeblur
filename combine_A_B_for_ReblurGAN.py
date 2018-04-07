
# coding: utf-8

# # Combine A and B for ReblurGAN
# * A is the input which is sharp image
# * B is the output which is blur image

# In[1]:


from pdb import set_trace as st
import os
import numpy as np
import cv2

root_dir = "../../data/train/" #training data for GoPro
det_dir = os.path.join(root_dir, "../../", "GOPRO_for_reblurGAN")
folders = sorted(os.listdir(root_dir))

for sp in folders:
    img_fold_A = os.path.join(root_dir,sp,"sharp")
    img_fold_B = os.path.join(root_dir,sp,"blur")
    img_list = os.listdir(img_fold_A)

    num_imgs = min(10000, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(det_dir, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)

