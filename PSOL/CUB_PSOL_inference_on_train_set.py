import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image
from utils.func import *
from utils.vis import *
from utils.IoU import *
from models.models import choose_locmodel,choose_clsmodel
from utils.augment import *
import argparse
from loader.cub_loader_adv import custom_Compose, custom_Resize, CUBirds_2011

parser = argparse.ArgumentParser(description='Parameters for PSOL evaluation')
parser.add_argument('--loc-model', metavar='locarg', type=str, default='vgg16',dest='locmodel')
parser.add_argument('--gpu',       help='which gpu to use',default='1',dest='gpu')
parser.add_argument('data',metavar='DIR',help='path to imagenet dataset')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
cudnn.benchmark = True


# augmentations and dataset
# we use Resize (instead of custom_Resize), since we only need img and we do not fetch bbox
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
loc_transform = custom_Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                normalize])

data_for_localizer = CUBirds_2011(root=args.data,
                                  split='trainval',
                                  target_type=['class', 'filename', 'original_wh'],
                                  transform=loc_transform)

# localizer
locname = args.locmodel
loc_model = choose_locmodel(locname, True)
print('Localization model ({}):\n'.format(locname))
print(loc_model)
loc_model = loc_model.to('cuda')
loc_model.eval()


save_pseudo_bboxes_path = './results/CUB_train_set/predicted_bounding_boxes'
if not os.path.exists(save_pseudo_bboxes_path):
    os.makedirs(save_pseudo_bboxes_path)


predicted_bboxes = {}

for img_index in range(len(data_for_localizer)):
    with torch.no_grad():
        # localization
        img_loc, (gt_label_1, filename, original_wh) = data_for_localizer[img_index]
        img_loc = img_loc.unsqueeze(0).to('cuda')
        reg_outputs = loc_model(img_loc)
        bbox = to_data(reg_outputs)
        bbox = torch.squeeze(bbox)  # The outputs are in (x, y, w, h) format
        bbox = bbox.numpy()         # This is the normalized (between 0 and 1) predicted bounding box: can think as the values are for input of size (w, h) = (1, 1)

    
    # convert predicted boxes to original_wh dimensions, but still in (x,y,w,h) format!
    w, h = original_wh
    bbox[0] = bbox[0]*w
    bbox[1] = bbox[1]*h
    bbox[2] = bbox[2]*w
    bbox[3] = bbox[3]*h
    predicted_bboxes[filename] = bbox
    

    # visualization code
    """
    from utils_custom_tvision_functions import imshow, plotregions
    %matplotlib qt
    imshow(img_loc)
    bbox = torch.tensor(bbox)
    bbox[2] -= bbox[0]
    bbox[3] -= bbox[1]
    plotregions([bbox])
    gt_bbox = torch.tensor(gt_bbox)
    gt_bbox[2] -= gt_bbox[0]
    gt_bbox[3] -= gt_bbox[1]
    plotregions([gt_bbox], color='r')
    """
    if (img_index+1) %100 == 0 or img_index == len(data_for_localizer)-1:
        print(img_index)

with open(os.path.join(save_pseudo_bboxes_path, 'psol_predicted_bounding_boxes.txt'), 'w') as fp:
    for k, v in predicted_bboxes.items():
        fp.write(k 
                 + ' ' + '{:.2f}'.format(v[0])
                 + ' ' + '{:.2f}'.format(v[1])
                 + ' ' + '{:.2f}'.format(v[2])
                 + ' ' + '{:.2f}\n'.format(v[3]))





# # GT-Known localization
# final_loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
# print("GT Loc: {:.4f} [{}/{}]\n".format(
#         (100.*final_loc_acc), np.sum(np.array(LocSet) > 0.5), len(LocSet)))