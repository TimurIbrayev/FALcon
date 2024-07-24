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
parser.add_argument('--cls-model', metavar='clsarg', type=str, default='resnet50',dest='clsmodel')
parser.add_argument('--ten-crop',  help='tencrop',   action='store_true',dest='tencrop')
parser.add_argument('--gpu',       help='which gpu to use',default='1',dest='gpu')
parser.add_argument('data',metavar='DIR',help='path to imagenet dataset')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
cudnn.benchmark = True


# augmentations and dataset
TEN_CROP = args.tencrop
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
loc_transform = custom_Compose([custom_Resize((224, 224)), # transforms both img and bbox
                                transforms.ToTensor(),
                                normalize])
# for cls_transform we use Resize (instead of custom_Resize), since we only need img and we do not fetch bbox for classifier
if TEN_CROP:
    cls_transform = custom_Compose([transforms.Resize((256, 256)), 
                                    transforms.TenCrop((256, 256)),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
else:
    cls_transform = custom_Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    normalize])

data_for_localizer = CUBirds_2011(root=args.data,
                                  split='test',
                                  target_type=['class', 'gt_bbox'],
                                  transform=loc_transform)

data_for_classifier = CUBirds_2011(root=args.data, 
                                   split='test', 
                                   target_type=['class'], 
                                   transform=cls_transform)
# simple sanity check
assert len(data_for_classifier) == len(data_for_localizer), "Mismatch between the number of images in the datasets for localization and classification!"


# localizer
locname = args.locmodel
loc_model = choose_locmodel(locname, True)
print('Localization model ({}):\n'.format(locname))
print(loc_model)
loc_model = loc_model.to('cuda')
loc_model.eval()

# classifier
clsname = args.clsmodel
cls_model = choose_clsmodel(clsname, pretrained=True, num_classes=200)
print('Classification model ({}):\n'.format(clsname))
print(cls_model)
cls_model = cls_model.to('cuda')
cls_model.eval()
temp_softmax = nn.Softmax(dim=1)




final_cls = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []

ClsSet = []
LocSet = []
IoUSet = []
IoUSetTop5 = []

for img_index in range(len(data_for_localizer)):
    with torch.no_grad():
        # localization
        img_loc, (gt_label_1, gt_bbox) = data_for_localizer[img_index]
        img_loc = img_loc.unsqueeze(0).to('cuda')
        reg_outputs = loc_model(img_loc)
        bbox = to_data(reg_outputs)
        bbox = torch.squeeze(bbox)  # The outputs are in (x, y, w, h) format
        bbox = bbox.numpy()         # This is the normalized (between 0 and 1) predicted bounding box: can think as the values are for input of size (w, h) = (1, 1)

        # classification
        if TEN_CROP:
            img_cls, gt_label_2 = data_for_classifier[img_index]
            img_cls = img_cls.to('cuda')
            cls_outputs = cls_model(img_cls)
            cls_outputs = temp_softmax(cls_outputs)
            cls_outputs = torch.mean(cls_outputs, dim=0, keepdim=True)
            cls_outputs = torch.topk(cls_outputs, 5, 1)[1]
        else:
            img_cls, gt_label_2 = data_for_classifier[img_index]
            img_cls = img_cls.unsqueeze(0).to('cuda')
            cls_outputs = cls_model(img_cls)
            cls_outputs = torch.topk(cls_outputs, 5, 1)[1]
        cls_outputs = to_data(cls_outputs)
        cls_outputs = torch.squeeze(cls_outputs)
        cls_outputs = cls_outputs.numpy()
    assert gt_label_1.item() == gt_label_2.item(), "Class labels do not match from datasets for localization and classification!"
    # top-1 classification results
    ClsSet.append(cls_outputs[0] == gt_label_2.item())
    
    # Note1a: gt_bbox is in the range of img_loc dimensions, i.e. (224, 224)!
    # Note1b: but, bbox is in range between 0 and 1. Hence, need to convert to img_loc dimensions!
    # Note2: both of them are in (x,y,w,h) format
    
    # convert predicted boxes to img_loc dimensions, but still in (x,y,w,h) format!
    b, c, h, w = img_loc.shape
    bbox[0] = bbox[0]*w
    bbox[1] = bbox[1]*h
    bbox[2] = bbox[2]*w
    bbox[3] = bbox[3]*h
    #pseudo_bboxes[filename] = temp_save_box
    
    # convert predicted boxes from (x,y,w,h) format to (x1,y1,x2,y2) format
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    # convert ground truth boxes from (x,y,w,h) format to (x1,y1,x2,y2) format
    gt_bbox = gt_bbox.numpy()
    gt_bbox[2] += gt_bbox[0]
    gt_bbox[3] += gt_bbox[1]
    
    # Note: CUB has only one object per image
    # GT-Known localization results
    iou = IoU(bbox, gt_bbox)
    max_iou = iou
    LocSet.append(max_iou)
    
    # top-1 localization accuracy results
    temp_loc_iou = max_iou
    if cls_outputs[0] != gt_label_2.item():
        max_iou = 0
    IoUSet.append(max_iou)
    
    # top-5 localization accuracy results
    max_iou = 0
    for i in range(5):
        if cls_outputs[i] == gt_label_2.item():
            max_iou = temp_loc_iou
    IoUSetTop5.append(max_iou)
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

# top-1 classification
final_cls_acc = np.sum(np.array(ClsSet)) / len(ClsSet)
# GT-Known localization
final_loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
# Top-1 localization accuracy 
final_clsloc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
# Top-5 localization accuracy
final_clsloctop5_acc = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
print("Top-1 Cls: {:.4f} [{}/{}] | GT Loc: {:.4f} [{}/{}] | Top-1 Loc: {:.4f} [{}/{}] | Top-5 Loc: {:.4f} [{}/{}] \n".format(
        (100.*final_cls_acc), np.sum(np.array(ClsSet)), len(ClsSet),
        (100.*final_loc_acc), np.sum(np.array(LocSet) > 0.5), len(LocSet),
        (100.*final_clsloc_acc), np.sum(np.array(IoUSet) > 0.5), len(IoUSet),
        (100.*final_clsloctop5_acc), np.sum(np.array(IoUSetTop5) > 0.5), len(IoUSetTop5)))