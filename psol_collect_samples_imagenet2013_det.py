#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:54:36 2022

@author: tibrayev
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchviz import make_dot
import matplotlib.pyplot as plt
import copy
import argparse

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from psol_config_imagenet import psol_config
from psol_models import choose_locmodel
from cls_models import choose_clsmodel
from utils.utils_dataloaders import get_dataloaders
from utils.utils_custom_tvision_functions import plot_curve, imshow, plotregions, plotspots, plotspots_at_regioncenters, region_iou, region_area


parser = argparse.ArgumentParser(description='Collect predicted and groundtruth samples', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start',   default=1,              type=int,   help='The index of the first sample to collect')
parser.add_argument('--end',     default=-1,          type=int,   help='The index of the last sample to collect')
global args
args = parser.parse_args()


#%% Instantiate parameters, dataloaders, and model
# Parameters
config_3        = psol_config
config_3_copy   = {k: v for k, v in config_3.__dict__.items() if '__' not in k}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SEED instantiation
SEED            = config_3.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
#random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Dataloaders
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.loader_type)

# For testing, we allow the datasets (CUB and ImageNet) to fetch more than one bounding box
# However, then it requires to work on valid_loader.dataset, rather than valid_loader itself,
# since we did not customized default collate_fn in the loaders!
valid_loader.dataset.fetch_one_bbox = False

# Loss(es)
bce_loss                      = nn.BCEWithLogitsLoss()
ce_loss                       = nn.CrossEntropyLoss()  


# cls model
cls_model = choose_clsmodel(config_3.cls_model_name, config_3.cls_pretrained, config_3.cls_ckpt_dir, config_3.num_classes).to(device)
for p in cls_model.parameters():
    p.requires_grad_(False)
cls_model.eval()
print("Classification model:\n")
print(cls_model)

# localization model
loc_model = choose_locmodel('densenet161', pretrained=True).to(device)
for p in loc_model.parameters():
    loc_model.requires_grad_(False)
loc_model.eval()
print("Localization model:\n")
print(loc_model)


#%% Evaluate the model performance.
args.end = len(valid_loader.dataset) if args.end == -1 else args.end
collected_samples = {}

if config_3.dataset == 'cub':
    ten_crop = None
elif config_3.dataset == 'imagenet' or config_3.dataset == 'imagenet2013-det':
    # ten_crop = transforms.Compose([transforms.TenCrop(size=(224, 224))])
    ten_crop = None

with torch.no_grad():
    for i in range(args.start-1, args.end, 1):
        if config_3.dataset == 'imagenet2013-det':
            image, target_class, target_bbox = valid_loader.dataset[i]
            image = image.unsqueeze(0).to(device)

        psol_results = {}
        (h, w) = config_3.full_res_img_size
        predicted_bbox      = loc_model(image) # PSOL localizer, predictions are in (w,y,w,h) format, in range(0,1)
        predicted_bbox[:, 0] *= w
        predicted_bbox[:, 1] *= h
        predicted_bbox[:, 2] *= w
        predicted_bbox[:, 3] *= h
        psol_results["xywh_box"] = copy.deepcopy(predicted_bbox[0])
        predicted_bbox[:, 2] += predicted_bbox[:, 0]
        predicted_bbox[:, 3] += predicted_bbox[:, 1]
        psol_results["xyxy_box"] = copy.deepcopy(predicted_bbox[0])
        
        if ten_crop:
            ten_cropped_image           = torch.cat(ten_crop(image))
            outputs                     = cls_model(ten_cropped_image)
            outputs_probabilities       = F.softmax(outputs, dim=-1)
            output_probabilities        = torch.mean(outputs_probabilities, dim=0, keepdim=True)
        else:
            output                      = cls_model(image)
            output_probabilities        = F.softmax(output, dim=-1)
        prediction_confidence   = torch.max(output_probabilities, dim=-1)[0].item()
        prediction_label        = torch.max(output_probabilities, dim=-1)[1].item()
        
        prediction_synset       = valid_loader.dataset.label_class_dict[prediction_label]
        psol_results["prediction_confidence"]           = prediction_confidence
        psol_results["prediction_synset"]               = prediction_synset
        if prediction_synset in valid_loader.dataset.detection_synsets_to_labels.keys():
            psol_results["prediction_label"] = valid_loader.dataset.detection_synsets_to_labels[prediction_synset]
        else:
            psol_results["prediction_label"] = 0


        sample_stats = {}
        if len(target_class) > 0:
            sample_stats["gt_synsets"] = copy.deepcopy(target_class)
            sample_stats["gt_labels"] = copy.deepcopy(torch.tensor([valid_loader.dataset.detection_synsets_to_labels[t_class] for t_class in target_class]))
            sample_stats["gt_bboxes"] = copy.deepcopy(target_bbox)
        elif len(target_class) == 0:
            sample_stats["gt_synsets"] = copy.deepcopy(target_class)
            sample_stats["gt_labels"] = copy.deepcopy(torch.tensor([-1]))
            sample_stats["gt_bboxes"] = copy.deepcopy(torch.zeros((1, 4)).to(device))
        sample_stats["gt_resized_wh"] = (config_3.full_res_img_size[1], config_3.full_res_img_size[0])
        sample_stats["predictions"] = copy.deepcopy([psol_results])

        collected_samples[i] = copy.deepcopy(sample_stats)
        if config_3.dataset == 'imagenet2013-det':
            torch.save(collected_samples, './psol/' + "{}_collected_sample_from{}to{}.pth".format(config_3.dataset, args.start, args.end))
        
        if (i+1) %100 == 0 or i == len(valid_loader.dataset)-1:
            print("{}/{} requested samples processed!\n".format(
                (i+1), (args.end - args.start + 1)))

#%% Analyze ImageNet on test set annotations as WSOD

# if config_3.dataset == 'imagenet2013-det':
#     from voclike_imagenet_evaluator import do_voc_evaluation
#     collected_samples = {}
#     path_to_samples = './psol/'
#     partial_sample_collections = list(filter((lambda x: ('imagenet2013-det_collected_sample_from' in x)), os.listdir(path_to_samples)))
#     for partial in partial_sample_collections:
#         ckpt = torch.load(os.path.join(path_to_samples, partial))
#         collected_samples.update(ckpt)
    
#     ## For WSOD results:
#     results_ap = do_voc_evaluation(collected_samples)
#     print("TEST (WSOD) STATS: mAP: {}".format(results_ap["map"]))
  

    
