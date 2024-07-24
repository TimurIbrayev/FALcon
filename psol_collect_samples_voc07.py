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

from FALcon_config_test_as_WSOL import FALcon_config
from psol_models import choose_locmodel
from utils.utils_dataloaders import get_dataloaders
from utils.utils_custom_tvision_functions import plot_curve, imshow, plotregions, plotspots, plotspots_at_regioncenters, region_iou, region_area


parser = argparse.ArgumentParser(description='Collect predicted and groundtruth samples', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start',   default=1,              type=int,   help='The index of the first sample to collect')
parser.add_argument('--end',     default=-1,          type=int,   help='The index of the last sample to collect')
global args
args = parser.parse_args()


#%% Instantiate parameters, dataloaders, and model
# Parameters
config_3        = FALcon_config
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
dataset                  = get_dataloaders(config_3, loader_type=config_3.loader_type)


# Loss(es)
bce_loss                      = nn.BCEWithLogitsLoss()
ce_loss                       = nn.CrossEntropyLoss()  


# localization model
psol_loc_model = choose_locmodel('vgg16', pretrained=True).to(device)

#%% Evaluate the model performance.
args.end = len(dataset) if args.end == -1 else args.end
collected_samples = {}

with torch.no_grad():
    for i in range(args.start-1, args.end, 1):
        if config_3.dataset == 'voc07':
            image, (target_class, target_bbox, target_difficulties) = dataset[i]
            target_class = torch.tensor(target_class)
            target_difficulties = torch.tensor(target_difficulties)

        image, target_class, target_bbox = image.unsqueeze(0).to(device), target_class.to(device), target_bbox.to(device)

        psol_results = {}
        (h, w) = config_3.full_res_img_size
        predicted_bbox = psol_loc_model(image) # PSOL localizer, predictions are in (w,y,w,h) format, in range(0,1)
        predicted_bbox[:, 0] *= w
        predicted_bbox[:, 1] *= h
        predicted_bbox[:, 2] *= w
        predicted_bbox[:, 3] *= h
        psol_results["xywh_box"] = copy.deepcopy(predicted_bbox[0])
        predicted_bbox[:, 2] += predicted_bbox[:, 0]
        predicted_bbox[:, 3] += predicted_bbox[:, 1]
        psol_results["xyxy_box"] = copy.deepcopy(predicted_bbox[0])
        psol_results["prediction_label"] = dataset._class_to_ind['bird']
        psol_results["final_glimpse_objectness"] = 1.0

        sample_stats = {}
        sample_stats["gt_labels"] = copy.deepcopy(target_class)
        sample_stats["gt_bboxes"] = copy.deepcopy(target_bbox)
        sample_stats["gt_difficulties"] = copy.deepcopy(target_difficulties)
        sample_stats["gt_resized_wh"] = (config_3.full_res_img_size[1], config_3.full_res_img_size[0])
        sample_stats["predictions"] = copy.deepcopy([psol_results])


        collected_samples[i] = copy.deepcopy(sample_stats)
        if 'voc' in config_3.dataset:
            torch.save(collected_samples, './psol/' + "{}_collected_sample_from{}to{}.pth".format(config_3.dataset, args.start, args.end))
        
        if (i+1) %100 == 0 or i == len(dataset)-1:
            print("{}/{} requested samples processed!\n".format(
                (i+1), (args.end - args.start + 1)))

#%% Analyze VOC07 on test set annotations

# if config_3.dataset == 'voc07':
#     from voc_evaluator import do_voc_evaluation
#     results_voc = do_voc_evaluation(collected_samples)
