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
        if config_3.dataset == 'voc12':
            image, (filename, (w_original, h_original)) = dataset[i]

        image = image.unsqueeze(0).to(device)

        psol_results = {}
        (h, w) = config_3.full_res_img_size
        predicted_bbox = psol_loc_model(image) # PSOL localizer, predictions are in (w,y,w,h) format, in range(0,1)
        
        
        
        predicted_bbox[:, 0] *= w_original
        predicted_bbox[:, 1] *= h_original
        predicted_bbox[:, 2] *= w_original
        predicted_bbox[:, 3] *= h_original
        psol_results["xywh_box"] = copy.deepcopy(predicted_bbox[0])
        predicted_bbox[:, 2] += predicted_bbox[:, 0]
        predicted_bbox[:, 3] += predicted_bbox[:, 1]
        psol_results["xyxy_box"] = copy.deepcopy(predicted_bbox[0])
        psol_results["prediction_label"] = dataset._class_to_ind['bird']
        psol_results["final_glimpse_objectness"] = 1.0

        sample_stats = {}
        sample_stats["filename"] = copy.deepcopy(filename)
        sample_stats["original_wh"] = (w_original, h_original)
        sample_stats["predictions"] = copy.deepcopy([psol_results])


        collected_samples[i] = copy.deepcopy(sample_stats)
        if 'voc' in config_3.dataset:
            torch.save(collected_samples, './psol/' + "{}_collected_sample_from{}to{}.pth".format(config_3.dataset, args.start, args.end))
        
        if (i+1) %100 == 0 or i == len(dataset)-1:
            print("{}/{} requested samples processed!\n".format(
                (i+1), (args.end - args.start + 1)))

#%% Analyze VOC07 on test set annotations    

# if config_3.dataset == 'voc12':
#     save_file = './psol/' +  "voc12_comp3_det_test_bird.txt"
#     with open(save_file, "w") as fp:
#         for k, v in collected_samples.items():
#             filename = v['filename'].split('/')[-1].split('.')[0]
#             confidence = v["predictions"][0]["final_glimpse_objectness"]
#             bbox =  torch.round(v["predictions"][0]["xyxy_box"] + 1)
            
#             fp.write(filename + " " + 
#                      "{:.6f}".format(confidence)     + " " + 
#                      "{:.2f}".format(bbox[0].item()) + " " + 
#                      "{:.2f}".format(bbox[1].item()) + " " +
#                      "{:.2f}".format(bbox[2].item()) + " " +
#                      "{:.2f}".format(bbox[3].item()) + "\n")