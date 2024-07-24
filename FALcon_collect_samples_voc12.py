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
from FALcon_models_vgg import customizable_VGG as custom_vgg
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


# FALcon (localization) model
model_3 = custom_vgg(config_3).to(device)
for p in model_3.parameters():
    model_3.requires_grad_(False)
model_3.eval()
print("FALcon (localization) model:\n")
print(model_3)


# Logger dictionary
if not os.path.exists(config_3.save_dir): os.makedirs(config_3.save_dir)

#%% AVS-specific functions, methods, and parameters
from AVS_functions import extract_and_resize_glimpses_for_batch, get_grid, get_new_random_glimpses_for_batch
from torchvision.ops import nms

#%% Simplify the code.
def FALcon_from_init_glimpse_loc(config, locmodel, input_image, init_glimpse_loc, switch_location_th):
    foveation_progress_per_glimpse_loc = []
    
    glimpses_locs_dims       = torch.zeros((input_image.shape[0], 4), dtype=torch.int).to(input_image.device)
    glimpses_locs_dims[:, 0] = init_glimpse_loc[0] + 0.5 - (config.glimpse_size_grid[0]/2.0)
    glimpses_locs_dims[:, 1] = init_glimpse_loc[1] + 0.5 - (config.glimpse_size_grid[1]/2.0)
    glimpses_locs_dims[:, 2] = config.glimpse_size_init[0]
    glimpses_locs_dims[:, 3] = config.glimpse_size_init[1]
    foveation_progress_per_glimpse_loc.append(glimpses_locs_dims.clone().detach())
    
    for g in range(config.num_glimpses):
        glimpses_extracted_resized = extract_and_resize_glimpses_for_batch(input_image, glimpses_locs_dims,
                                                                           config.glimpse_size_fixed[1], config.glimpse_size_fixed[0]) # glimpse_size_fixed[width, height]
            
        glimpses_change_predictions, switch_location_predictions = locmodel(glimpses_extracted_resized)

        switch_location_probability     = torch.sigmoid(switch_location_predictions.clone().detach()).squeeze(1)  
        switch_location_actions         = (switch_location_probability >= switch_location_th).item()
        if switch_location_actions:
            break

        # Current glimpse bounds.
        x_min_current   = (glimpses_locs_dims[:, 0]).clone().detach()
        x_max_current   = (glimpses_locs_dims[:, 0]+glimpses_locs_dims[:, 2]).clone().detach()
        y_min_current   = (glimpses_locs_dims[:, 1]).clone().detach()
        y_max_current   = (glimpses_locs_dims[:, 1]+glimpses_locs_dims[:, 3]).clone().detach()

        # Change glimpse dimensions according to model predictions.
        glimpses_change_probability     = torch.sigmoid(glimpses_change_predictions.clone().detach())
        glimpses_change_actions         = (glimpses_change_probability >= config.glimpse_change_th)
        # Check so that glimpses do not go out of the image boundaries.
        x_min_new       = torch.clamp(x_min_current - glimpses_change_actions[:, 0]*config.glimpse_size_step[0], min=0)
        x_max_new       = torch.clamp(x_max_current + glimpses_change_actions[:, 1]*config.glimpse_size_step[0], max=input_image.shape[-1]) #(height, width) as used in transforms.Resize
        y_min_new       = torch.clamp(y_min_current - glimpses_change_actions[:, 2]*config.glimpse_size_step[1], min=0)
        y_max_new       = torch.clamp(y_max_current + glimpses_change_actions[:, 3]*config.glimpse_size_step[1], max=input_image.shape[-2]) #(height, width) as used in transforms.Resize
            
        # Store the new glimpse locations and dimensions.
        glimpses_locs_dims[:, 0] = x_min_new.clone().detach()
        glimpses_locs_dims[:, 1] = y_min_new.clone().detach()
        glimpses_locs_dims[:, 2] = x_max_new.clone().detach() - glimpses_locs_dims[:, 0]
        glimpses_locs_dims[:, 3] = y_max_new.clone().detach() - glimpses_locs_dims[:, 1]
        foveation_progress_per_glimpse_loc.append(glimpses_locs_dims.clone().detach())

    foveation_results = {}
    foveation_results["final_glimpse_switch_probability"]   = switch_location_probability.item()
    foveation_results["final_glimpse_objectness"]           = 1.0 - switch_location_probability.item()
    foveation_results["final_glimpse_loc_and_dim"]          = copy.deepcopy(glimpses_locs_dims)
    foveation_results["foveation_progress"]                 = copy.deepcopy(foveation_progress_per_glimpse_loc)
    return foveation_results


#%% Evaluate the model performance.
args.end = len(dataset) if args.end == -1 else args.end
collected_samples = {}
cnt_discarded = 0

with torch.no_grad():
    for i in range(args.start-1, args.end, 1):
        if config_3.dataset == 'voc12':
            image, (filename, (w_original, h_original)) = dataset[i]

        image = image.unsqueeze(0).to(device)

        # Coordinates of all grid cells (either of the centers or top left corner coordinates of each grid cell)
        all_grid_cells_centers = get_grid((config_3.full_res_img_size[1], config_3.full_res_img_size[0]),
                                           config_3.glimpse_size_grid, grid_center_coords=True).to(device)
        
        # first, we foveate from every grid cell
        switch_location_th = config_3.switch_location_th
        all_potential_locations = []
        for grid_cell in all_grid_cells_centers:
            foveation_results = FALcon_from_init_glimpse_loc(config = config_3, 
                                                             locmodel = model_3, 
                                                             input_image = image, 
                                                             init_glimpse_loc = grid_cell, 
                                                             switch_location_th = switch_location_th)
            
            # store the ones which had high objectness (i.e. 1-switch_probability) scores
            if foveation_results["final_glimpse_switch_probability"] < switch_location_th:
                foveation_results["xywh_box"] = copy.deepcopy(foveation_results["final_glimpse_loc_and_dim"][0])
                foveation_results["xyxy_box"] = copy.deepcopy(foveation_results["final_glimpse_loc_and_dim"][0])
                foveation_results["xyxy_box"][2] += foveation_results["xyxy_box"][0]
                foveation_results["xyxy_box"][3] += foveation_results["xyxy_box"][1]
                all_potential_locations.append(copy.deepcopy(foveation_results))
                
        # second, we filter based on objectness
        all_potential_locations_filtered_objectness = []
        if len(all_potential_locations) > 0:
            xyxy_boxes = []
            obj_scores = []
            for potential_location in all_potential_locations:
                xyxy_boxes.append(potential_location["xyxy_box"])
                obj_scores.append(potential_location["final_glimpse_objectness"])
            xyxy_boxes = torch.stack(xyxy_boxes, dim=0)*1.0
            obj_scores = torch.tensor(obj_scores).to(xyxy_boxes.device)
            nms_objectness_filtered_idx = nms(xyxy_boxes, obj_scores, config_3.objectness_based_nms_th)
            for idx in nms_objectness_filtered_idx:
                potential_location          = all_potential_locations[idx.item()]
                
                potential_location["prediction_confidence"]  = 1.0
                potential_location["prediction_label"]       = dataset._class_to_ind['bird']
                all_potential_locations_filtered_objectness.append(potential_location)
        
        elif len(all_potential_locations) == 0:
            dummy_result = {}
            dummy_result["xyxy_box"] = torch.zeros((1, 4)).to(device)
            dummy_result["prediction_label"] = -1
            dummy_result["final_glimpse_objectness"] = 0.0
            all_potential_locations_filtered_objectness.append(copy.deepcopy(dummy_result))
            cnt_discarded += 1

        sample_stats = {}
        sample_stats["filename"] = copy.deepcopy(filename)
        sample_stats["original_wh"] = (w_original, h_original)
        sample_stats["gt_resized_wh"] = (config_3.full_res_img_size[1], config_3.full_res_img_size[0])
        sample_stats["predictions"] = copy.deepcopy(all_potential_locations_filtered_objectness)


        collected_samples[i] = copy.deepcopy(sample_stats)
        if 'voc' in config_3.dataset:
            torch.save(collected_samples, config_3.save_dir + "{}_collected_sample_from{}to{}.pth".format(config_3.dataset, args.start, args.end))
        
        if (i+1) %100 == 0 or i == len(dataset)-1:
            print("{}/{} requested samples processed!\n".format(
                (i+1), (args.end - args.start + 1)))

#%% Analyze VOC07 on test set annotations

# if config_3.dataset == 'voc12':
#     save_file = config_3.save_dir +  "voc12_comp3_det_test_bird.txt"
#     with open(save_file, "w") as fp:
#         for k, v in collected_samples.items():
#             if v["predictions"][0]["prediction_label"] == -1:
#                 continue
#             else:
#                 filename = v['filename'].split('/')[-1].split('.')[0]
#                 w_original, h_original = v['original_wh']
#                 w_resized, h_resized = v['gt_resized_wh']
#                 ratio_x = w_original*1.0/w_resized
#                 ratio_y = h_original*1.0/h_resized
                
#                 for prediction in v["predictions"]:
#                     confidence = prediction["final_glimpse_objectness"]
#                     bbox = prediction["xyxy_box"].float()
#                     bbox[0] = bbox[0] * ratio_x
#                     bbox[1] = bbox[1] * ratio_y
#                     bbox[2] = bbox[2] * ratio_x
#                     bbox[3] = bbox[3] * ratio_y
#                     bbox = torch.round(bbox + 1)
                    
#                     fp.write(filename + " " + 
#                              "{:.6f}".format(confidence)     + " " + 
#                              "{:.2f}".format(bbox[0].item()) + " " + 
#                              "{:.2f}".format(bbox[1].item()) + " " +
#                              "{:.2f}".format(bbox[2].item()) + " " +
#                              "{:.2f}".format(bbox[3].item()) + "\n")