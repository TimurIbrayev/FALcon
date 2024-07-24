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
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.loader_type)

# For testing, we allow the datasets (CUB and ImageNet) to fetch more than one bounding box
# However, then it requires to work on valid_loader.dataset, rather than valid_loader itself,
# since we did not customized default collate_fn in the loaders!
valid_loader.dataset.fetch_one_bbox = False

# Loss(es)
bce_loss                      = nn.BCEWithLogitsLoss()
ce_loss                       = nn.CrossEntropyLoss()  


# cls model
if config_3.dataset == 'cub' or config_3.dataset == 'imagenet' or config_3.dataset == 'imagenet2013-det':
    cls_model = choose_clsmodel(config_3.cls_model_name, config_3.cls_pretrained, config_3.cls_ckpt_dir, config_3.num_classes).to(device)
    for p in cls_model.parameters():
        p.requires_grad_(False)
    cls_model.eval()
    print("Classification model:\n")
    print(cls_model)


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
args.end = len(valid_loader.dataset) if args.end == -1 else args.end
collected_samples = {}
if config_3.dataset == 'cub':
    ten_crop = None
elif config_3.dataset == 'imagenet' or config_3.dataset == 'imagenet2013-det':
    #ten_crop = transforms.Compose([transforms.TenCrop(size=(224, 224))])
    ten_crop = None

with torch.no_grad():
    for i in range(args.start-1, args.end, 1):
        if config_3.dataset == 'cub':
            image, (target_class, target_bbox) = valid_loader.dataset[i]
            target_class = target_class.unsqueeze(0)
            image, target_class, target_bbox = image.unsqueeze(0).to(device), target_class.to(device), target_bbox.to(device)
        elif config_3.dataset == 'imagenet':
            image, target_class, target_bbox = valid_loader.dataset[i]
            target_class = torch.tensor(target_class)
            image, target_class, target_bbox = image.unsqueeze(0).to(device), target_class.to(device), target_bbox.to(device)
        elif config_3.dataset == 'imagenet2013-det':
            image, target_class, target_bbox = valid_loader.dataset[i]
            image, target_bbox = image.unsqueeze(0).to(device), target_bbox.to(device)


        # Coordinates of all grid cells (either of the centers or top left corner coordinates of each grid cell)
        all_grid_cells_centers = get_grid((config_3.full_res_img_size[1], config_3.full_res_img_size[0]),
                                           config_3.glimpse_size_grid, grid_center_coords=True).to(device)
        
        # first, we foveate from every grid cell
        for switch_location_th in np.arange(config_3.switch_location_th, 1.0, 0.05):
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
                    all_potential_locations.append(foveation_results)
            if len(all_potential_locations) > 0:
                break
                
        # second, we filter based on objectness
        all_potential_locations_filtered_objectness = []
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
            # pass filtered foveated glimpses through classification model to get classification confidence and predicted class
            glimpses_extracted_resized  = extract_and_resize_glimpses_for_batch(image, potential_location["final_glimpse_loc_and_dim"], 
                                                                                config_3.full_res_img_size[1], config_3.full_res_img_size[0]) # glimpse_size_fixed[width, height]
            if ten_crop:
                ten_cropped_glimpses_extracted_resized = torch.cat(ten_crop(glimpses_extracted_resized))
                outputs                     = cls_model(ten_cropped_glimpses_extracted_resized)
                outputs_probabilities       = F.softmax(outputs, dim=-1)
                output_probabilities        = torch.mean(outputs_probabilities, dim=0, keepdim=True)
            else:
                output                      = cls_model(glimpses_extracted_resized)
                output_probabilities        = F.softmax(output, dim=-1)
            prediction_confidence       = torch.max(output_probabilities, dim=-1)[0].item()
            prediction_label            = torch.max(output_probabilities, dim=-1)[1].item()
            
            potential_location["prediction_confidence"]  = prediction_confidence
            potential_location["prediction_label"]       = prediction_label
            all_potential_locations_filtered_objectness.append(potential_location)
        
        # third, we filter based on classification confidence
        all_potential_locations_filtered_confidence = []
        xyxy_boxes = []
        cls_scores = []
        for potential_location in all_potential_locations_filtered_objectness:
            xyxy_boxes.append(potential_location["xyxy_box"])
            cls_scores.append(potential_location["prediction_confidence"])
        xyxy_boxes = torch.stack(xyxy_boxes, dim=0)*1.0
        cls_scores = torch.tensor(cls_scores).to(xyxy_boxes.device)
        nms_confidence_filtered_idx = nms(xyxy_boxes, cls_scores, config_3.confidence_based_nms_th)
        for idx in nms_confidence_filtered_idx:
            all_potential_locations_filtered_confidence.append(all_potential_locations_filtered_objectness[idx.item()])

        sample_stats = {}
        sample_stats["gt_labels"] = copy.deepcopy(target_class)
        if config_3.dataset == 'imagenet2013-det':
            sample_stats["gt_labels"] = copy.deepcopy(valid_loader.dataset)
        sample_stats["gt_bboxes"] = copy.deepcopy(target_bbox)
        sample_stats["predictions"] = copy.deepcopy(all_potential_locations_filtered_confidence)

        collected_samples[i] = copy.deepcopy(sample_stats)
        if config_3.dataset == 'cub' or config_3.dataset == 'imagenet':
            torch.save(collected_samples, config_3.save_dir + "collected_sample_from{}to{}.pth".format(args.start, args.end))
        
        if (i+1) %100 == 0 or i == len(valid_loader.dataset)-1:
            print("{}/{} requested samples processed!\n".format(
                (i+1), (args.end - args.start + 1)))

#%% Analyze ImageNet on test set annotations as WSOD

# if config_3.dataset == 'imagenet':
#     from voclike_imagenet_evaluator import do_voc_evaluation
#     collected_samples = {}
#     path_to_samples = './results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/'
#     partial_sample_collections = list(filter((lambda x: ('collected_sample_from' in x)), os.listdir(path_to_samples)))
#     for partial in partial_sample_collections:
#         ckpt = torch.load(os.path.join(path_to_samples, partial))
#         collected_samples.update(ckpt)
    
    
#     ## For WSOL results:
#     acc_correct_class               = 0
#     acc_localization                = 0
#     acc_class_localized             = 0
#     total_samples                   = 0
#     for sample_id, sample_stats in collected_samples.items():
#         target_bbox     = sample_stats["gt_bboxes"]
#         target_class    = sample_stats["gt_labels"]
        
#         # collect WSOL results statistics
#         total_samples   += 1
        
#         is_correct_label = []
#         is_correct_box   = []
#         cnt_predictions  = 0
#         for prediction in sample_stats["predictions"]:
#             for t_class, t_bbox in zip(target_class, target_bbox):
#                 if prediction["prediction_label"] == t_class:
#                     is_correct_label.append(True)
#                 else:
#                     is_correct_label.append(False)
#                 iou = region_iou(prediction["final_glimpse_loc_and_dim"], t_bbox.unsqueeze(0))
#                 if (iou >= 0.5).item():
#                     is_correct_box.append(True)
#                 else:
#                     is_correct_box.append(False)
#             cnt_predictions += 1
#             if cnt_predictions == 1: # limit the number of predictions per image
#                 break

#         is_correct_label = torch.tensor(is_correct_label)
#         is_correct_box   = torch.tensor(is_correct_box)
#         acc_correct_class   += torch.any(is_correct_label).sum().item()
#         acc_localization    += torch.any(is_correct_box).sum().item()
#         acc_class_localized += torch.any(torch.logical_and(is_correct_label, is_correct_box)).sum().item()
#     print("TEST (WSOL) STATS: Top-1 Cls: {:.4f} [{}/{}] | GT Loc: {:.4f} [{}/{}] | Top-1 Loc: {:.4f} [{}/{}]\n".format(
#         (100.*acc_correct_class/total_samples), acc_correct_class, total_samples,
#         (100.*acc_localization/total_samples), acc_localization, total_samples,
#         (100.*acc_class_localized/total_samples), acc_class_localized, total_samples))


#     ## For WSOD results:
#     results_ap = do_voc_evaluation(collected_samples)
#     print("TEST (WSOD) STATS: mAP: {}".format(results_ap["map"]))
    
   
    
# if config_3.dataset == 'cub':
#     collected_samples = {}
#     path_to_samples = './results/cub/wsol_method_PSOL/trained_on_trainval_split_evaluated_on_test_split/arch_vgg11_pretrained_init_normalization_none_seed_16/'
#     partial_sample_collections = list(filter((lambda x: (('collected_sample_from' in x) and (not ('voc' in x)))), os.listdir(path_to_samples)))
#     for partial in partial_sample_collections:
#         ckpt = torch.load(os.path.join(path_to_samples, partial))
#         collected_samples.update(ckpt)
    
    
#     ## For WSOL results:
#     acc_correct_class               = 0
#     acc_localization                = 0
#     acc_class_localized             = 0
#     total_samples                   = 0
#     for sample_id, sample_stats in collected_samples.items():
#         target_bbox     = sample_stats["gt_bboxes"]
#         target_class    = sample_stats["gt_labels"]
        
#         # collect WSOL results statistics
#         total_samples   += 1
        
#         is_correct_label = []
#         is_correct_box   = []
#         cnt_predictions  = 0
#         for prediction in sample_stats["predictions"]:
#             for t_class, t_bbox in zip(target_class, target_bbox):
#                 if prediction["prediction_label"] == t_class:
#                     is_correct_label.append(True)
#                 else:
#                     is_correct_label.append(False)
#                 iou = region_iou(prediction["final_glimpse_loc_and_dim"], t_bbox.unsqueeze(0))
#                 if (iou >= 0.5).item():
#                     is_correct_box.append(True)
#                 else:
#                     is_correct_box.append(False)
#             cnt_predictions += 1
#             if cnt_predictions == 1: # limit the number of predictions per image
#                 break

#         is_correct_label = torch.tensor(is_correct_label)
#         is_correct_box   = torch.tensor(is_correct_box)
#         acc_correct_class   += torch.any(is_correct_label).sum().item()
#         acc_localization    += torch.any(is_correct_box).sum().item()
#         acc_class_localized += torch.any(torch.logical_and(is_correct_label, is_correct_box)).sum().item()
#     print("TEST (WSOL) STATS: Top-1 Cls: {:.4f} [{}/{}] | GT Loc: {:.4f} [{}/{}] | Top-1 Loc: {:.4f} [{}/{}]\n".format(
#         (100.*acc_correct_class/total_samples), acc_correct_class, total_samples,
#         (100.*acc_localization/total_samples), acc_localization, total_samples,
#         (100.*acc_class_localized/total_samples), acc_class_localized, total_samples))