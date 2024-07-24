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

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from FALcon_config_imagenet import FALcon_config
from FALcon_models_vgg import customizable_VGG as custom_vgg
from utils.utils_dataloaders import get_dataloaders
from utils.utils_custom_tvision_functions import plot_curve, imshow, plotregions, plotspots, plotspots_at_regioncenters, region_iou, region_area


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
if config_3.train_loader_type == 'train_and_val':
	train_loader, valid_loader    = get_dataloaders(config_3, loader_type=config_3.train_loader_type)
elif config_3.train_loader_type == 'train':
	train_loader = get_dataloaders(config_3, loader_type=config_3.train_loader_type)
	valid_loader = get_dataloaders(config_3, loader_type=config_3.valid_loader_type)
else:
	raise ValueError("Unrecognized train_loader_type for ImageNet dataset train script!")	
# This training script supports only single bounding box. Hence, need to turn off multiple bbox fetching.
train_loader.dataset.fetch_one_bbox = True
valid_loader.dataset.fetch_one_bbox = True

# Loss(es)
bce_loss                      = nn.BCEWithLogitsLoss()
ce_loss                       = nn.CrossEntropyLoss()  



# FIXME: Model
if 'vgg' in config_3.model_name:
    model_3 = custom_vgg(config_3).to(device)
# elif 'resnet' in config_3.model_name:
#     model_3 = custom_resnet(config_3).to(device)
else:
    raise ValueError("Unknown model_name specified.")



# Optimizer
optimizer = torch.optim.SGD(model_3.parameters(), lr=config_3.lr_start, weight_decay=config_3.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=config_3.milestones)

# Logger dictionary
log_dict     = {'train_loss':[], 
                'train_loss_classification': [], 
                'train_loss_glimpse_dim_change': [],
                'train_loss_glimpse_loc_change': [],
                'train_acc_correct_class':[],
                'train_acc_localization':[],
                'train_acc_class_localized':[],
                'test_loss':[],
                'test_loss_classification': [],
                'test_loss_glimpse_dim_change': [],
                'test_loss_glimpse_loc_change': [],
                'test_acc_correct_class':[],
                'test_acc_localization':[],
                'test_acc_class_localized':[],
                }
if not os.path.exists(config_3.save_dir): os.makedirs(config_3.save_dir)

#%% AVS-specific functions, methods, and parameters
from AVS_functions import extract_and_resize_glimpses_for_batch, get_grid, guess_TF_init_glimpses_for_batch


#%% Train the model
for epoch in range(0, config_3.epochs):
    print("Epoch: {}/{}\n".format(epoch+1, config_3.epochs))
# =============================================================================
#   TRAINING
# =============================================================================
    model_3.train()
    train_loss                      = 0.0
    train_loss_classification       = 0.0
    train_loss_glimpse_dim_change   = 0.0
    train_loss_glimpse_loc_change   = 0.0
    acc_correct_class           = 0
    acc_localization            = 0
    acc_class_localized         = 0
    acc_switching               = 0
    train_ave_iou               = 0.0
    total_samples               = 0
    glimpses_locs_dims_array    = []
    for i, (images, targets, bboxes) in enumerate(train_loader):
        translated_images, targets_classes, bbox_targets  = images.to(device), targets.to(device), bboxes.to(device)
# =============================================================================
#       DATA STRUCTURES to keep track of glimpses
# =============================================================================
        # Data structure to keep track of glimpse locations and dimensions
        # glimpses_locs_dims[batch_size, 4] - where second dimension is 4-sized tuple,
        # representing (x_TopLeftCorner, y_TopLeftCorner, width, height) of each glimpse in the batch
        glimpses_locs_dims      = torch.zeros((translated_images.shape[0], 4), dtype=torch.int).to(device)
        
        # Data structure to store actual batch of extracted and resized glimpses to be fetched to the network
        # !!!: in order to be able to process the batch of different sized glimpses in one feedforward path through the network,
        # after extracting glimpses from each individual image (according to locations and dimensions of corresponding glimpses)
        # we are resizing all of them to some fixed, pre-determined fixed size, which is determined by config.glimpse_size_fixed parameter!
        # !!!: glimpses_extracted_resized[batch_size, input_channels, fixed_glimpse_height, fixed_glimpse_width]
        glimpses_extracted_resized = torch.zeros((translated_images.shape[0],   
                                                  translated_images.shape[1], 
                                                  config_3.glimpse_size_fixed[1], 
                                                  config_3.glimpse_size_fixed[0])).to(device) # glimpse_size_fixed[width, height]


# =============================================================================
#       Getting initial glimpse locations        
# =============================================================================
        all_grid_cells_centers  = get_grid((config_3.full_res_img_size[1], config_3.full_res_img_size[0]),
                                            config_3.glimpse_size_grid, grid_center_coords=True).to(device)

        init_glimpses_in_bbox   = guess_TF_init_glimpses_for_batch(all_grid_cells_centers, bbox_targets, is_inside_bbox=True)
        init_glimpses_out_bbox  = guess_TF_init_glimpses_for_batch(all_grid_cells_centers, bbox_targets, is_inside_bbox=False)
        
        init_glimpses_wrong     = torch.rand(translated_images.shape[0]).to(device) < config_3.ratio_wrong_init_glimpses
        init_glimpses_correct   = torch.logical_not(init_glimpses_wrong).clone().detach()
        
        glimpses_locs_centers   = torch.zeros_like(init_glimpses_in_bbox)
        glimpses_locs_centers[init_glimpses_correct, :] = init_glimpses_in_bbox[init_glimpses_correct, :]
        glimpses_locs_centers[init_glimpses_wrong, :]   = init_glimpses_out_bbox[init_glimpses_wrong, :]

        # NOTE: guess_TF_init_glimpses_for_batch outputs center coordinates, 
        # i.e. (x_Center, y_Center) for possible initial guess glimpses
        # Hence, we translate the center of the initial point-of-interest into (x_TopLeftCorner, y_TopLeftCorner, width, height)       
        glimpses_locs_dims[:, 0] = glimpses_locs_centers[:, 0] + 0.5 - (config_3.glimpse_size_grid[0]/2.0)
        glimpses_locs_dims[:, 1] = glimpses_locs_centers[:, 1] + 0.5 - (config_3.glimpse_size_grid[1]/2.0)
        glimpses_locs_dims[:, 2] = config_3.glimpse_size_init[0]
        glimpses_locs_dims[:, 3] = config_3.glimpse_size_init[1]
        if i == 0:
            glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())


# =============================================================================
#       M3 Stage
# =============================================================================
        optimizer.zero_grad()
        loss_classification         = torch.tensor([0.0]) #None
        loss_glimpse_dim_change     = None
        loss_glimpse_loc_change     = None
        
        for g in range(config_3.num_glimpses):
            # Extract and resize the batch of glimpses based on their current locations and dimensions.
            glimpses_extracted_resized = extract_and_resize_glimpses_for_batch(translated_images, glimpses_locs_dims,
                                                                               config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0]) # glimpse_size_fixed[width, height]
            
            # Process the batch of extracted and resized glimpses through the network.
            # glimpses_change_actions[batch_size, 4] - where second dimension is 4-sized tuple, 
            # representing (dx-, dx+, dy-, dy+) changes of every glimpse in x and y directions
            glimpses_change_predictions, switch_location_predictions = model_3(glimpses_extracted_resized)
            outputs_classes = torch.zeros((translated_images.shape[0], config_3.num_classes)).to(device) # FIXME: change for classifier model

            # # Estimate classification loss.
            # if g == 0:
            #     loss_classification  = ce_loss(outputs_classes, targets_classes)
            # else:
            #     loss_classification += ce_loss(outputs_classes, targets_classes)


            # Current glimpse bounds.
            x_min_current   = (glimpses_locs_dims[:, 0]).clone().detach()
            x_max_current   = (glimpses_locs_dims[:, 0]+glimpses_locs_dims[:, 2]).clone().detach()
            y_min_current   = (glimpses_locs_dims[:, 1]).clone().detach()
            y_max_current   = (glimpses_locs_dims[:, 1]+glimpses_locs_dims[:, 3]).clone().detach()
            
            # Glimpse bounds IF their expansion is allowed with step dictated by config.glimpse_size_step.
            x_min_future    = x_min_current - config_3.glimpse_size_step[0]
            x_max_future    = x_max_current + config_3.glimpse_size_step[0]
            y_min_future    = y_min_current - config_3.glimpse_size_step[1]
            y_max_future    = y_max_current + config_3.glimpse_size_step[1]
            
            # Estimate targets which are true IF the current glimpse can be expanded so that new glimpse dimensions are within bounding box.
            x_min_target    = (x_min_future >= bbox_targets[:, 0]).float()
            x_max_target    = (x_max_future <= (bbox_targets[:, 0]+bbox_targets[:, 2])).float()
            y_min_target    = (y_min_future >= bbox_targets[:, 1]).float()
            y_max_target    = (y_max_future <= (bbox_targets[:, 1]+bbox_targets[:, 3])).float()
            
            # Estimate the loss.
            glimpses_change_target      = torch.stack([x_min_target, x_max_target, y_min_target, y_max_target], dim=1)
            if g == 0:
                loss_glimpse_dim_change     = bce_loss(glimpses_change_predictions, glimpses_change_target.clone().detach())
                loss_glimpse_loc_change     = bce_loss(switch_location_predictions, init_glimpses_wrong.float().unsqueeze(1).clone().detach())
            else:
                loss_glimpse_dim_change    += bce_loss(glimpses_change_predictions, glimpses_change_target.clone().detach())
                loss_glimpse_loc_change    += bce_loss(switch_location_predictions, init_glimpses_wrong.float().unsqueeze(1).clone().detach())


            # Change glimpse dimensions according to model predictions.
            glimpses_change_probability     = torch.sigmoid(glimpses_change_predictions.clone().detach())
            glimpses_change_actions         = (glimpses_change_probability >= config_3.glimpse_change_th)
            # Check so that glimpses do not go out of the image boundaries.
            x_min_new       = torch.clamp(x_min_current - glimpses_change_actions[:, 0]*config_3.glimpse_size_step[0], min=0)
            x_max_new       = torch.clamp(x_max_current + glimpses_change_actions[:, 1]*config_3.glimpse_size_step[0], max=config_3.full_res_img_size[1]) #(height, width) as used in transforms.Resize
            y_min_new       = torch.clamp(y_min_current - glimpses_change_actions[:, 2]*config_3.glimpse_size_step[1], min=0)
            y_max_new       = torch.clamp(y_max_current + glimpses_change_actions[:, 3]*config_3.glimpse_size_step[1], max=config_3.full_res_img_size[0]) #(height, width) as used in transforms.Resize
            
            # Store the new glimpse locations and dimensions.
            glimpses_locs_dims[:, 0] = x_min_new.clone().detach()
            glimpses_locs_dims[:, 1] = y_min_new.clone().detach()
            glimpses_locs_dims[:, 2] = x_max_new.clone().detach() - glimpses_locs_dims[:, 0]
            glimpses_locs_dims[:, 3] = y_max_new.clone().detach() - glimpses_locs_dims[:, 1]
            
            
            # Switch glimpse location according to model predictions.
            # NOTE: curriculum learning: switch location only if it is wrong initial location
            switch_location_probability     = torch.sigmoid(switch_location_predictions.clone().detach()).squeeze(1)
            switch_location_actions         = (switch_location_probability >= config_3.switch_location_th)
            switch_location_to_correct      = torch.logical_and(init_glimpses_wrong, switch_location_actions)
            
            # switch_location_to_correct is a mask of every sample for which CURRENT location was wrong AND network predicted it needs to SWITCH
            glimpses_locs_dims[switch_location_to_correct, 0]   = (init_glimpses_in_bbox[switch_location_to_correct, 0] + 0.5 - (config_3.glimpse_size_grid[0]/2.0)).int()
            glimpses_locs_dims[switch_location_to_correct, 1]   = (init_glimpses_in_bbox[switch_location_to_correct, 1] + 0.5 - (config_3.glimpse_size_grid[0]/2.0)).int()
            glimpses_locs_dims[switch_location_to_correct, 2]   = config_3.glimpse_size_init[0]
            glimpses_locs_dims[switch_location_to_correct, 3]   = config_3.glimpse_size_init[1]
            
            # Record if any of the locations was changed based on model predictions.
            init_glimpses_wrong[switch_location_to_correct] = False
            init_glimpses_correct = torch.logical_not(init_glimpses_wrong).clone().detach()

            if i == 0:
                glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())


        # Backpropagate the errors for all iterations.
        loss                = loss_glimpse_dim_change + loss_glimpse_loc_change #+ loss_classification
        loss.backward()
        optimizer.step()


        _, pred_classes                  = torch.max(outputs_classes.data, 1)
        total_samples                   += targets_classes.size(0)
        train_loss                      += loss.item()
        train_loss_classification       += loss_classification.item()
        train_loss_glimpse_dim_change   += loss_glimpse_dim_change.item()
        train_loss_glimpse_loc_change   += loss_glimpse_loc_change.item()

        iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
        train_ave_iou               += iou.sum().item()

        correct_classes              = pred_classes == targets_classes
        correct_tp_loc               = iou >= config_3.iou_th
        

        acc_correct_class           += correct_classes.sum().item()
        acc_localization            += correct_tp_loc.sum().item()
        acc_class_localized         += correct_classes[correct_tp_loc].sum().item()
        acc_switching               += init_glimpses_correct.sum().item()

    print("Train Loss: {:.3f} | {:.3f} | {:.3f} | {:.3f}, Switching Acc: {:.4f}\n".format(
        (train_loss/(i+1)), (train_loss_classification/(i+1)), (train_loss_glimpse_dim_change/(i+1)), 
        (train_loss_glimpse_loc_change/(i+1)), (100.*acc_switching/total_samples)))

    print("Top-1 Cls: {:.4f} [{}/{}] | GT Loc: {:.4f} [{}/{}] | Top-1 Loc: {:.4f} [{}/{}]\n".format(
        (100.*acc_correct_class/total_samples), acc_correct_class, total_samples,
        (100.*acc_localization/total_samples), acc_localization, total_samples,
        (100.*acc_class_localized/total_samples), acc_class_localized, total_samples))
    log_dict['train_loss'].append(train_loss/(i+1))
    log_dict['train_loss_classification'].append(train_loss_classification/(i+1))
    log_dict['train_loss_glimpse_dim_change'].append(train_loss_glimpse_dim_change/(i+1))
    log_dict['train_loss_glimpse_loc_change'].append(train_loss_glimpse_loc_change/(i+1))
    log_dict['train_acc_correct_class'].append(100.*acc_correct_class/total_samples)
    log_dict['train_acc_localization'].append(100.*acc_localization/total_samples)
    log_dict['train_acc_class_localized'].append(100.*acc_class_localized/total_samples)
    print("IoU@{}: Average: {:.3f} | TPR: {:.4f} [{}/{}]\n".format(
        config_3.iou_th, (train_ave_iou/total_samples),
        (1.*acc_localization/total_samples), acc_localization, total_samples))



# =============================================================================
#   EVALUATION
# =============================================================================
    model_3.eval()
    test_loss                       = 0.0
    test_loss_classification        = 0.0
    test_loss_glimpse_dim_change    = 0.0
    test_loss_glimpse_loc_change    = 0.0
    acc_correct_class           = 0
    acc_localization            = 0
    acc_class_localized         = 0
    acc_switching               = 0
    test_ave_iou                = 0.0
    total_samples               = 0
    glimpses_locs_dims_array    = []
    with torch.no_grad():
        for i, (images, targets, bboxes) in enumerate(valid_loader):
            translated_images, targets_classes, bbox_targets  = images.to(device), targets.to(device), bboxes.to(device)
# =============================================================================
#       DATA STRUCTURES to keep track of glimpses
# =============================================================================
            glimpses_locs_dims          = torch.zeros((translated_images.shape[0], 4), dtype=torch.int).to(device)
            glimpses_extracted_resized  = torch.zeros((translated_images.shape[0],   
                                                       translated_images.shape[1], 
                                                       config_3.glimpse_size_fixed[1], 
                                                       config_3.glimpse_size_fixed[0])).to(device) # glimpse_size_fixed[width, height]

# =============================================================================
#       Getting initial glimpse locations        
# =============================================================================
            all_grid_cells_centers  = get_grid((config_3.full_res_img_size[1], config_3.full_res_img_size[0]),
                                            config_3.glimpse_size_grid, grid_center_coords=True).to(device)

            init_glimpses_in_bbox   = guess_TF_init_glimpses_for_batch(all_grid_cells_centers, bbox_targets, is_inside_bbox=True)
            init_glimpses_out_bbox  = guess_TF_init_glimpses_for_batch(all_grid_cells_centers, bbox_targets, is_inside_bbox=False)
        
            init_glimpses_wrong     = torch.rand(translated_images.shape[0]).to(device) < config_3.ratio_wrong_init_glimpses
            init_glimpses_correct   = torch.logical_not(init_glimpses_wrong).clone().detach()
        
            glimpses_locs_centers   = torch.zeros_like(init_glimpses_in_bbox)
            glimpses_locs_centers[init_glimpses_correct, :] = init_glimpses_in_bbox[init_glimpses_correct, :]
            glimpses_locs_centers[init_glimpses_wrong, :]   = init_glimpses_out_bbox[init_glimpses_wrong, :]

            # NOTE: guess_TF_init_glimpses_for_batch outputs center coordinates, 
            # i.e. (x_Center, y_Center) for possible initial guess glimpses
            # Hence, we translate the center of the initial point-of-interest into (x_TopLeftCorner, y_TopLeftCorner, width, height)       
            glimpses_locs_dims[:, 0] = glimpses_locs_centers[:, 0] + 0.5 - (config_3.glimpse_size_grid[0]/2.0)
            glimpses_locs_dims[:, 1] = glimpses_locs_centers[:, 1] + 0.5 - (config_3.glimpse_size_grid[1]/2.0)
            glimpses_locs_dims[:, 2] = config_3.glimpse_size_init[0]
            glimpses_locs_dims[:, 3] = config_3.glimpse_size_init[1]
            if i == len(valid_loader) - 1:
                glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())

# =============================================================================
#       M3 Stage
# =============================================================================
            loss_classification         = torch.tensor([0.0]) #None
            loss_glimpse_dim_change     = None
            loss_glimpse_loc_change     = None
        
            for g in range(config_3.num_glimpses):
                # Extract and resize the batch of glimpses based on their current locations and dimensions.
                glimpses_extracted_resized = extract_and_resize_glimpses_for_batch(translated_images, glimpses_locs_dims,
                                                                                   config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0]) # glimpse_size_fixed[width, height]
                
                # Process the batch of extracted and resized glimpses through the network.
                # glimpses_change_actions[batch_size, 4] - where second dimension is 4-sized tuple, 
                # representing (dx-, dx+, dy-, dy+) changes of every glimpse in x and y directions
                glimpses_change_predictions, switch_location_predictions = model_3(glimpses_extracted_resized)
                outputs_classes = torch.zeros((translated_images.shape[0], config_3.num_classes)).to(device) # FIXME: change for classifier model


                # # Estimate classification loss.
                # if g == 0:
                #     loss_classification  = ce_loss(outputs_classes, targets_classes)
                # else:
                #     loss_classification += ce_loss(outputs_classes, targets_classes)


                # Current glimpse bounds.
                x_min_current   = (glimpses_locs_dims[:, 0]).clone().detach()
                x_max_current   = (glimpses_locs_dims[:, 0]+glimpses_locs_dims[:, 2]).clone().detach()
                y_min_current   = (glimpses_locs_dims[:, 1]).clone().detach()
                y_max_current   = (glimpses_locs_dims[:, 1]+glimpses_locs_dims[:, 3]).clone().detach()
                
                # Glimpse bounds IF their expansion is allowed with step dictated by config.glimpse_size_step.
                # !!!: Not needed during eval stage, remove if needed.
                x_min_future    = x_min_current - config_3.glimpse_size_step[0]
                x_max_future    = x_max_current + config_3.glimpse_size_step[0]
                y_min_future    = y_min_current - config_3.glimpse_size_step[1]
                y_max_future    = y_max_current + config_3.glimpse_size_step[1]
                
                # Estimate targets which are true IF the current glimpse can be expanded so that new glimpse dimensions are within bounding box.
                # !!!: Not needed during eval stage, remove if needed.
                x_min_target    = (x_min_future >= bbox_targets[:, 0]).float()
                x_max_target    = (x_max_future <= (bbox_targets[:, 0]+bbox_targets[:, 2])).float()
                y_min_target    = (y_min_future >= bbox_targets[:, 1]).float()
                y_max_target    = (y_max_future <= (bbox_targets[:, 1]+bbox_targets[:, 3])).float()

                # Estimate the loss.
                # !!!: Not needed during eval stage, remove if needed.
                glimpses_change_target      = torch.stack([x_min_target, x_max_target, y_min_target, y_max_target], dim=1)
                if g == 0:
                    loss_glimpse_dim_change     = bce_loss(glimpses_change_predictions, glimpses_change_target.clone().detach())
                    loss_glimpse_loc_change     = bce_loss(switch_location_predictions, init_glimpses_wrong.float().unsqueeze(1).clone().detach())
                else:
                    loss_glimpse_dim_change    += bce_loss(glimpses_change_predictions, glimpses_change_target.clone().detach())
                    loss_glimpse_loc_change    += bce_loss(switch_location_predictions, init_glimpses_wrong.float().unsqueeze(1).clone().detach())


                # Change glimpse dimensions according to model predictions.
                glimpses_change_probability = torch.sigmoid(glimpses_change_predictions.clone().detach())
                glimpses_change_actions     = (glimpses_change_probability >= config_3.glimpse_change_th)
                # Check so that glimpses do not go out of the image boundaries.
                x_min_new       = torch.clamp(x_min_current - glimpses_change_actions[:, 0]*config_3.glimpse_size_step[0], min=0)
                x_max_new       = torch.clamp(x_max_current + glimpses_change_actions[:, 1]*config_3.glimpse_size_step[0], max=config_3.full_res_img_size[1]) #(height, width) as used in transforms.Resize
                y_min_new       = torch.clamp(y_min_current - glimpses_change_actions[:, 2]*config_3.glimpse_size_step[1], min=0)
                y_max_new       = torch.clamp(y_max_current + glimpses_change_actions[:, 3]*config_3.glimpse_size_step[1], max=config_3.full_res_img_size[0]) #(height, width) as used in transforms.Resize
                
                # Store the new glimpse locations and dimensions.
                glimpses_locs_dims[:, 0] = x_min_new.clone().detach()
                glimpses_locs_dims[:, 1] = y_min_new.clone().detach()
                glimpses_locs_dims[:, 2] = x_max_new.clone().detach() - glimpses_locs_dims[:, 0]
                glimpses_locs_dims[:, 3] = y_max_new.clone().detach() - glimpses_locs_dims[:, 1]
                
                
                # Switch glimpse location according to model predictions.
                # NOTE: curriculum learning: switch location only if it is wrong initial location
                switch_location_probability     = torch.sigmoid(switch_location_predictions.clone().detach()).squeeze(1)
                switch_location_actions         = (switch_location_probability >= config_3.switch_location_th)
                switch_location_to_correct      = torch.logical_and(init_glimpses_wrong, switch_location_actions)
                
                # switch_location_to_correct is a mask of every sample for which CURRENT location was wrong AND network predicted it needs to SWITCH
                glimpses_locs_dims[switch_location_to_correct, 0]   = (init_glimpses_in_bbox[switch_location_to_correct, 0] + 0.5 - (config_3.glimpse_size_grid[0]/2.0)).int()
                glimpses_locs_dims[switch_location_to_correct, 1]   = (init_glimpses_in_bbox[switch_location_to_correct, 1] + 0.5 - (config_3.glimpse_size_grid[0]/2.0)).int()
                glimpses_locs_dims[switch_location_to_correct, 2]   = config_3.glimpse_size_init[0]
                glimpses_locs_dims[switch_location_to_correct, 3]   = config_3.glimpse_size_init[1]
                
                # Record if any of the locations was changed based on model predictions.
                init_glimpses_wrong[switch_location_to_correct] = False
                init_glimpses_correct = torch.logical_not(init_glimpses_wrong).clone().detach()
                
                if i == len(valid_loader) - 1:
                    glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())

        
            # Estimate the RL agent loss.
            loss                         = loss_glimpse_dim_change + loss_glimpse_loc_change #+ loss_classification


            _, pred_classes                  = torch.max(outputs_classes.data, 1)
            total_samples                   += targets_classes.size(0)
            test_loss                       += loss.item()
            test_loss_classification        += loss_classification.item()
            test_loss_glimpse_dim_change    += loss_glimpse_dim_change.item()
            test_loss_glimpse_loc_change    += loss_glimpse_loc_change.item()
            
            iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
            test_ave_iou                += iou.sum().item()
            
            correct_classes              = pred_classes == targets_classes
            correct_tp_loc               = iou >= config_3.iou_th
            

            acc_correct_class           += correct_classes.sum().item()
            acc_localization            += correct_tp_loc.sum().item()
            acc_class_localized         += correct_classes[correct_tp_loc].sum().item()
            acc_switching               += init_glimpses_correct.sum().item()
            # if i == len(valid_loader) - 1:
            #     extract_info_per_sample(images=translated_images, 
            #                             bbox_targets=bbox_targets, 
            #                             glimpses=glimpses_locs_dims_array, 
            #                             simscores=simscores,
            #                             rewards=rewards_array,
            #                             save_dir=config_3.save_dir, 
            #                             epoch=epoch+1, sample_id=4)
            #     extract_info_per_sample(images=translated_images, 
            #                             bbox_targets=bbox_targets, 
            #                             glimpses=glimpses_locs_dims_array, 
            #                             simscores=simscores,
            #                             rewards=rewards_array,
            #                             save_dir=config_3.save_dir, 
            #                             epoch=epoch+1, sample_id=0)
            
    print("Validation Loss: {:.3f} | {:.3f} | {:.3f} | {:.3f}, Switching Acc: {:.4f}\n".format(
        (test_loss/(i+1)), (test_loss_classification/(i+1)), (test_loss_glimpse_dim_change/(i+1)),
        (test_loss_glimpse_loc_change/(i+1)), (100.*acc_switching/total_samples)))

    print("Top-1 Cls: {:.4f} [{}/{}] | GT Loc: {:.4f} [{}/{}] | Top-1 Loc: {:.4f} [{}/{}]\n".format(
        (100.*acc_correct_class/total_samples), acc_correct_class, total_samples,
        (100.*acc_localization/total_samples), acc_localization, total_samples,
        (100.*acc_class_localized/total_samples), acc_class_localized, total_samples))
    log_dict['test_loss'].append(test_loss/(i+1))
    log_dict['test_loss_classification'].append(test_loss_classification/(i+1))
    log_dict['test_loss_glimpse_dim_change'].append(test_loss_glimpse_dim_change/(i+1))
    log_dict['test_loss_glimpse_loc_change'].append(test_loss_glimpse_loc_change/(i+1))
    log_dict['test_acc_correct_class'].append(100.*acc_correct_class/total_samples)
    log_dict['test_acc_localization'].append(100.*acc_localization/total_samples)
    log_dict['test_acc_class_localized'].append(100.*acc_class_localized/total_samples)
    print("IoU@{}: Average: {:.3f} | TPR: {:.4f} [{}/{}]\n".format(
        config_3.iou_th, (test_ave_iou/total_samples), 
        (1.*acc_localization/total_samples), acc_localization, total_samples))


    if optimizer.param_groups[0]['lr'] > config_3.lr_min:
        lr_scheduler.step()

    # Storing results
    ckpt = {}
    ckpt['model']   = model_3.state_dict()
    ckpt['log']     = log_dict
    ckpt['config_3'] = config_3_copy
    torch.save(ckpt, config_3.ckpt_dir)

# # Plotting statistics
# plot_dir = config_3.save_dir
# if not os.path.exists(plot_dir): os.makedirs(plot_dir)
# epochs = range(1, config_3.epochs+1)
# plot_curve(epochs, log_dict['train_loss'], 'training loss', 'epoch', 'train loss',     plot_dir + 'train_loss.png')
# plot_curve(epochs, log_dict['train_acc'],  'Performance',   'epoch', 'train accuracy', plot_dir + 'train_acc.png')
# plot_curve(epochs, log_dict['test_acc'],   'Performance',   'epoch', 'test accuracy',  plot_dir + 'test_acc.png')
# plot_curve(epochs, log_dict['test_loss'],  'testing loss',  'epoch', 'test loss',      plot_dir + 'test_loss.png')
    
    
#%%
#%matplotlib qt
# sample_id = 2
# imshow(translated_images[sample_id])
# plotregions(bbox_targets[sample_id].unsqueeze(0), color='r')


# plotregions(glimpses_locs_dims_array[0][sample_id].unsqueeze(0))
# plotregions(glimpses_locs_dims_array[1][sample_id].unsqueeze(0), color='darkorange')
# plotregions(glimpses_locs_dims_array[2][sample_id].unsqueeze(0), color='k')
# plotregions(glimpses_locs_dims_array[3][sample_id].unsqueeze(0), color='y')
# plotregions(glimpses_locs_dims_array[4][sample_id].unsqueeze(0), color='m')
# plotregions(glimpses_locs_dims_array[5][sample_id].unsqueeze(0), color='b')
# plotregions(glimpses_locs_dims_array[6][sample_id].unsqueeze(0), color='w')
# plotregions(glimpses_locs_dims_array[7][sample_id].unsqueeze(0), color='c')
# plotregions(glimpses_locs_dims_array[8][sample_id].unsqueeze(0))
# plotregions(glimpses_locs_dims_array[9][sample_id].unsqueeze(0), color='darkorange')
# plotregions(glimpses_locs_dims_array[10][sample_id].unsqueeze(0), color='k')
# plotregions(glimpses_locs_dims_array[11][sample_id].unsqueeze(0), color='y')
# plotregions(glimpses_locs_dims_array[12][sample_id].unsqueeze(0), color='m')
# plotregions(glimpses_locs_dims_array[13][sample_id].unsqueeze(0), color='b')
# plotregions(glimpses_locs_dims_array[14][sample_id].unsqueeze(0), color='w')
# plotregions(glimpses_locs_dims_array[15][sample_id].unsqueeze(0), color='c')
# plotregions(glimpses_locs_dims_array[16][sample_id].unsqueeze(0))
