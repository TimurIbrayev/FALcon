#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:59 2021

@author: tibrayev

Defines all hyperparameters.
"""

class FALcon_config(object):
    # SEED
    seed                    = 16
    
    # dataset
    dataset                 = 'imagenet'
    
    if dataset == 'cub':
        dataset_dir             = '/home/nano01/a/tibrayev/CUB_200-2011_raw'
        num_classes             = 200
        in_num_channels         = 3
        full_res_img_size       = (256, 256) #(height, width) as used in transforms.Resize
        correct_imbalance       = False
        selected_attributes     = ['all'] # obsolete, not used
        num_attributes          = 312 if 'all' in selected_attributes else len(selected_attributes) # obsolete, not used
        gt_bbox_dir             = None # Not needed, since CUB dataloader stores default ground truth bounding box dir
        wsol_method             = 'PSOL'
        pseudo_bbox_dir         = './{}/results/CUB_train_set/predicted_bounding_boxes/psol_predicted_bounding_boxes.txt'.format(wsol_method)
        loader_type                 = 'test'

    elif dataset == 'imagenet':
        dataset_dir             = '/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012'
        num_classes             = 1000
        in_num_channels         = 3
        full_res_img_size       = (256, 256) #(height, width) as used in transforms.Resize
        gt_bbox_dir             = dataset_dir + '/anno_val'
        wsol_method             = 'PSOL'
        pseudo_bbox_dir         = './{}/results/ImageNet_train_set/predicted_bounding_boxes/'.format(wsol_method)
        loader_random_seed      = 1
        valid_split_size        = 0.1 # should be in range [0, 1)
        loader_type                 = 'test'

    elif dataset == 'voc07':
        dataset_dir             = '/home/nano01/a/tibrayev/PascalVOC/VOC07/'
        dataset_year            = '2007'
        loader_type             = 'test'
        # for model config, just like for CUB
        num_classes             = 200
        in_num_channels         = 3
        full_res_img_size       = (256, 256) #(height, width) as used in transforms.Resize
    
    elif dataset == 'voc12':
        dataset_dir             = '/home/nano01/a/tibrayev/PascalVOC/VOC12/'
        dataset_year            = '2012'
        loader_type             = 'test-data'
        # for model config, just like for CUB
        num_classes             = 200
        in_num_channels         = 3
        full_res_img_size       = (256, 256) #(height, width) as used in transforms.Resize
    
    elif dataset == 'imagenet2013-det':
        dataset_dir             = '/home/nano01/a/tibrayev/imagenet/imagenet2013-detection'
        loader_type             = 'valid'
        num_classes             = 1000
        in_num_channels         = 3
        full_res_img_size       = (256, 256) #(height, width) as used in transforms.Resize

    else:
        raise ValueError("Received unknown dataset type request for running test on!")
        
    
    # cls model
    if dataset == 'cub' or dataset == 'voc07' or dataset == 'voc12':
        cls_model_name              = 'resnet50'
        cls_pretrained              = True
        cls_ckpt_dir                = './PSOL/results/PSOL/CUB/checkpoint_classification_cub_ddt_resnet50_99.pth.tar'
    elif dataset == 'imagenet' or dataset == 'imagenet2013-det':
        cls_model_name              = 'resnet50'
        cls_pretrained              = True
        cls_ckpt_dir                = None


    # FALcon model and experiment directory
    # (recommended to be the same as where your trained FALcon model is stored)
    if dataset == 'cub' or dataset == 'voc07' or dataset == 'voc12':
        save_dir                    = './results/cub/wsol_method_PSOL/trained_on_trainval_split_evaluated_on_test_split/arch_vgg11_pretrained_init_normalization_none_seed_16/'
        model_name                  = 'vgg11'
    elif dataset == 'imagenet' or dataset == 'imagenet2013-det':
        save_dir                    = './results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/'
        model_name                  = 'vgg16'
    initialize                  = 'resume_from_pretrained'
    ckpt_dir                    = save_dir + 'model.pth'
    batch_size_eval             = 50
    
    assert initialize in ['resume_from_pretrained', 'resume_from_random'], ...
    "Test script configuration only accepts 'resume' options ('resume_from_pretrained', 'resume_from_random') for model initialization and requires checkpoint at location specified by 'ckpr_dir'"
    initialize, init_factual    = initialize.split("_from_")
    
    if 'vgg' in model_name:
        downsampling            = 'M'
        fc1                     = 256
        fc2                     = 128
        dropout                 = 0.5
        norm                    = 'none'
        init_weights            = True
        adaptive_avg_pool_out   = (1, 1)
        saccade_fc1             = 256
        saccade_dropout         = False
        assert model_name in ['custom_vgg8_narrow_k2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        "Specify which VGG model to use for training. Options ('custom_vgg8_narrow_k2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn')"
        assert norm in ['none', 'batchnorm', 'evonorm'], ...
        "Specify which normalization type to use for normalization layers. Options ('batchnorm', 'instancenorm', 'layernorm', 'evonorm')"
    else:
        raise ValueError("Current test script supports only VGG-type models for FALcon model!")
    
    
    # FALcon-specific parameters
    num_glimpses            = 8*2
    fovea_control_neurons   = 4
    
    if dataset == 'cub' or dataset == 'voc07' or dataset == 'voc12':
        glimpse_size_grid       = (40, 40) #(width, height) of each grid when initially dividing image into grid cells
        glimpse_size_init       = (40, 40) #(width, height) size of initial foveation glimpse at the selected grid cell (usually, the same as above)
    elif dataset == 'imagenet' or dataset == 'imagenet2013-det':
        glimpse_size_grid       = (20, 20) #(width, height) of each grid when initially dividing image into grid cells
        glimpse_size_init       = (20, 20) #(width, height) size of initial foveation glimpse at the selected grid cell (usually, the same as above)
    glimpse_size_fixed      = (96, 96) #(width, height) size of foveated glimpse as perceived by the network
    glimpse_size_step       = (20, 20) #step size of foveation in (x, y) direction at each action in each (+dx, -dx, +dy, -dy) directions
    glimpse_change_th       = 0.5      #threshold, deciding whether or not to take the action based on post-sigmoid logit value 
    iou_th                  = 0.5
    # switching cell behavior
    ratio_wrong_init_glimpses   = 0.5 # ratio of the incorrect initial glimpses to the total glimpses in the batch
    switch_location_th          = 0.2
    objectness_based_nms_th     = 0.5
    confidence_based_nms_th     = 0.5
