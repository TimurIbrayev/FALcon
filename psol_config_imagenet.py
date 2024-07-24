#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:59 2021

@author: tibrayev

Defines all hyperparameters.
"""

class psol_config(object):
    # SEED
    seed                    = 16
    
    # dataset
    dataset                 = 'imagenet2013-det'

    if dataset == 'imagenet':
        dataset_dir             = '/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012'
        num_classes             = 1000
        in_num_channels         = 3
        full_res_img_size       = (224, 224) #(height, width) as used in transforms.Resize
        gt_bbox_dir             = dataset_dir + '/anno_val'
        wsol_method             = 'PSOL'
        pseudo_bbox_dir         = './{}/results/ImageNet_train_set/predicted_bounding_boxes/'.format(wsol_method)
        loader_random_seed      = 1
        valid_split_size        = 0.1 # should be in range [0, 1)
        loader_type                 = 'test'

    elif dataset == 'imagenet2013-det':
        dataset_dir             = '/home/nano01/a/tibrayev/imagenet/imagenet2013-detection'
        loader_type             = 'valid'
        num_classes             = 1000
        in_num_channels         = 3
        full_res_img_size       = (224, 224) #(height, width) as used in transforms.Resize
    else:
        raise ValueError("Received unknown dataset type request for running test on!")
        
    
    # cls model
    cls_model_name              = 'resnet50'
    cls_pretrained              = True
    cls_ckpt_dir                = None

    batch_size_eval             = 50
    iou_th                      = 0.5

