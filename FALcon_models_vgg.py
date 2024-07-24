#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:55:07 2020
Verified on Wed May 25 2022

@modified by: tibrayev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.evonorm import EvoNormSample2d as enorm
from torch.hub import load_state_dict_from_url

# 'D' - stands for downsampling, for which choices are: 'M' - max pooling, 'A' - average pooling, 'C' - strided convolution
cfgs = {
    # first, custom ones, not present in original VGG paper:
    'custom_vgg6_narrow' :     [16, 'D', 32,  'D', 32,  'D'],
    'custom_vgg6' :            [64, 'D', 128, 'D', 128, 'D'],
    'custom_vgg9' :            [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D'],
    'custom_vgg8_narrow_k4':   [16, 'D', 32,  'D', 64,  'D', 128, 'D', 128],
    'custom_vgg8_narrow_k2':   [32, 'D', 64,  'D', 128, 'D', 256, 'D', 256],
    'custom_vgg11_narrow_k4':  [16, 'D', 32,  'D', 64,  64,  'D', 128, 128, 'D', 128, 128],
    'custom_vgg11_narrow_k2':  [32, 'D', 64,  'D', 128, 128, 'D', 256, 256, 'D', 256, 256],
    
    # next, default ones under VGG umbrella term:
    'vgg11':    [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg11_bn': [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg13':    [64, 64, 'D', 128, 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg13_bn': [64, 64, 'D', 128, 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg16':    [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 'D', 512, 512, 512, 'D', 512, 512, 512],
    'vgg16_bn': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 'D', 512, 512, 512, 'D', 512, 512, 512],
    'vgg19':    [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 256, 'D', 512, 512, 512, 512, 'D', 512, 512, 512, 512],
    'vgg19_bn': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 256, 'D', 512, 512, 512, 512, 'D', 512, 512, 512, 512],
}
    
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

   
class VGG(nn.Module):
    def __init__(self, config):
        super(VGG, self).__init__()

        # Dataset configuration
        self.dataset        = config.dataset
        self.num_classes    = config.num_classes
        self.in_channels    = config.in_num_channels
        self.in_feat_dim    = config.glimpse_size_fixed if isinstance(config.glimpse_size_fixed, tuple) else (config.glimpse_size_fixed, config.glimpse_size_fixed)


        # Network configuration
        self.vgg_name       = config.model_name.lower()
        if config.downsampling in ['M', 'A', 'C']: 
            self.downsampling   = config.downsampling
        else:
            raise ValueError("Error: Unknown downsampling. Choices are 'M' - max pooling, 'A' - average pooling, 'C' - strided convolution!") 
        self.fc1            = config.fc1
        self.fc2            = config.fc2
        self.dropout        = config.dropout
        self.norm           = config.norm
        self.feat_avg_pool  = config.adaptive_avg_pool_out if isinstance(config.adaptive_avg_pool_out, tuple) else (config.adaptive_avg_pool_out, config.adaptive_avg_pool_out)


        # Creating layers
        ### Feature extraction
        self.features, feature_channels, feature_dim    = self._make_feature_layers(cfgs[self.vgg_name])
        feature_flat_dims                               = feature_channels * feature_dim[0] * feature_dim[1]
        # ### Classifier(s)
        # self.classifier_classes                         = self._make_classifier_layers(feature_flat_dims, config.num_classes)
        # if config.num_attributes is None:
        #     self.classifier_attributes                  = None
        # else:
        #     self.classifier_attributes                  = self._make_classifier_layers(feature_flat_dims, config.num_attributes)
        ### Fovea control
        self.fovea_control_neurons                      = config.fovea_control_neurons
        self.fovea_control                              = self._make_fovea_layers(feature_flat_dims, self.fovea_control_neurons)
        ### Saccade control
        self.saccade_fc1                                = config.saccade_fc1
        self.saccade_dropout                            = config.saccade_dropout
        self.saccade_control                            = self._make_saccade_layers(feature_flat_dims, 1)
        
        # Weight initialization
        if config.init_weights: self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, enorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, with_latent = False):
        features                = self.features(x)
        features_flat           = torch.flatten(features, 1)

        # ### classifier predictions
        # outputs_for_classes             = self.classifier_classes(features_flat)
        # if self.classifier_attributes is None:
        #     outputs                     = (outputs_for_classes, None)
        # else:
        #     outputs_for_attributes      = self.classifier_attributes(features_flat)
        #     outputs                     = (outputs_for_classes, outputs_for_attributes)

        ### foveated glimpse change predictions
        glimpse_changes     = self.fovea_control(features_flat)
        
        ### switch location predictions
        switch_location     = self.saccade_control(features_flat)

        # Outputs
        if with_latent:
            return glimpse_changes, switch_location, features_flat
        else:
            return glimpse_changes, switch_location



    def _make_feature_layers(self, cfg):
        layers = []
        in_channels     = copy.deepcopy(self.in_channels)
        feature_dim     = copy.deepcopy(self.in_feat_dim)
        
        for v in cfg:
            if v == 'D':
                if self.downsampling == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'A':
                    layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'C':
                    layers += [nn.Conv2d(kernel_size=2, stride=2, bias=False)]
                feature_dim = tuple(f//2 for f in feature_dim)
                
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.norm.lower() == 'none':
                    layers += [conv2d, nn.ReLU(inplace=True)]
                elif self.norm.lower() == 'batchnorm':
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                elif self.norm.lower() == 'evonorm':
                    layers += [conv2d, enorm(v), nn.ReLU(inplace=True)]
                else:
                    raise ValueError("Received unknown type of normalization layer {}. Allowed types are: ('none', 'batchnorm', 'evonorm').".format(self.norm))
                in_channels = v
        
        layers += [nn.AdaptiveAvgPool2d(self.feat_avg_pool)]
        if (feature_dim[0]%self.feat_avg_pool[0] != 0) or (feature_dim[1]%self.feat_avg_pool[1] != 0):
            print("Warning! Expected feature size map is {}, but adaptive average pooling output requested is {},\n".format(feature_dim, self.feat_avg_pool) +
                  "meaning that some portion of the feature map will be dropped due to mismatch.")
            print("Consider changing the size of input {} or the size of adaptive average pooling output {} to process entire feature map!".format(self.in_feat_dim, self.feat_avg_pool))
        feature_dim = self.feat_avg_pool
        
        return nn.Sequential(*layers), in_channels, feature_dim


    def _make_classifier_layers(self, feature_flat_dims, num_output_nodes) :
        layers = []
        
        if self.fc1 == 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, num_output_nodes)]
        elif self.fc1 == 0 and self.fc2 != 0:
            raise ValueError("Received ambiguous pair of classifier parameters: fc1 = 0, but fc2 = {}. ".format(self.fc2) + 
                             "If only two FC layers are needed (including last linear classifier), please specify its dims as fc1 and set fc2=0.")
        elif self.fc1 != 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc1, num_output_nodes)]   
        else:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc1, self.fc2)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc2, num_output_nodes)]        
        return nn.Sequential(*layers)    


    def _make_fovea_layers(self, feature_flat_dims, num_output_nodes) :
        layers = []
        
        if self.fc1 == 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, num_output_nodes)]
        elif self.fc1 == 0 and self.fc2 != 0:
            raise ValueError("Received ambiguous pair of classifier parameters: fc1 = 0, but fc2 = {}. ".format(self.fc2) + 
                             "If only two FC layers are needed (including last linear classifier), please specify its dims as fc1 and set fc2=0.")
        elif self.fc1 != 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.fc1, num_output_nodes)]   
        else:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.fc1, self.fc2)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.fc2, num_output_nodes)]        
        return nn.Sequential(*layers)

    def _make_saccade_layers(self, feature_flat_dims, num_output_nodes) :
        layers = []
        
        layers += [nn.Linear(feature_flat_dims, self.saccade_fc1)]
        layers += [nn.ReLU(inplace=True)]
        if self.saccade_dropout:
            layers += [nn.Dropout(self.saccade_dropout)]
        layers += [nn.Linear(self.saccade_fc1, num_output_nodes)]
        return nn.Sequential(*layers)  



def customizable_VGG(config):
    """
    Mostly handles loading ImageNet pretrained models
    """
    model                   = VGG(config)
    model_name              = config.model_name.lower()
    assert model_name in cfgs.keys(), ...
    "Received the request for vgg model ({}), which is not supported by the current script. Supported vgg models are: ({})".format(model_name, cfgs.keys())
    
    if config.initialize == 'pretrained':
        if not model_name in model_urls.keys():
            print("Selected VGG configuration ({}) does not have ImageNet pretrained version! Initialized from scratch!\n").format(model_name)
        else:
            #load pretrained
            assert (('bn' in model_name) == (config.norm.lower() == 'batchnorm')), "Selected to load pretrained VGG of the configuration ({}), but expected batchnorm layer with setting (config.norm == '{}')".format(model_name, config.norm.lower())
            state_dict = load_state_dict_from_url(model_urls[model_name],
                                                  progress=True)
            model_sd = model.state_dict()
            for (k, v) in model_sd.items():
                if k in state_dict.keys():
                    if not (v.shape == state_dict[k].shape): #skip classifier layers when dimensions changed from default
                        print("Did not copy ({}) [shape: {}] from pretrained model's ({}) [shape: {}] due to size mismatch: Keeping Initialized Parameter!".format(
                            k, v.shape, k, state_dict[k].shape))
                    else:
                        model_sd[k] = state_dict[k].clone().detach()
                        # print("Copied ({}) from pretrained model's ({})".format(k, k))
                else:
                    print("Did not find ({}) in pretrained model state dictionary: Skipped!".format(k))
            model.load_state_dict(model_sd)
            print("Initialized the model with ImageNet pretrained version!")            
    elif config.initialize == 'random':
        print("Initialized from scratch!\n")
    elif config.initialize == 'resume':
        ckpt = torch.load(config.ckpt_dir)
        model.load_state_dict(ckpt['model'])
        print("Selected VGG configuration ({}) was loaded from checkpoint: {}\n".format(model_name, config.ckpt_dir))
    else:
        return ValueError("Unknown initialization method!")
    return model







def sim(vec_1, vec_2, temperature=1.0):
    # or, just use torch.cosine_similarity
    vec_1_normalized    = vec_1/(vec_1.norm(p=2, dim=-1, keepdim=True))
    vec_2_normalized    = vec_2/(vec_2.norm(p=2, dim=-1, keepdim=True))
    sim = torch.sum(vec_1_normalized * vec_2_normalized, dim=-1) / temperature
    return sim


def test():
    for d in ['MNIsT', 'FashionMnISt', 'Cifar10', 'CiFAR100']:
        for a in cfgs.keys():
            print("test under {} for {}".format(d.lower(), a))
            model = customizable_VGG(dataset=d, vgg_name=a)
            print("feature_flat_dims: {}".format(model.classifier[0].in_features))
            if d == 'MNIsT' or d == 'FashionMnISt':
                x = torch.rand(2, 1, 28, 28)
                y = model(x)
                assert y.shape == (2, 10)
            elif d == 'Cifar10':
                x = torch.rand(5, 3, 32, 32)
                y = model(x)
                assert y.shape == (5, 10)
            elif d == 'CiFAR100':
                x = torch.rand(3, 3, 32, 32)
                y = model(x)
                assert y.shape == (3, 100)

if __name__ == '__main__':
    test()
            