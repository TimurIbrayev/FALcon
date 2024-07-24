#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:48:58 2023

@author: tibrayev
"""

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import numpy as np
import copy
import numbers
from collections.abc import Sequence
import math
import warnings

#%% ===========================================================================
#   Custom Torchvision functions
# =============================================================================
class custom_Compose:
    """
    CUSTOMIZATION: modified call to output boolean flags, indicating whether or not
    the image was (a) resized, (b) horizontally flipped.
    
    Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.supported_transforms_with_bounding_box = [custom_RandomHorizontalFlip, custom_Resize, custom_RandomResizedCrop]

    def __call__(self, img, bbox):
        # assumption 1: bbox is always provided, even when None
        # assumption 2: bbox is always in (x,y,w,h) format in the range of the image (i.e. when input, output, or between transforms)
        for t in self.transforms:
            if any(isinstance(t, transform_with_bounding_box) for transform_with_bounding_box in self.supported_transforms_with_bounding_box):
                img, bbox = t(img, bbox)
            else:
                img = t(img)
        return img, bbox

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class custom_RandomHorizontalFlip(torch.nn.Module):
    """
    CUSTOMIZATION: modified forward to output boolean flag, indicating whether or not
    the input image was flipped.
    
    Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, bbox):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if bbox is None:
                return F.hflip(img), bbox
            else:
                if torch.is_tensor(img):
                    channels, h, w = img.shape
                else:
                    w, h = img.size
                bbox_flipped = self.hflip_box(bbox, w)
                return F.hflip(img), bbox_flipped
        else:
            return img, bbox
        
    def hflip_box(self, box, x_bound):
        assert ((box.ndim == 2) and (box.shape[1] == 4)), "horizontal flip expects 2 dimensional tensor with the second dimension being 4 sized tuple for (x, y, width, height)"
        # assumption: box is 4-sized tuple of (x, y, width, height)
        reducer = torch.zeros_like(box)
        reducer[:, 0] = x_bound - box[:, 2]
        return torch.abs(box - reducer)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    

class custom_Resize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set True for
            ``InterpolationMode.BILINEAR`` only mode.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.

    """

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = self._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = F._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def _setup_size(self, size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)
    
        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]
    
        if len(size) != 2:
            raise ValueError(error_msg)
        return size

    def forward(self, img, bbox):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if not bbox is None:
            if torch.is_tensor(img):
                channels, h, w = img.shape
            else:
                w, h = img.size
            bbox_resized = self.resize_boxes(bbox, (h, w), self.size)
        else:
            bbox_resized = bbox
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias), bbox_resized
    
    def resize_boxes(self, boxes, original_size, new_size):
        assert ((boxes.ndim == 2) and (boxes.shape[1] == 4)), "resize_boxes expects 2 dimensional tensor with the second dimension being 4 sized tuple for (x, y, width, height)"
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        # FYI: here assumption that there is multiple boxes input. Change to unbind(0) if only single box is provided at once, i.e. boxes.ndim=1
        xmin, ymin, width, height = boxes.unbind(1)
    
        xmin = xmin * ratio_width
        width = width * ratio_width
        ymin = ymin * ratio_height
        height = height * ratio_height
        # FYI: here assumption that there is multiple boxes input. Change to torch.stack(..., dim=0) if only single box is provided at once, i.e. boxes.ndim=1
        return torch.stack((xmin, ymin, width, height), dim=1)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)


class custom_RandomResizedCrop(torch.nn.Module):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, bbox_area_tolerance=0.5, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = self._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = F._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.bbox_area_tolerance = bbox_area_tolerance

    def _setup_size(self, size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)
    
        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]
    
        if len(size) != 2:
            raise ValueError(error_msg)
        return size

    def get_params(self, img, scale, ratio, bbox):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        if torch.is_tensor(img):
            channels, height, width = img.shape
        else:
            width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(40):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio))) # new width
            h = int(round(math.sqrt(target_area / aspect_ratio))) # new height

            if 0 < w <= width and 0 < h <= height:
                
                i = torch.randint(0, height - h + 1, size=(1,)).item() # new y
                j = torch.randint(0, width - w + 1, size=(1,)).item()  # new x
                
                if bbox is None:
                    return i, j, h, w, bbox
                else:
                    area_original, area_new, bbox_new = self.calibrate_bbox(i, j, h, w, bbox)
                    if torch.all(area_new >= self.bbox_area_tolerance * area_original): # want to preserve some fraction of the original area of the bounding box!
                        return i, j, h, w, bbox_new # bbox_new in (x,y,w,h) format in the range of (j,i,w,h) (i.e. in the range of a crop dimensions)

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        if bbox is None:
            return i, j, h, w, bbox
        else:
            _, _, bbox_new = self.calibrate_bbox(i, j, h, w, bbox)
            return i, j, h, w, bbox_new
    
    
    def calibrate_bbox(self, i, j, h, w, bbox):
        assert ((bbox.ndim) == 2 and (bbox.shape[1] == 4)), "calibrate_bbox expects 2 dimensional tensor with the second dimension being 4 sized tuple for (x, y, width, height)"
        area_original = bbox[:, 2] * bbox[:, 3]
        # combination of compute_intersec and normalize_intersec provided by PSOL authors
        bbox_new = copy.deepcopy(bbox)
        # adjust coordinates based on the coordinates of image crop points
        bbox_new[:, 0] = torch.maximum(torch.tensor([j]), bbox[:, 0]) # new x1
        bbox_new[:, 1] = torch.maximum(torch.tensor([i]), bbox[:, 1]) # new y1
        bbox_new[:, 2] = torch.minimum(torch.tensor([j + w]), bbox[:, 2] + bbox[:, 0]) # new x2
        bbox_new[:, 3] = torch.minimum(torch.tensor([i + h]), bbox[:, 3] + bbox[:, 1]) # new y2
        
        # adjust coordinate to be w.r.t. new (0, 0)
        bbox_new[:, 0] -= j
        bbox_new[:, 1] -= i
        bbox_new[:, 2] -= j
        bbox_new[:, 3] -= i
        # convert from (x1y1x2y2) to (xywh)
        bbox_new[:, 2] -= bbox_new[:, 0]
        bbox_new[:, 3] -= bbox_new[:, 1]
        area_new = bbox_new[:, 2]*bbox_new[:, 3]
        return area_original, area_new, bbox_new
    
    def resize_boxes(self, boxes, original_size, new_size):
        assert ((boxes.ndim == 2) and (boxes.shape[1] == 4)), "resize_boxes expects 2 dimensional tensor with the second dimension being 4 sized tuple for (x, y, width, height)"
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        # FYI: here assumption that there is multiple boxes input. Change to unbind(0) if only single box is provided at once, i.e. boxes.ndim=1
        xmin, ymin, width, height = boxes.unbind(1)
    
        xmin = xmin * ratio_width
        width = width * ratio_width
        ymin = ymin * ratio_height
        height = height * ratio_height
        # FYI: here assumption that there is multiple boxes input. Change to torch.stack(..., dim=0) if only single box is provided at once, i.e. boxes.ndim=1
        return torch.stack((xmin, ymin, width, height), dim=1)
        
    def forward(self, img, bbox): # Main assumption: bbox is in (x,y,w,h) format in the image range
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w, bbox_cropped = self.get_params(img, self.scale, self.ratio, bbox)
        # NEED TO RESIZE bbox_cropped from the size of a crop to the size of resized crop
        if bbox_cropped is None:
            bbox_new_resized = None
        else:
            bbox_new_resized = self.resize_boxes(bbox_cropped, (h, w), self.size)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), bbox_new_resized

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


#%% ===========================================================================
#   Dataloader functions
# =============================================================================
# from dataloader_celebA_inthewild import CelebA_inthewild # HAVENT BEEN CALIBRATED
from dataloader_cub import CUBirds_2011
from dataloader_imagenet import ImageNetDataset
from dataloader_voc import VOCDetection
from dataloader_imagenet2013_det_valonly import ImageNetDataset2013_detection

def get_dataloaders(config, loader_type='train'):
    assert config.dataset.lower() in ['celeba', 'cub', 'imagenet', 'voc07', 'voc12', 'imagenet2013-det'], "Received unsupported type of dataset!"

    ### CelebA dataset  
    if config.dataset.lower() == 'celeba':
        raise NotImplementedError("CelebA is not implemented for the current version of the framework!")
        # assert loader_type in ['train', 'valid', 'test', 'all', 'trainval'], "Received unsupported type of dataset split!"
        # selected_attributes             = config.selected_attributes
        # correct_imbalance               = config.correct_imbalance
        # at_least_true_attributes        = config.at_least_true_attributes
        # treat_attributes_as_classes     = config.treat_attributes_as_classes
        
        # if loader_type == 'train' or loader_type == 'trainval':
        #     transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
        #                                 custom_RandomHorizontalFlip(p=0.5),
        #                                 transforms.ToTensor()])
        #     data    = CelebA_inthewild(config.dataset_dir, 
        #                                split                            = loader_type, 
        #                                target_type                      = ['attr', 'bbox'], 
        #                                transform                        = transform, 
        #                                selected_attributes              = selected_attributes,
        #                                at_least_true_attributes         = at_least_true_attributes, 
        #                                treat_attributes_as_classes      = treat_attributes_as_classes)
        #     loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True,
        #                                           num_workers=10, pin_memory=False)
            
        #     if correct_imbalance:
        #         # Since CelebA is imbalanced dataset, during training we can provide weights to the BCEloss
        #         # First, count how many positive and negative samples are there for each attribute
        #         cnt_pos_attr = data.sel_attr.sum(dim=0)
        #         cnt_neg_attr = data.sel_attr.shape[0] - cnt_pos_attr
        #         # Then, divide the number of negative samples by the number of positive samples to have scaling factor
        #         # for each individual attribute. As a result, you will effectively (from the perspective of loss)
        #         # have the same number of positive examples as that of negative examples.
        #         # See documentation of BCELoss for more details.
        #         pos_weight   = cnt_neg_attr*1.0/cnt_pos_attr

        #         print("Attempt to correct imbalance in attributes distribution!")
        #         print("Number of positive samples per attribute:")
        #         print("{}".format(cnt_pos_attr))
        #         print("Positive weights are:")
        #         print("{}".format(pos_weight))
        #         return loader, pos_weight
        #     else:
        #         equal_weights = torch.tensor([1.0 for _ in range(config.num_classes)])
        #         return loader, equal_weights

            
        # elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
        #     transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
        #                                 transforms.ToTensor()])
        #     data    = CelebA_inthewild(config.dataset_dir, 
        #                                split                            = loader_type, 
        #                                target_type                      = ['attr', 'bbox'], 
        #                                transform                        = transform, 
        #                                selected_attributes              = selected_attributes,
        #                                at_least_true_attributes         = at_least_true_attributes, 
        #                                treat_attributes_as_classes      = treat_attributes_as_classes)
        #     loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
        #     return loader


    ### Birds dataset
    elif config.dataset.lower() == 'cub':
        assert loader_type in ['train', 'valid', 'test', 'all', 'trainval'], "Received unsupported type of dataset split!"
        correct_imbalance               = config.correct_imbalance
        
        if loader_type == 'train' or loader_type == 'trainval':
            transform   = custom_Compose([custom_Resize(size=config.full_res_img_size),
                                          custom_RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            data        = CUBirds_2011(root                             = config.dataset_dir, 
                                       pseudo_bbox_dir                  = config.pseudo_bbox_dir,
                                       split                            = loader_type, 
                                       target_type                      = ['class', 'pseudo_bbox'], 
                                       transform                        = transform)
            loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True)

            if correct_imbalance:
                # Since Birds is imbalanced dataset, during training we can provide weights to the BCEloss
                # First, count how many positive and negative samples are there for each attribute
                cnt_pos_attr = data.sel_attr.sum(dim=0)
                cnt_neg_attr = data.sel_attr.shape[0] - cnt_pos_attr
                # Then, divide the number of negative samples by the number of positive samples to have scaling factor
                # for each individual attribute. As a result, you will effectively (from the perspective of loss)
                # have the same number of positive examples as that of negative examples.
                # See documentation of BCELoss for more details.
                pos_weight   = cnt_neg_attr*1.0/cnt_pos_attr

                print("Attempt to correct imbalance in attributes distribution!")
                print("Number of positive samples per attribute:")
                print("{}".format(cnt_pos_attr))
                print("Positive weights are:")
                print("{}".format(pos_weight))
                return loader, pos_weight
            else:
                equal_weights = torch.tensor([1.0 for _ in range(config.num_classes)])
                return loader, equal_weights
        
        elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
            transform   = custom_Compose([custom_Resize(size=config.full_res_img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            data        = CUBirds_2011(root                             = config.dataset_dir, 
                                       split                            = loader_type, 
                                       target_type                      = ['class', 'gt_bbox'],
                                       transform                        = transform)
            loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
            return loader



    ### ImageNet dataset
    elif config.dataset.lower() == 'imagenet':
        assert loader_type in ['train', 'valid', 'test', 'all', 'trainval', 'train_and_val'], "Received unsupported type of dataset split!"
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        
        if loader_type == 'train_and_val': # split ImageNet train set into train and val sets for hyperparameter search
            train_data  = ImageNetDataset(root                  = config.dataset_dir,
                                          pseudo_bbox_dir       = config.pseudo_bbox_dir,
                                          gt_bbox_dir           = None,
                                          train                 = True,
                                          input_size            = config.full_res_img_size,
                                          transform             = transform)
            
            valid_data  = ImageNetDataset(root                  = config.dataset_dir,
                                          pseudo_bbox_dir       = config.pseudo_bbox_dir,
                                          gt_bbox_dir           = None,
                                          train                 = True,
                                          input_size            = config.full_res_img_size,
                                          transform             = transform)
            
            print('\nForming the samplers for train and validation splits with split fraction={}'.format(config.valid_split_size))
            num_train_samples = len(train_data)
            indices = list(range(num_train_samples))
            split = int(np.floor(config.valid_split_size * num_train_samples))
    
            np.random.seed(config.loader_random_seed)
            np.random.shuffle(indices)
        
            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)    

            print('Preparing dataloaders...\n')
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size_train, sampler=train_sampler, num_workers=8, pin_memory=True)
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size_train, sampler=valid_sampler, num_workers=8, pin_memory=True)
            return train_loader, valid_loader


        elif loader_type == 'train': # entire ImageNet train set
            train_data  = ImageNetDataset(root                  = config.dataset_dir,
                                          pseudo_bbox_dir       = config.pseudo_bbox_dir,
                                          gt_bbox_dir           = None,
                                          train                 = True,
                                          input_size            = config.full_res_img_size,
                                          transform             = transform)

            print('Preparing dataloader...\n')
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size_train, shuffle=True, num_workers=8, pin_memory=True)
            return train_loader


        elif loader_type == 'test': # entire ImageNet val set
            test_data  = ImageNetDataset(root                  = config.dataset_dir,
                                         pseudo_bbox_dir       = None,
                                         gt_bbox_dir           = config.gt_bbox_dir, # default: os.path.join(config.dataset_dir, 'anno_val')
                                         train                 = False,
                                         input_size            = config.full_res_img_size,
                                         transform             = transform)

            print('Preparing dataloader...\n')
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size_eval, shuffle=False, num_workers=8, pin_memory=True)
            return test_loader


        elif loader_type in ['trainval', 'valid']:
            print("Current loader for ImageNet supports either 'train' for entire ImageNet train set or 'train_and_val' for ImageNet train set split into train and val sets for hyperparameter search!")
        elif loader_type == 'all':
            print("Current loader for ImageNet does not support 'all' loader_type! Supported loader_types are ('train_and_val', 'train', 'test').")



    ### PascalVOC dataset
    elif 'voc' in config.dataset.lower():
        assert loader_type in ['train', 'trainval', 'valid', 'test', 'test-data'], "Received unsupported type of dataset split!"
        assert config.dataset_year in ['2007', '2012'], "Current framework supports only VOC07 and VOC12, but received request for VOC of {} year!".format(config.dataset_year)
        
        if loader_type in ['train', 'trainval']:
            transform   = custom_Compose([custom_Resize(size=config.full_res_img_size),
                                          custom_RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            data        = VOCDetection(root                 = config.dataset_dir,
                                       year                 = config.dataset_year,
                                       image_set            = loader_type,
                                       download             = False,
                                       transforms           = transform)
            loader = data
            # FIXME: add collate_fn to extract data in batches
            # loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True)
            return loader
        elif loader_type == 'valid':
            transform   = custom_Compose([custom_Resize(size=config.full_res_img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            data        = VOCDetection(root                 = config.dataset_dir,
                                       year                 = config.dataset_year,
                                       image_set            = 'val',
                                       download             = False,
                                       transforms           = transform)
            loader = data
            # loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
            return loader
        elif loader_type == 'test' and config.dataset_year == '2007':
            transform   = custom_Compose([custom_Resize(size=config.full_res_img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            data        = VOCDetection(root                 = config.dataset_dir,
                                       year                 = config.dataset_year,
                                       image_set            = 'test',
                                       download             = False,
                                       transforms           = transform)
            loader = data
            # loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
            return loader
        elif loader_type == 'test-data' and config.dataset_year == '2012':
            transform   = custom_Compose([custom_Resize(size=config.full_res_img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            data        = VOCDetection(root                 = config.dataset_dir,
                                       year                 = config.dataset_year,
                                       image_set            = 'test-data',
                                       download             = False,
                                       transforms           = transform)
            loader = data
            # loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
            return loader
        else:
            raise ValueError("Received unexpected combination of loader_type={} and dataset_year={} for VOC dataset!".format(loader_type, config.dataset_year))

            
    ### ImageNet-2013-Detection dataset
    elif config.dataset.lower() == 'imagenet2013-det':
        assert loader_type in ['valid'], "Received unsupported type of dataset split!"
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        

        test_data  = ImageNetDataset2013_detection(root                  = None,
                                                   train                 = False,
                                                   input_size            = config.full_res_img_size,
                                                   transform             = transform)
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size_eval, shuffle=False, num_workers=8, pin_memory=True)
        return test_loader
