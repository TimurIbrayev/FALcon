#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:40:10 2022
Verified on May 25 2022


@author: tibrayev
"""

from collections import namedtuple
import csv
from functools import partial
from itertools import compress
import torch
import os
import PIL
import copy
import numbers
from collections.abc import Sequence
import math

from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision import transforms
import torchvision.transforms.functional as F


CSV = namedtuple("CSV", ["header", "index", "data"])


def resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    # FYI: here assumption that there is only one box input. Change to unbind(1) if multiple boxes are provided at once.
    xmin, ymin, width, height = boxes.unbind(0)

    xmin = xmin * ratio_width
    width = width * ratio_width
    ymin = ymin * ratio_height
    height = height * ratio_height
    return torch.stack((xmin, ymin, width, height), dim=0)

def hflip_box(box, x_bound):
    # assumption: box is 4-sized tuple of (x, y, width, height)
    reducer = torch.zeros_like(box)
    reducer[0] = x_bound - box[2]
    return torch.abs(box - reducer)      



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
                bbox_flipped = hflip_box(bbox, w)
                return F.hflip(img), bbox_flipped
        else:
            return img, bbox

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
            bbox_resized = resize_boxes(bbox, (h, w), self.size)
        else:
            bbox_resized = bbox
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias), bbox_resized

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
                    if area_new >= self.bbox_area_tolerance * area_original: # want to preserve some fraction of the original area of the bounding box!
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
        area_original = bbox[2] * bbox[3]
        # combination of compute_intersec and normalize_intersec provided by PSOL authors
        bbox_new = copy.deepcopy(bbox)
        # adjust coordinates based on the coordinates of image crop points
        bbox_new[0] = max(j, bbox[0]) # new x1
        bbox_new[1] = max(i, bbox[1]) # new y1
        bbox_new[2] = min(j + w, bbox[2] + bbox[0]) # new x2
        bbox_new[3] = min(i + h, bbox[3] + bbox[1]) # new y2
        
        # adjust coordinate to be w.r.t. new (0, 0)
        bbox_new[0] -= j
        bbox_new[1] -= i
        bbox_new[2] -= j
        bbox_new[3] -= i
        # convert from (x1y1x2y2) to (xywh)
        bbox_new[2] -= bbox_new[0]
        bbox_new[3] -= bbox_new[1]
        area_new = bbox_new[2]*bbox_new[3]
        return area_original, area_new, bbox_new
        
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
            bbox_new_resized = resize_boxes(bbox_cropped, (h, w), self.size)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), bbox_new_resized

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string





class CUBirds_2011(VisionDataset):
    """`Caltech-UCSD Birds-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>` Dataset.
    
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``class``, ``attr``, ``bbox``,
            or ``parts (Not implemented!)``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                
                - ``class`` (int): one of 200 classes images are categorized into
                - ``attr`` (np.array shape=(312,) dtype=int): binary (0, 1) labels for attributes
                - ``bbox`` (np.array shape=(4,) dtype=float): bounding box (x, y, width, height)
                - ``parts`` (Not Implemented!): (x, y) coordinates of parts of objects
            
            Defaults to ``class``.
        
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
            
        This dataloader assumes that the data is already downloaded and unzipped in the root directory provided.
    """
    
    base_folder = "CUB_200_2011"
    
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "class",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pseudo_bbox_dir = None,
    ) -> None:

        # in order to take into account possibility of image being resized and/or flipped,
        # and hence, requiring bounding box to be reszied and/or flipped accordingly,
        # this dataloader works only with custom_Compose transform (see in custom_tvision_utils.py)
        if (transform is not None) and (not isinstance(transform, custom_Compose)):
            # in case transform is not wrapped into transforms.Compose,
            # then, we can simply wrap it into custom_Compose
            if not isinstance(transform, transforms.Compose):
                if isinstance(transform, list):
                    transform = custom_Compose(transform) # transform is already a list of transforms
                else:
                    transform = custom_Compose([transform]) # transform is only a single transform (e.g. transforms.ToTensor())
            # else, we assume that the transform is already wrapped into transforms.Compose 
            # and throw error
            else:
                raise ValueError("Expected either list of transforms or set of transforms wrapped into custom_Compose")        

        super(CUBirds_2011, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        
        self.split = split
        
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        if "parts" in self.target_type:
            raise NotImplementedError("target_type 'parts' is not implemented by the current loader!")
        if not self.target_type: # in case of empty list as target_type
            self.target_type.append("class")
        
        split_map = {
            "train": 1,
            "test": 0,
            "valid": 2,
            "trainval": (1, 2),
            "all": None}
        
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "test", "valid", "all", "trainval"))]
        
        # fetching.
        splits          = self._load_csv("train_test_val_split.txt",data_type='int')
        filename        = self._load_csv("images.txt",              data_type='str')
        labels          = self._load_csv("image_class_labels.txt",  data_type='int')
        label_names     = self._load_csv("classes.txt",             data_type='str')
        gt_bboxes       = self._load_csv("bounding_boxes.txt",      data_type='float')
        attr            = self._load_csv("attributes/image_attribute_labels.txt", data_type='int', specific_columns=[1, 3])
        attr_names      = self._load_csv("attributes/attributes.txt", data_type='str')    
        if not pseudo_bbox_dir is None:
            self.pseudo_bbox_dir    = pseudo_bbox_dir
            self.pseudo_bboxes      = self._load_pseudo_boxes(self.pseudo_bbox_dir) # expected format: "filename": [x,y,w,h]


        # pre-processing.
        if split_ is None:
            mask            = slice(None)
            self.filename   = [fname[0] for fname in filename.data]
        else:
            if isinstance(split_, tuple):
                mask            = torch.logical_or((splits.data == split_[0]).squeeze(), (splits.data == split_[1]).squeeze())
            else:
                mask            = (splits.data == split_).squeeze()
            self.filename   = [fname[0] for fname, m in zip(filename.data, mask) if m]
        self.labels         = (labels.data[mask] - 1).squeeze() # dataset labels start from 1
        self.label_names    = {int(k)-1: v[0] for k, v in zip(label_names.index, label_names.data)}
        self.gt_bboxes      = gt_bboxes.data[mask]
        self.attr           = attr.data.reshape(len(filename.index), len(attr_names.index), 2)[:, :, -1][mask]
        self.attr_names     = {int(k)-1: v[0] for k, v in zip(attr_names.index, attr_names.data)}
        

    
    def _load_pseudo_boxes(self, path_to_boxes):
        filenamed_pseudo_bboxes = {}
        with open(path_to_boxes) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        for data_line in data:
            filenamed_pseudo_bboxes[data_line[0]] = [float(data_line[i]) for i in range(1, 5)]
        return filenamed_pseudo_bboxes

    def _load_csv(
                        self,
                        filename: str,
                        header: Optional[int] = None,
                        data_type: Optional[str] = 'int',
                        specific_columns: Optional[List[int]] = None,
                  ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        if specific_columns is not None:        
            data = [row[specific_columns[0]:specific_columns[1]] for row in data]
        else:
            data = [row[1:] for row in data]
        if data_type=='int':
            data_int = [list(map(int, i)) for i in data]
            return CSV(headers, indices, torch.tensor(data_int))
        elif data_type=='float':
            data_int = [list(map(int, map(float, i))) for i in data]
            return CSV(headers, indices, torch.tensor(data_int))
        elif data_type=='str':
            return CSV(headers, indices, data)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "images", self.filename[index]))
        
        if X.mode != 'RGB': # some of the images in the birds dataset are black and white
            X = X.convert("RGB")
        
        if any(target in ('gt_bbox', 'pseudo_bbox') for target in self.target_type):
            assert (not (('gt_bbox' in self.target_type) and ('pseudo_bbox' in self.target_type))), "Only gt_bbox or pseudo_bbox can be requested at a time, but requested both!"
            if 'gt_bbox' in self.target_type:
                bbox = self.gt_bboxes[index, :]
            elif 'pseudo_bbox' in self.target_type:
                bbox = torch.tensor(self.pseudo_bboxes[self.filename[index]])
        else:
            bbox = None        

        w, h = X.size
        if self.transform is not None:
            X, bbox = self.transform(X, bbox)
        
        target: Any = []
        for t in self.target_type:
            if t == "class":
                target.append(self.labels[index])
            elif t == "attr":
                target.append(self.attr[index, :])
            elif t == "gt_bbox":
                target.append(bbox)   
            elif t == 'filename':
                target.append(self.filename[index])
            elif t == 'pseudo_bbox':
                target.append(bbox)
            elif t == 'original_wh':
                target.append(torch.tensor([w, h]))
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        target = tuple(target) if len(target) > 1 else target[0]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return X, target
    
    def __len__(self) -> int:
        return len(self.filename)
    
    def extra_repr(self) -> str:
        lines = ["target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
    
    
# def split_train_into_train_and_valid():
#     data    = CUBirds_2011('/home/nano01/a/tibrayev/CUB_200-2011_raw/', split = 'train', target_type = ['class', 'bbox'])
#     splits  = data._load_csv("train_test_split.txt", data_type='int')
#     labels  = data._load_csv("image_class_labels.txt",  data_type='int')    
#     sample_cnt_per_class = {k: (data.labels==k).sum().item() for k in range(200)}
#     num_of_samples_for_validation = 3
#     selected_for_validation = []
#     sample_cnt_preceding = 0
#     for c in range(200):
#         selected_indices = torch.randperm(sample_cnt_per_class[c])[:num_of_samples_for_validation].sort()[0]
#         selected_for_validation.append(selected_indices + sample_cnt_preceding)
#         sample_cnt_preceding += sample_cnt_per_class[c]
#     selected_for_validation = torch.cat(selected_for_validation)
#     print("Samples selected for validation are: \n{}".format(selected_for_validation))
    
#     f = open('/home/nano01/a/tibrayev/CUB_200-2011_raw/CUB_200_2011/train_test_val_split.txt', 'w', buffering=1)
#     cnt_train_samples_observed_so_far = 0
#     cnt_valid_samples = 0
#     valid_cnt_per_class = {k: 0 for k in range(200)} #for sanity check
#     for l in range(len(splits.index)):
#         index = splits.index[l]
#         value = splits.data[l].item()
#         if value == 1:
#             if cnt_train_samples_observed_so_far in selected_for_validation:
#                 value = 2
#                 cnt_valid_samples += 1
#                 valid_cnt_per_class[labels.data[l].item()-1] += 1
#             cnt_train_samples_observed_so_far += 1
#         f.write("{} {}\n".format(index, value))
#     print("Completed split!")
#     f.close()
    
#     # for sanity check
#     splits_new  = data._load_csv("train_test_val_split.txt", data_type='int')
#     mask        = (splits_new.data == 1).squeeze()
#     labels_new  = (labels.data[mask] - 1).squeeze()
#     train_cnt_per_class = {k: (labels_new==k).sum().item() for k in range(200)}
#     for s, v, t in zip(sample_cnt_per_class.items(), valid_cnt_per_class.items(), train_cnt_per_class.items()):
#         assert s[1] == v[1] + t[1]

    








