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

from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision import transforms
import torchvision.transforms.functional as F

CSV = namedtuple("CSV", ["header", "index", "data"])

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

    def __call__(self, img):
        resized     = False
        hflipped    = False
        for t in self.transforms:
            if isinstance(t, custom_RandomHorizontalFlip):
                img, hflipped = t(img)              
            elif isinstance(t, transforms.Resize):
                resized = True
                img = t(img)
            else:
                img = t(img)
        return img, resized, hflipped

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

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), True
        return img, False

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


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
        
        w, h = X.size
        if self.transform is not None:
            X, resized, hflipped = self.transform(X)
            if torch.is_tensor(X):
                channels, h_new, w_new = X.shape
            else:
                w_new, h_new = X.size
        else:
            resized = hflipped = False
            w_new, h_new = w, h
        
        target: Any = []
        for t in self.target_type:
            if t == "class":
                target.append(self.labels[index])
            elif t == "attr":
                target.append(self.attr[index, :])
            elif t == "gt_bbox":
                bbox = self.gt_bboxes[index, :]
                if resized:
                    bbox = resize_boxes(bbox, (h, w), (h_new, w_new))
                if hflipped:
                    bbox = hflip_box(bbox, w_new)
                target.append(bbox)   
            elif t == 'filename':
                target.append(self.filename[index])
            elif t == 'pseudo_bbox':
                bbox = torch.tensor(self.pseudo_bboxes[self.filename[index]])
                if resized:
                    bbox = resize_boxes(bbox, (h, w), (h_new, w_new))
                if hflipped:
                    bbox = hflip_box(bbox, w_new)
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
            

    








