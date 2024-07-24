import torch.utils.data as data
import torch
import torchvision
from PIL import Image
import os
import os.path
import numpy as np
import json
from torchvision.transforms import functional as F
import warnings
import random
import math
import copy
import numbers
import xml.etree.ElementTree as ET


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    return pil_loader(path)

class ResizedImgAndBBox(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        #resize to 256
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                img = copy.deepcopy(img)
                ow, oh = w, h
            if w < h:
                ow = size
                oh = int(size*h/w)
            else:
                oh = size
                ow = int(size*w/h)
        else:
            ow, oh = size[::-1]
            w, h = img.size


        intersec = copy.deepcopy(bbox)
        ratew = ow / w
        rateh = oh / h

        intersec[:, 0] = bbox[:, 0]*ratew
        intersec[:, 2] = bbox[:, 2]*ratew
        intersec[:, 1] = bbox[:, 1]*rateh
        intersec[:, 3] = bbox[:, 3]*rateh


        #intersec = normalize_intersec(i, j, h, w, intersec)
        return (oh, ow), intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        size, crop_bbox = self.get_params(img, bbox, self.size)
        return F.resize(img, self.size, self.interpolation), crop_bbox



class RandomHorizontalFlipImgAndBBox(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox):
        if random.random() < self.p:
            x_bound, _ = img.size
            # assumption: box is 4-sized tuple of (x1, y1, x2, y2) format            
            flipbox = copy.deepcopy(bbox)
            flipbox[:, 0] = x_bound - bbox[:, 2]
            flipbox[:, 2] = x_bound - bbox[:, 0]
            return F.hflip(img), flipbox
        return img, bbox






def load_all_test_bbox(img_filenames, bbox_dir):
    final_bbox_dict = {}
    for img_filename in img_filenames: # img_filename:  'ILSVRC2012_val_00007366.JPEG'
        filename = img_filename.split('.')[0] # filename:  'ILSVRC2012_val_00007366'
        xml_file = os.path.join(bbox_dir, filename + '.xml')
        
        tree = ET.parse(xml_file)
        objs = tree.findall('object')
        gt_boxes    = []
        gt_labels   = []
        for obj in objs:
            bbox        = obj.find('bndbox')
            cls_name    = obj.find('name').text
            x1 = float(bbox.find('xmin').text)-1
            y1 = float(bbox.find('ymin').text)-1
            x2 = float(bbox.find('xmax').text)-1
            y2 = float(bbox.find('ymax').text)-1
            
            gt_boxes.append((x1, y1, x2, y2))
            gt_labels.append(cls_name)
        final_bbox_dict[img_filename] = {"gt_boxes": gt_boxes, "gt_labels": gt_labels}
    return final_bbox_dict





class ImageNetDataset2013_detection(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, input_size=256, train=True, transform=None, target_transform=None, loader=default_loader):
        from torchvision.datasets import ImageFolder
        self.train              = train
        self.input_size         = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        
        if self.train:
            raise ValueError("This dataloader only supports validation split of ILSVRC2013 dataset")

        # first, we need to get synset list of ImageNet-1k (i.e. ILSVRC2012 dataset for classification)
        self.classification_dataset = ImageFolder('/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012/train/')
        self.label_class_dict = {}
        for k, v in self.classification_dataset.class_to_idx.items():
            self.label_class_dict[v] = k
        
        # second, fetch all filenames from ILSVRC2013-det val split
        self.val_img_dir     = '/home/nano01/a/tibrayev/imagenet/imagenet2013-detection/val/'
        self.val_anno_dir    = '/home/nano01/a/tibrayev/imagenet/imagenet2013-detection/anno_val/'
        self.img_filenames  = os.listdir(self.val_img_dir)
        self.img_filenames.sort()
        if len(self.img_filenames) == 0:
            raise(RuntimeError("Found 0 images in : " + self.val_img_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.bbox_dict_all = load_all_test_bbox(self.img_filenames, self.val_anno_dir)

        self.transform          = transform
        self.target_transform   = target_transform
        self.loader             = loader

        self.detection_synsets  = {}
        for i in range(len(self.img_filenames)):
            img_filename = self.img_filenames[i]
            objects         = self.bbox_dict_all[img_filename]
            bbox            = objects["gt_boxes"]
            bbox_labels     = objects["gt_labels"]
            for gt_label in bbox_labels:
                if gt_label in self.detection_synsets.keys():
                    self.detection_synsets[gt_label] += 1
                else:
                    self.detection_synsets[gt_label]  = 1
        self.detection_synsets_to_labels = {k: (v+1) for v, k in enumerate(self.detection_synsets.keys())}
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        img_filename = self.img_filenames[index]
        img = self.loader(os.path.join(self.val_img_dir, img_filename))
        
        
        objects         = self.bbox_dict_all[img_filename]
        bbox            = objects["gt_boxes"]
        bbox_labels     = objects["gt_labels"]
        
        w, h = img.size
        
        if len(bbox) > 0:
            bbox = np.array(bbox, dtype='float32')
            
            # During test, we expect bbox to be in (x1, y1, x2, y2) format already, 
            # but, not normalized: to be in range of image dimensions (width and height)
            img, bbox       = ResizedImgAndBBox(self.input_size)(img, bbox)
    
            #convert from (x1, y1, x2, y2) back to (x, y, w, h)
            bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
            bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
            bbox = torch.tensor(bbox)
        
        elif len(bbox) == 0:
            bbox_dummy = np.zeros((1, 4))
            img, bbox_dummy       = ResizedImgAndBBox(self.input_size)(img, bbox_dummy)

        if self.transform is not None:
            img = self.transform(img)
            
        # image, bounding box labels (one for each bounding box), and ground truth bounding boxes (multiple)
        return img, bbox_labels, bbox

        

    def __len__(self):
        return len(self.img_filenames)


if __name__ == '__main__':
    a =ImageNetDataset('/mnt/ramdisk/ImageNet/val/', train=False)
