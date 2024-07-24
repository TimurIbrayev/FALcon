import os
import collections
from torchvision.datasets.vision import VisionDataset
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import warnings
import torch

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': os.path.join('TrainVal', 'VOCdevkit', 'VOC2011')
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': os.path.join('VOCdevkit', 'VOC2010')
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': os.path.join('VOCdevkit', 'VOC2009')
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': os.path.join('VOCdevkit', 'VOC2008')
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    },
    '2007-test': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    }
}


class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        if year == "2007-test":
            if image_set == "test":
                warnings.warn(
                    "Acessing the test image set of the year 2007 with year='2007-test' is deprecated. "
                    "Please use the combination year='2007' and image_set='test' instead."
                )
                year = "2007"
            else:
                raise ValueError(
                    "In the test image set of the year 2007 only image_set='test' is allowed. "
                    "For all other image sets use year='2007' instead."
                )
        self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        elif year == '2012':
            valid_image_sets.append("test-data")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        key = "2012-test-data" if year == "2012" and image_set == "test-data" else key
        if not key == "2012-test-data":
            dataset_year_dict = DATASET_YEAR_DICT[key]
    
            self.url = dataset_year_dict["url"]
            self.filename = dataset_year_dict["filename"]
            self.md5 = dataset_year_dict["md5"]
    
            base_dir = dataset_year_dict["base_dir"]
            voc_root = os.path.join(self.root, base_dir)
    
            if download:
                download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
    
            if not os.path.isdir(voc_root):
                raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
    
            splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
            split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
    
            image_dir = os.path.join(voc_root, "JPEGImages")
            self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
    
            target_dir = os.path.join(voc_root, self._TARGET_DIR)
            self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]
    
            assert len(self.images) == len(self.targets)
        
        else: # "2012-test-data"
            voc_root = os.path.join(self.root, os.path.join('VOCdevkit', 'VOC2012'))
            if not os.path.isdir(voc_root):
                raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
            splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
            split_f = os.path.join(splits_dir, "test.txt")
            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
            
            image_dir = os.path.join(voc_root, "JPEGImages")
            self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
            
            self.targets = []
            
        self.PASCAL_VOC_LABELS = ['__background__',  # always index 0
                                  'aeroplane', 'bicycle', 'bird', 'boat',
                                  'bottle', 'bus', 'car', 'cat', 'chair',
                                  'cow', 'diningtable', 'dog', 'horse',
                                  'motorbike', 'person', 'pottedplant',
                                  'sheep', 'sofa', 'train', 'tvmonitor']
        self._class_to_ind = dict(zip(self.PASCAL_VOC_LABELS, range(len(self.PASCAL_VOC_LABELS))))
            

    def __len__(self) -> int:
        return len(self.images)


class VOCDetection(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    @property
    def annotations(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        
        # VOC12
        if len(self.annotations) == 0: 
            filename    = self.images[index]
            w, h = img.size
            if self.transforms is not None:
                img, _ = self.transforms(img, None)
            output_targets = []
            output_targets.append(filename)
            output_targets.append((w,h))
            return img, output_targets
        
        # VOC07
        else: 
            target      = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
            filename    = target['annotation']['filename'] # might be useful!
            objects     = target['annotation']['object']
            
            labels = []
            difficulties = []
            bbox = torch.zeros((len(objects), 4)) # store in (x,y,w,h) format
            for obj_id, obj in enumerate(objects):
                label               = obj['name']
                labels.append(self._class_to_ind[label])
                diff                = int(obj['difficult'])
                difficulties.append(diff)
                box                 = obj['bndbox']
                bbox[obj_id, 0]     = float(box['xmin']) - 1 # VOC format starts with 1
                bbox[obj_id, 1]     = float(box['ymin']) - 1
                bbox[obj_id, 2]     = float(box['xmax']) - float(box['xmin']) - 1
                bbox[obj_id, 3]     = float(box['ymax']) - float(box['ymin']) - 1
            
            w, h = img.size
            if self.transforms is not None:
                img, bbox = self.transforms(img, bbox)
            
            output_targets = []
            output_targets.append(labels)
            output_targets.append(bbox)
            output_targets.append(difficulties)
            return img, output_targets

    def parse_voc_xml(self, node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
