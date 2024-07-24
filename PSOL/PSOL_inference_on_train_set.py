import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image
from utils.func import *
from utils.vis import *
from utils.IoU import *
from models.models import choose_locmodel,choose_clsmodel
from utils.augment import *
import argparse

parser = argparse.ArgumentParser(description='Parameters for PSOL evaluation')
parser.add_argument('--loc-model', metavar='locarg', type=str, default='vgg16',dest='locmodel')
parser.add_argument('--cls-model', metavar='clsarg', type=str, default='vgg16',dest='clsmodel')
parser.add_argument('--input_size',default=256,dest='input_size')
parser.add_argument('--crop_size',default=224,dest='crop_size')
parser.add_argument('--ten-crop', help='tencrop', action='store_true',dest='tencrop')
parser.add_argument('--gpu',help='which gpu to use',default='0',dest='gpu')
parser.add_argument('data',metavar='DIR',help='path to imagenet dataset')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
cudnn.benchmark = True
TEN_CROP = args.tencrop
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
])
cls_transform = transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
])
ten_crop_aug = transforms.Compose([
    transforms.Resize((args.input_size,args.input_size)),
    transforms.TenCrop(args.crop_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
])

# localizer
locname = args.locmodel
model = choose_locmodel(locname, True)
print(model)
model = model.to(0)
model.eval()

# classifier
clsname = args.clsmodel
cls_model = choose_clsmodel(clsname)
cls_model = cls_model.to(0)
cls_model.eval()

# dataset directories
root = args.data
val_imagedir = os.path.join(root, 'train')


classes = os.listdir(val_imagedir)
classes.sort()
temp_softmax = nn.Softmax()
#print(classes[0])
class_to_idx = {classes[i]:i for i in range(len(classes))}



result = {}

accs = []
accs_top5 = []
loc_accs = []
cls_accs = []
final_cls = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []
final_ind = []
save_pseudo_bboxes_path = './results/ImageNet_train_set/predicted_bounding_boxes'
if not os.path.exists(save_pseudo_bboxes_path):
    os.makedirs(save_pseudo_bboxes_path)
for k in range(1000):
    cls = classes[k]

    total = 0
    IoUSet = []
    IoUSetTop5 = []
    LocSet = []
    ClsSet = []

    files = os.listdir(os.path.join(val_imagedir, cls))
    files.sort()
    
    pseudo_bboxes = {}

    for (i, name) in enumerate(files):
        # raw_img = cv2.imread(os.path.join(imagedir, cls, name))
        now_index = int(name.split('_')[-1].split('.')[0])
        final_ind.append(now_index-1)

        raw_img = Image.open(os.path.join(val_imagedir, cls, name)).convert('RGB')
        w, h = raw_img.size
        
        with torch.no_grad():
            img = transform(raw_img)
            img = torch.unsqueeze(img, 0)
            img = img.to(0)
            reg_outputs = model(img) # localizer

            bbox = to_data(reg_outputs)
            bbox = torch.squeeze(bbox)  # The outputs are in (x, y, w, h) format
            #bbox = bbox.numpy()         # This is the normalized (between 0 and 1) predicted bounding box: can think as the values are for input of size (w, h) = (1, 1)
            
            pseudo_bboxes[os.path.join(val_imagedir, cls, name)] = [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]
    with open(os.path.join(save_pseudo_bboxes_path, '%s_bbox.json' % cls), 'w') as fp:
        json.dump(pseudo_bboxes, fp)
        
    if (k+1) %100==0:
        print(k)
