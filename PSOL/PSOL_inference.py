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
val_imagedir = os.path.join(root, 'val')
# anno_root = os.path.join(root,'bbox')
val_annodir = os.path.join(root, 'anno_val')


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
for k in range(1000):
    cls = classes[k]

    total = 0
    IoUSet = []
    IoUSetTop5 = []
    LocSet = []
    ClsSet = []

    files = os.listdir(os.path.join(val_imagedir, cls))
    files.sort()

    for (i, name) in enumerate(files):
        # raw_img = cv2.imread(os.path.join(imagedir, cls, name))
        now_index = int(name.split('_')[-1].split('.')[0])
        final_ind.append(now_index-1)
        xmlfile = os.path.join(val_annodir, cls, name.split('.')[0] + '.xml')
        gt_boxes = get_cls_gt_boxes(xmlfile, cls)
        if len(gt_boxes)==0:
            continue

        raw_img = Image.open(os.path.join(val_imagedir, cls, name)).convert('RGB')
        w, h = raw_img.size
        
        with torch.no_grad():
            img = transform(raw_img)
            img = torch.unsqueeze(img, 0)
            img = img.to(0)
            reg_outputs = model(img) # localizer

            bbox = to_data(reg_outputs)
            bbox = torch.squeeze(bbox)  # The outputs are in (x, y, w, h) format
            bbox = bbox.numpy()         # This is the normalized (between 0 and 1) predicted bounding box: can think as the values are for input of size (w, h) = (1, 1)
            if TEN_CROP: # classifier
                img = ten_crop_aug(raw_img)
                img = img.to(0)
                vgg16_out = cls_model(img)
                vgg16_out = temp_softmax(vgg16_out)
                vgg16_out = torch.mean(vgg16_out,dim=0,keepdim=True)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            else:
                img = cls_transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                vgg16_out = cls_model(img)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            vgg16_out = to_data(vgg16_out)
            vgg16_out = torch.squeeze(vgg16_out)
            vgg16_out = vgg16_out.numpy()
            out = vgg16_out
        ClsSet.append(out[0]==class_to_idx[cls]) # top-1 classification results

        #handle resize and centercrop for gt_boxes
        for j in range(len(gt_boxes)):
            temp_list = list(gt_boxes[j])
            raw_img_i, gt_bbox_i = ResizedBBoxCrop((256,256))(raw_img, temp_list)
            raw_img_i, gt_bbox_i = CenterBBoxCrop((224))(raw_img_i, gt_bbox_i) # gt_bbox_i are normalized (between 0 and 1) bounding box dimensions in (x1y1x2y2) format
            w, h = raw_img_i.size

            gt_bbox_i[0] = gt_bbox_i[0] * w
            gt_bbox_i[2] = gt_bbox_i[2] * w
            gt_bbox_i[1] = gt_bbox_i[1] * h
            gt_bbox_i[3] = gt_bbox_i[3] * h

            gt_boxes[j] = gt_bbox_i

        w, h = raw_img_i.size

        bbox[0] = bbox[0] * w
        bbox[2] = bbox[2] * w + bbox[0]
        bbox[1] = bbox[1] * h
        bbox[3] = bbox[3] * h + bbox[1] # This is the predicted bounding box for input of size (w, h) = (224, 224)

        max_iou = -1
        for gt_bbox in gt_boxes:
            iou = IoU(bbox, gt_bbox)
            if iou > max_iou:
                max_iou = iou

        LocSet.append(max_iou)
        temp_loc_iou = max_iou
        if out[0] != class_to_idx[cls]:
            max_iou = 0

        # print(max_iou)
        result[os.path.join(cls, name)] = max_iou
        IoUSet.append(max_iou)
        #cal top5 IoU
        max_iou = 0
        for i in range(5):
            if out[i] == class_to_idx[cls]:
                max_iou = temp_loc_iou
        IoUSetTop5.append(max_iou)
        #visualization code
        '''
        opencv_image = deepcopy(np.array(raw_img_i))
        opencv_image = opencv_image[:, :, ::-1].copy()
        for gt_bbox in gt_boxes:
            cv2.rectangle(opencv_image, (int(gt_bbox[0]), int(gt_bbox[1])),
                          (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 4)
        cv2.rectangle(opencv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 255), 4)
        cv2.imwrite(os.path.join(savepath, str(name) + '.jpg'), np.asarray(opencv_image))
        '''
    cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    final_clsloc.extend(IoUSet)
    cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
    final_clsloctop5.extend(IoUSetTop5)
    loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
    final_loc.extend(LocSet)
    cls_acc = np.sum(np.array(ClsSet))/len(ClsSet)
    final_cls.extend(ClsSet)
    print('{} cls-loc acc is {}, loc acc is {}, vgg16 cls acc is {}'.format(cls, cls_loc_acc, loc_acc, cls_acc))
    with open('inference_CorLoc.txt', 'a+') as corloc_f:
        corloc_f.write('{} {}\n'.format(cls, loc_acc))
    accs.append(cls_loc_acc)
    accs_top5.append(cls_loc_acc_top5)
    loc_accs.append(loc_acc)
    cls_accs.append(cls_acc)
    if (k+1) %100==0:
        print(k)


print(accs)
print('Cls-Loc acc {}'.format(np.mean(accs)))
print('Cls-Loc acc Top 5 {}'.format(np.mean(accs_top5)))

print('GT Loc acc {}'.format(np.mean(loc_accs)))
print('{} cls acc {}'.format(clsname, np.mean(cls_accs)))
with open('Corloc_result.txt', 'w') as f:
    for k in sorted(result.keys()):
        f.write('{} {}\n'.format(k, str(result[k])))
