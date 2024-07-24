import os
import sys
import cv2
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from skimage import measure
# from scipy.misc import imresize
from utils.func import *
from utils.vis import *
from utils.IoU import *
import argparse
from loader.cub_loader import custom_Compose, CUBirds_2011
import csv

parser = argparse.ArgumentParser(description='Parameters for DDT generate box')
parser.add_argument('--input_size',default=448,dest='input_size')
parser.add_argument('data',metavar='DIR',help='path to imagenet dataset')
parser.add_argument('--gpu',help='which gpu to use',default='0',dest='gpu')
parser.add_argument('--output_path',default='results/DDT/CUB_test/Projection/VGG16-448',dest='output_path')
parser.add_argument('--batch_size',default=256,type=int,dest='batch_size')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"
cudnn.benchmark = True
model_ft = models.vgg16(pretrained=True)
model = model_ft.features
#removed = list(model.children())[:-1]
#model = torch.nn.Sequential(*removed)
model = torch.nn.DataParallel(model).cuda()
model.eval()
projdir = args.output_path
if not os.path.exists(projdir):
    os.makedirs(projdir)

transform = custom_Compose([
    transforms.Resize((args.input_size,args.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
batch_size = args.batch_size
cub_data = CUBirds_2011(root=args.data, 
                        split='test', 
                        target_type=['class', 'filename', 'original_wh'],
                        transform=transform)
loader  = torch.utils.data.DataLoader(cub_data, batch_size=batch_size, shuffle=False)


ddt_bbox = {}
for class_ind in range(200):
    now_class_dict = {}
    now_class_img_sizes = {}
    feature_list = []

    with torch.no_grad():
        # collect features for all samples per class
        for input_img, (label, filename, original_wh) in loader:
            mask = label == class_ind
            if mask.float().sum() == 0:
                continue
            else:
                path   = [fname for fname, m in zip(filename, mask) if m]
                orig_img_sizes = [img_size for img_size, m in zip(original_wh, mask) if m]
                # input_img = to_variable(input_img)
                output = model(input_img)
                output = output[mask]
                output = to_data(output)
                output = torch.squeeze(output).numpy()
                if len(output.shape) == 3:
                    output = np.expand_dims(output,0)
                output = np.transpose(output,(0,2,3,1))
                n,h,w,c = output.shape
                for i in range(n):
                    now_class_dict[path[i]] = output[i,:,:,:]
                    now_class_img_sizes[path[i]] = orig_img_sizes[i]
                output = np.reshape(output,(n*h*w,c))
                feature_list.append(output)
        print("Class: {}, test cnt files: {}".format(class_ind, len(now_class_dict)))
        X = np.concatenate(feature_list,axis=0)
        mean_matrix = np.mean(X, 0)
        X = X - mean_matrix
        print("Before PCA")
        trans_matrix = sk_pca(X, 1)
        print("AFTER PCA")
        
        cls = class_ind
        # save json
        d = {'mean_matrix': mean_matrix.tolist(), 'trans_matrix': trans_matrix.tolist()}
        with open(os.path.join(projdir, '%s_trans.json' % cls), 'w') as f:
            json.dump(d, f)
        # load json
        with open(os.path.join(projdir, '%s_trans.json' % cls), 'r') as f:
            t = json.load(f)
            mean_matrix = np.array(t['mean_matrix'])
            trans_matrix = np.array(t['trans_matrix'])

        print('trans_matrix shape is {}'.format(trans_matrix.shape))
        cnt = 0
        for k, v in now_class_dict.items():
            w = 14
            h = 14
            he = 448
            wi = 448
            v = np.reshape(v,(h * w,512))
            v = v - mean_matrix

            heatmap = np.dot(v, trans_matrix.T)
            heatmap = np.reshape(heatmap, (h, w))
            highlight = np.zeros(heatmap.shape)
            highlight[heatmap > 0] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1

            # visualize heatmap
            # show highlight in origin image
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(highlight, (he, wi), interpolation=cv2.INTER_NEAREST)
            props = measure.regionprops(highlight_big.astype(int))

            if len(props) == 0:
                #print(highlight)
                bbox = [0, 0, wi, he]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]] # x1, y1, x2, y2

            temp_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] # x, y, w, h
            temp_save_box = [x / 448 for x in temp_bbox]    # Coordinates normalized to the image dimensions (448x448 in this case) to range (0,1)
            
            # This is what is eventually saved: (x,y,w,h) of the DDT predicted bounding box restored to original image dimensions (before transforms.Resize)
            w, h = now_class_img_sizes[k]
            temp_save_box[0] *= w
            temp_save_box[1] *= h
            temp_save_box[2] *= w
            temp_save_box[3] *= h
            ddt_bbox[k] = temp_save_box

            highlight_big = np.expand_dims(np.asarray(highlight_big), 2)
            highlight_3 = np.concatenate((np.zeros((he, wi, 1)), np.zeros((he, wi, 1))), axis=2)
            highlight_3 = np.concatenate((highlight_3, highlight_big), axis=2)
            cnt +=1
            if cnt < 10:
                savepath = 'results/DDT/CUB_test/Visualization/VGG16-vis-test/%s' % cls
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                from PIL import Image
                raw_img = Image.open(os.path.join(cub_data.root, cub_data.base_folder, "images", k)).convert("RGB")
                raw_img = raw_img.resize((448,448))
                raw_img = np.asarray(raw_img)
                raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)
                cv2.rectangle(raw_img, (temp_bbox[0], temp_bbox[1]),
                              (temp_bbox[2] + temp_bbox[0], temp_bbox[3] + temp_bbox[1]), (255, 0, 0), 4)
                save_name = k.split('/')[-1]
                cv2.imwrite(os.path.join(savepath, save_name), np.asarray(raw_img))

with open(os.path.join(projdir, 'ddt_bounding_boxes.txt'), 'w') as fp:
    for k, v in ddt_bbox.items():
        fp.write(k 
                 + ' ' + '{:.2f}'.format(v[0])
                 + ' ' + '{:.2f}'.format(v[1])
                 + ' ' + '{:.2f}'.format(v[2])
                 + ' ' + '{:.2f}\n'.format(v[3]))