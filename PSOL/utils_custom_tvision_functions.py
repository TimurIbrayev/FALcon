#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:53:23 2022

@author: tibrayev
"""

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


#%% ===========================================================================
#   Visualization functions
# =============================================================================
def plot_curve(x, y, title, xlabel, ylabel, fname):
    plt.figure()
    plt.plot(x, y, 'b')
    #plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    #plt.show()

def imshow(input, normalize=True):
    input_to_show = input.cpu().clone().detach()
    if normalize:
        input_to_show = (input_to_show - input_to_show.min())/(input_to_show.max() - input_to_show.min())
    plt.figure()
    if input_to_show.ndim == 4 and input_to_show.size(1) == 3:
        plt.imshow(input_to_show[0].permute(1,2,0))
    elif input_to_show.ndim == 4 and input_to_show.size(1) == 1:
        plt.imshow(input_to_show[0,0])
    elif input_to_show.ndim == 3 and input_to_show.size(0) == 3:
        plt.imshow(input_to_show.permute(1,2,0))
    elif input_to_show.ndim == 3 and input_to_show.size(0) == 1:
        plt.imshow(input_to_show[0])
    elif input_to_show.ndim == 2:
        plt.imshow(input_to_show)
    else:
        raise ValueError("Input with {} dimensions is not supported by this function!".format(input_to_show.ndim))

def plotregions(list_of_regions, glimpse_size = None, color='g', **kwargs):
    if glimpse_size is None:
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            width = region[2].item()
            height = region[3].item()
            # Add the patch to the Axes
            # FYI: Rectangle doc says the first argument defines bottom left corner. However, in reality it changes based on plt axis. 
            # So, if the origin of plt (0,0) is at top left, then (x,y) specify top left corner. 
            # Essentially, (x,y) needs to point to x min and y min of bbox.
            plt.gca().add_patch(Rectangle((xmin,ymin), width, height, linewidth=2, edgecolor=color, facecolor='none', **kwargs))
    elif glimpse_size is not None:
        if isinstance(glimpse_size, tuple):
            width, height = glimpse_size
        else:
            width = height = glimpse_size
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            plt.gca().add_patch(Rectangle((xmin,ymin), width, height, linewidth=2, edgecolor=color, facecolor='none', **kwargs))

def plotspots(list_of_spots, color='g', **kwargs):
    for spot in list_of_spots:
        x = spot[0].item()
        y = spot[1].item()
        # Add the circle to the Axes
        plt.gca().add_patch(Circle((x,y), radius=2, edgecolor=color, facecolor=color, **kwargs))

def plotspots_at_regioncenters(list_of_regions, glimpse_size = None, color='g', **kwargs):
    if glimpse_size is None:
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            width = region[2].item()
            height = region[3].item()
            x_center = xmin + (width / 2.0)
            y_center = ymin + (height / 2.0)
            plt.gca().add_patch(Circle((x_center, y_center), radius=2, edgecolor=color, facecolor=color, **kwargs))
    elif glimpse_size is not None:
        if isinstance(glimpse_size, tuple):
            width, height = glimpse_size
        else:
            width = height = glimpse_size
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            x_center = xmin + (width / 2.0)
            y_center = ymin + (height / 2.0)
            plt.gca().add_patch(Circle((x_center, y_center), radius=2, edgecolor=color, facecolor=color, **kwargs))



def extract_simscore_graphs(simscores, save_dir, epoch, sample_id=0):
    sample_simscores = []
    for simscore in simscores:
        sample_simscores.append(simscore[sample_id].item())
    plot_curve(range(1, len(simscores)+1), sample_simscores, 'similarity scores for sample {}'.format(sample_id), 
               'glimpse iteration', 'similarity score (cosine similarity)', save_dir+'sample_{}_epoch_{}_simscores.png'.format(sample_id, epoch))

def extract_glimpses_per_image(images, bbox_targets, glimpses, save_dir, epoch, sample_id=0):
    imshow(images[sample_id])
    plotregions(bbox_targets[sample_id].unsqueeze(0), color='r')
    plotregions(glimpses[0][sample_id].unsqueeze(0))
    plotregions(glimpses[1][sample_id].unsqueeze(0), color='darkorange')
    plotregions(glimpses[2][sample_id].unsqueeze(0), color='k')
    plotregions(glimpses[3][sample_id].unsqueeze(0), color='y')
    plotregions(glimpses[4][sample_id].unsqueeze(0), color='m')
    plotregions(glimpses[5][sample_id].unsqueeze(0), color='b')
    plotregions(glimpses[6][sample_id].unsqueeze(0), color='w')
    plotregions(glimpses[7][sample_id].unsqueeze(0), color='c')
    plotregions(glimpses[-1][sample_id].unsqueeze(0), color='g')
    f = plt.gcf()
    f.savefig(save_dir+'sample_{}_epoch_{}_glimpses.png'.format(sample_id, epoch))
    
def extract_reward_bars(rewards, save_dir, epoch, sample_id=0):
    sample_rewards = []
    for reward in rewards:
        sample_rewards.append(reward[sample_id].item())
    rewards_color = [{r<0: 'red', r>=0: 'green'}[True] for r in sample_rewards]
    plt.figure()
    plt.bar(range(1, len(rewards) + 1), sample_rewards, width=0.25, color=rewards_color)
    plt.title('rewards for sample {}'.format(sample_id))
    plt.xlabel('glimpse iteration')
    plt.ylabel('reward value')
    plt.savefig(save_dir+'sample_{}_epoch_{}_rewards.png'.format(sample_id, epoch))

def extract_info_per_sample(images, bbox_targets, glimpses, simscores, rewards, save_dir, epoch, sample_id=0):
    extract_glimpses_per_image(images, bbox_targets, glimpses, save_dir, epoch, sample_id)
    extract_simscore_graphs(simscores, save_dir, epoch, sample_id)
    extract_reward_bars(rewards, save_dir, epoch, sample_id)



#%% ===========================================================================
#   IoU functions
# =============================================================================
# taken from torchvision.ops.boxes.py
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# taken from torchvision.ops.boxes.py
# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2] right-bottom

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def region_iou(region1, region2):
    """
    Return intersection-over-union (Jaccard index) of regions.
    
    Here, we define region as a structure in (x1, y1, width, height) format 
    and boxes as a structure in (x1, y1, x2, y2) format.

    Hence, both sets of regions are expected to be in (x1, y1, width, height) format.

    Arguments:
        region1 (Tensor[N, 4])
        region2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in region1 and region2
    """
    boxes1 = region1.clone().detach()
    boxes1[:, 2] += boxes1[:, 0] # x2 = x1 + width
    boxes1[:, 3] += boxes1[:, 1] # y2 = y1 + height
    boxes2 = region2.clone().detach()
    boxes2[:, 2] += boxes2[:, 0]
    boxes2[:, 3] += boxes2[:, 1]
    return box_iou(boxes1, boxes2)

def region_area(regions):
    """
    Computes the area of a set of regions.
    
    Here, we define region as a structure in (x1, y1, width, height) format.

    Arguments:
        regions (Tensor[N, 4]): regions for which the area will be computed. They
            are expected to be in (x1, y1, width, height) format

    Returns:
        area (Tensor[N]): area for each region
    """
    return regions[:, 2] * regions[:, 3]


#%% ===========================================================================
#       Custom Contrastive Loss function
# =============================================================================
def custom_CL(vec_1, vec_2):
    assert vec_1.shape[0] == vec_2.shape[0], "vec_1 and vec_2 should be equal length"
    epsilon = 1e-12
    
    N = vec_1.shape[0]
    vec_1_normalized = vec_1/(vec_1.norm(p=2, dim=-1, keepdim=True) + epsilon)
    vec_2_normalized = vec_2/(vec_2.norm(p=2, dim=-1, keepdim=True) + epsilon)
    
    cosine_similarities = []
    
    for i in range(N):
        similarity_positive_pair    = (vec_1_normalized[i]*vec_2_normalized[i]).sum()/0.5
        numerator                   = similarity_positive_pair.exp()
        similarity_with_all_targets = torch.matmul(vec_2_normalized, vec_1_normalized[i].unsqueeze(-1))/0.5
        mask                        = torch.logical_not(torch.all((vec_2_normalized == vec_2_normalized[i]), dim=-1)) # exclude duplicate targets
        denominator                 = similarity_with_all_targets[mask].exp().sum() + numerator
        if i == 0:
            loss                    = -torch.log(numerator/denominator)
        else:
            loss                   += -torch.log(numerator/denominator)
        cosine_similarities.append(similarity_positive_pair.clone().detach())
    
    loss /= (1.0*N)
    return loss, torch.stack(cosine_similarities)
    
    
def contrastive_loss_compact(vec_1, vec_2, temperature=1.0):
 
    epsilon = 0.0#1e-12
    
    N = vec_1.shape[0]
    vec_1_normalized = vec_1/(vec_1.norm(p=2, dim=-1, keepdim=True) + epsilon)
    vec_2_normalized = vec_2/(vec_2.norm(p=2, dim=-1, keepdim=True) + epsilon)
    long_vector      = torch.cat((vec_1_normalized, vec_2_normalized), dim=0)
    
    
    pos_sim         = torch.exp(torch.sum(vec_1_normalized * vec_2_normalized, dim=-1) / temperature)
    numerator       = torch.cat((pos_sim, pos_sim), dim=0)
    positive_similarities = torch.log(pos_sim).detach() * temperature
    
    sim_matrix      = torch.exp(torch.mm(long_vector, long_vector.t().contiguous()) / temperature)
    mask            = (torch.ones_like(sim_matrix) - torch.eye(2 * N, device=sim_matrix.device)).bool()
    sim_matrix      = sim_matrix.masked_select(mask).view(2 * N, -1)
    denominator     = sim_matrix.sum(dim=-1) + epsilon
    
    loss            = (-torch.log(numerator/denominator)).mean()
    return loss, positive_similarities












