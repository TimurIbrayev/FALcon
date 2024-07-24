#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:54:34 2022
Verified on May 25 2022

@author: tibrayev
"""
import torch
import torchvision.transforms.functional as F_vision
import math

def location_bounds(glimpse_w, input_w):
    """Given input image width and glimpse width returns (lower,upper) bound in (-1,1) for glimpse centers.
    :param: int  glimpse_w      width of glimpse patch
    :param: int  input_w        width of input image
    :return: int lower          lower bound in (-1,1) for glimpse center locations
    :return: int upper
    """
    offset = float(glimpse_w) / input_w
    lower = (-1 + offset)
    upper = (1 - offset)
    assert lower >= -1 and lower <= 1, 'lower must be in (-1,1), is {}'.format(lower)
    assert upper >= -1 and upper <= 1, 'upper must be in (-1,1), is {}'.format(upper)
    return lower, upper


#%%
def extract_and_resize_glimpses_for_batch(images, glimpses_locs_dims, resized_height, resized_width):
    """
    Given the batch of images and the batch of glimpse locations and their sizes, 
    this function extracts these glimpses from images and resizes them to the same size.

    Parameters
    ----------
    images : Tensor[batch_size, channels, height, width]
        The batch of tensor images, from which glimpses are to be extracted.
    glimpses_locs_dims : Tensor[batch_size, 4]
        The batch of glimpse locations and their sizes, where second dimension is 4-sized tuple,
        representing (x_TopLeftCorner, y_TopLeftCorner, width, height) of each glimpse in the batch.
    resized_height: Int
        The height of glimpses in the output batch of glimpses
    resized_width: Int
        The width of glimpses in the output batch of glimpses

    Returns
    -------
    batch_extracted_and_resized_glimpses : Tensor[batch_size, channels, resized_height, resized_width]
        The output batch of extracted and resized glimpses, extracted from the batch of images.
        
    Note: It is user's responsibility to make sure that the glimpse dimensions do not exceed image dimensions.
    """
    batch_extracted_and_resized_glimpses = []
    
    left_coords = glimpses_locs_dims[:, 0]
    top_coords  = glimpses_locs_dims[:, 1]
    widths      = glimpses_locs_dims[:, 2]
    heights     = glimpses_locs_dims[:, 3]
    h_fixed     = resized_height
    w_fixed     = resized_width
    
    for image, left, top, width, height in zip(images, left_coords, top_coords, widths, heights):
        resized_glimpse = F_vision.resized_crop(image, top, left, height, width, (h_fixed, w_fixed))
        batch_extracted_and_resized_glimpses.append(resized_glimpse)
    batch_extracted_and_resized_glimpses = torch.stack(batch_extracted_and_resized_glimpses, dim=0)
    return batch_extracted_and_resized_glimpses


def cut_and_mask_glimpses_for_batch(images, glimpses_locs_dims):
    to_round = False if 'int' in str(glimpses_locs_dims.dtype) else True
    if to_round:
        left_coords     = torch.round(glimpses_locs_dims[:, 0]).int()
        top_coords      = torch.round(glimpses_locs_dims[:, 1]).int()
        widths          = torch.round(glimpses_locs_dims[:, 2]).int()
        heights         = torch.round(glimpses_locs_dims[:, 3]).int()
    else:
        left_coords     = glimpses_locs_dims[:, 0]
        top_coords      = glimpses_locs_dims[:, 1]
        widths          = glimpses_locs_dims[:, 2]
        heights         = glimpses_locs_dims[:, 3]
    right_coords    = left_coords + widths
    bottom_coords   = top_coords + heights
    
    batch_cut_and_masked_glimpses = torch.zeros_like(images)
    for img_id, (left, top, right, bottom) in enumerate(zip(left_coords, top_coords, right_coords, bottom_coords)):
        batch_cut_and_masked_glimpses[img_id, :, top:bottom, left:right] = images[img_id, :, top:bottom, left:right]
    return batch_cut_and_masked_glimpses


#%%
def get_grid(full_res_img_size, grid_size, grid_center_coords = False):
    """
    Divide entire image into grid of cells of fixed size
    """
    assert isinstance(full_res_img_size, (int, tuple)), "input argument should be either int or tuple"
    assert isinstance(grid_size, (int, tuple)), "input argument should be either int or tuple"
    full_res_img_size   = full_res_img_size if isinstance(full_res_img_size, tuple) else (full_res_img_size, full_res_img_size)
    grid_size           = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)

    all_grid_locations = []
    grid_dim_x = math.ceil(full_res_img_size[0]/grid_size[0])
    grid_dim_y = math.ceil(full_res_img_size[1]/grid_size[1])
    if grid_center_coords:
        offset_x = grid_size[0]/2.0 - 0.5
        offset_y = grid_size[1]/2.0 - 0.5
    else:
        offset_x = 0.0
        offset_y = 0.0
    for j in range(grid_dim_y):
        for i in range(grid_dim_x):
            grid_cell_x = grid_size[0]*i + offset_x
            grid_cell_y = grid_size[1]*j + offset_y
            all_grid_locations.append(torch.tensor([grid_cell_x, grid_cell_y]))
    all_grid_locations = torch.stack(all_grid_locations)
    return all_grid_locations # shape: [(grid_dim_x*grid_dim_y), 2] with (x, y) coordinates

def filter_glimpse_locations(all_grid_cells_centers, bboxes):
    x_min = bboxes[:, 0].unsqueeze(-1)
    x_max = x_min + bboxes[:, 2].unsqueeze(-1)
    y_min = bboxes[:, 1].unsqueeze(-1)
    y_max = y_min + bboxes[:, 3].unsqueeze(-1)
    glimpse_centers = all_grid_cells_centers.unsqueeze(0).expand(bboxes.shape[0], -1, -1)
    x = torch.logical_and((glimpse_centers[:, :, 0] >= x_min), (glimpse_centers[:, :, 0] <= x_max))
    y = torch.logical_and((glimpse_centers[:, :, 1] >= y_min), (glimpse_centers[:, :, 1] <= y_max))
    # binary map for glimpses where bounding box is located. torch.logical_and(x, y).view(self.config.grid_dim_x, self.config.grid_dim_y)
    # NOTE: see also note about order of loops in all_glimpse_locations.
    return torch.logical_and(x, y)

def guess_TF_init_glimpses_for_batch(all_grid_cells_centers, bbox_coords, is_inside_bbox=True):
    "NOTE: all_grid_cells_centers should point to centers of grid cells"
    "NOTE: bbox_coords should be 4D tuple, with (x_TopLeft, y_TopLeft, width, height) coordinates"
    filtered_interior_glimpses = filter_glimpse_locations(all_grid_cells_centers, bbox_coords)
    
    batch_glimpses_locs = []
    for sample_id in range(bbox_coords.shape[0]):
        glimpses_within_the_bbox = all_grid_cells_centers[(filtered_interior_glimpses[sample_id] == is_inside_bbox)]
        
        if glimpses_within_the_bbox.shape[0] == 0: # FIXME: in case, none of the glimpses is within bounding box, just select random any possible glimpses.
            selected_glimpse_idx = torch.randint(0, all_grid_cells_centers.shape[0], (1,))
            selected_glimpse_loc = all_grid_cells_centers[selected_glimpse_idx]
        else:
            selected_glimpse_idx = torch.randint(0, glimpses_within_the_bbox.shape[0], (1,)) # NOTE: we are only selecting initial guess glimpse, hence 1.
            selected_glimpse_loc = glimpses_within_the_bbox[selected_glimpse_idx]
        batch_glimpses_locs.append(selected_glimpse_loc)
    batch_glimpses_locs = torch.cat(batch_glimpses_locs, dim=0) # shape: [batch_size, 2] with (x,y) coordinates of the center for each of initial glimpses
    return batch_glimpses_locs # shape: [batch_size, 2] with (x_Center, y_Center) coordinates of the center for initial guess glimpses


def get_new_random_glimpses_for_batch(all_grid_cells_centers, unexplored_glimpse_locations, switch_location):
    # Function to randomly pick one of unexplored_glimpse_locations based on whether switch_location=True for that sample in the batch
    # Returns tensor of shape[num_of_required_switches, 2], i.e. the size of return tensor depends on how many switch_location=True
    # Always make sure outside the function to check that switch_location is not empty, i.e. check that switch_location.sum() != 0
    assert switch_location.sum() > 0, "None of the samples in the batch requires switching current glimpse location!"
    new_glimpse_locs = []
    for sample_id in range(switch_location.shape[0]):
        if switch_location[sample_id]:
            # unexplored_glimpse_locations[sample_id].nonzero() - list of indices of only unexplored glimpses for particular sample_id
            unexplored_glimpse_idx      = torch.randint(0, unexplored_glimpse_locations[sample_id].nonzero().shape[0], (1,))
            switch_glimpse_idx          = unexplored_glimpse_locations[sample_id].nonzero()[unexplored_glimpse_idx].item()
            unexplored_glimpse_locations[sample_id][switch_glimpse_idx] = 0
            switch_glimpse_loc          = all_grid_cells_centers[switch_glimpse_idx]
            new_glimpse_locs.append(switch_glimpse_loc.clone().detach())
    new_glimpse_locs = torch.stack(new_glimpse_locs, dim=0) # shape: [num_of_required_switches, 2] with (x,y) coordinates of the center for each new random glimpse grid cell
    return new_glimpse_locs # shape: [(switch_location=True).sum(), 2] with (x_Center, y_Center) coordinates of the center of the grid cell for new random glimpses




