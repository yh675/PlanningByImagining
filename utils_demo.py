import gflags
import os
import shutil
import sys

import numpy as np
import yaml

import load_params_demo

argv = gflags.FLAGS(sys.argv)


class Resize_params:
    def __init__(self, scale_percent, img):

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        self.dim = dim


def empty_folder(path):
    make_path = os.path.exists(path)
    if make_path:
        shutil.rmtree(path)
    os.makedirs(path)


def process_for_astar(img):

    rows = img.shape[0]
    cols = img.shape[1]

    white = [255, 255, 255]
    black = [0, 0, 0]

    with open(gflags.FLAGS.labels_yaml) as file:
        yaml_doc = yaml.full_load(file)

    obstacles = yaml_doc['map_obstacles']['simple']
    color_map = yaml_doc['color_map']

    ob_colors = [color_map[ob] for ob in obstacles]
    ob_colors = np.asarray(ob_colors)

    for i in range(rows):
        for j in range(cols):
            x = ob_colors - img[i, j, :]
            check_arr = np.sum(np.abs(x) ** 2, axis=-1) ** (1. / 2)

            if np.amin(check_arr) < 5.: #if that pixel is an obstacle
                img[i, j, :] = black
            else: #if that pixel is not an obstacle
                img[i, j, :] = white

    return img


def get_img_coords(img):
    img_coords = []  # are indexed by [row, column]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_coords.append([i, j])

    return img_coords


def img_coords_mask(img):
    img_coords = []  # are indexed by [row, column]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, :].tolist() == [255, 255, 255]: #if that pixel is white
                img_coords.append([i, j])

    return img_coords


def check_astar_loc(start_loc, goal_loc, my_map):#confirm that the start and goal pixels are not on obstacles
    row_s = start_loc[1]
    col_s = start_loc[0]

    row_g = goal_loc[1]
    col_g = goal_loc[0]

    if my_map[row_s, col_s] != 1 and my_map[row_g, col_g] != 1: #if neither are on an obstacle
        return True
    else:
        return False


def check_replan(path, current_step, obs_map):
    path = path[current_step:-1]
    for p in path:
        if obs_map[p[1], p[0], :].tolist() != [255, 255, 255]:
            return True

    return False


