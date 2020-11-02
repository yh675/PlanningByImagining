# astar implementation with some inspiration from https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
import os
import gflags
import sys

import numpy as np
from numpy import linalg as LA
import math
from operator import attrgetter
import cv2

import load_params_demo
import utils_demo

argv = gflags.FLAGS(sys.argv)

#colors
blue = [255, 0, 0]
red = [0, 0, 255]
green = [0, 255, 0]
white = [255, 255, 255]
black = [0, 0, 0]


class Measure_params:
    def __init__(self, range, n_seg, centroid, slice_tol, ray_seg):
        self.range = range
        self.n_seg = n_seg
        self.centroid = centroid
        self.slice_tol = slice_tol
        self.ray_seg = ray_seg


class Measurement:
    def __init__(self):
        self.geometry = self.Geometry()
        self.coords = self.Coords()
        self.slice_tol = None
        self.ray = self.Ray()


    class Geometry:
        def __init__(self):
            self.range = None
            self.n_seg = None
            self.centroid = None
            self.slice_tol = None
            self.ray_seg = None

    class Coords:
        def __init__(self):
            self.point_coords = None
            self.slice_coords = None
            self.slice_keep_coords = None
            self.slice_obs_coords = None
            self.see_obs_list = None
            self.pts_obs_list = None

    class Ray:
        def __init__(self):
            self.edge_x = None
            self.edge_y = None
            self.ray_x = None
            self.ray_y = None

    def init(self, params, time_step, seq, img_coords, output_dir, resize=None):

        # img = cv2.imread(output_dir + '{}_s{}_output.png'.format(time_step, seq))
        img = cv2.imread(os.path.join(output_dir, '{}_s{}_output.png'.format(time_step, seq)))

        obs_map = utils_demo.process_for_astar(img)
        if resize is not None:
            obs_map = cv2.resize(obs_map, resize.dim, interpolation=cv2.INTER_AREA)

        obs_map = refine_map(obs_map)

        self.geometry.range = params.range
        self.geometry.n_seg = params.n_seg
        self.geometry.centroid = params.centroid
        self.geometry.ray_seg = params.ray_seg
        self.geometry.slice_tol = params.slice_tol

        #get all points from truth map that are within the circle
        img_coords = np.asarray(img_coords)
        X = img_coords[:, 1] #column
        Y = img_coords[:, 0] #row
        cx = self.geometry.centroid[1]
        cy = self.geometry.centroid[0]

        check = np.square((cx - X)) + np.square((cy - Y))
        check = np.where(check < self.geometry.range ** 2)[0]

        self.coords.point_coords = img_coords[check] #this is a numpy array

        X = self.coords.point_coords[:, 1]
        Y = self.coords.point_coords[:, 0]

        sliceno = np.int32((math.pi + np.arctan2(Y - cy, X - cx)) * (self.geometry.n_seg / (2 * math.pi)) - \
                           self.geometry.slice_tol)


        slice_coords = []

        for un in np.arange(self.geometry.n_seg):
            slice_coords.append(self.coords.point_coords[np.where(sliceno == un)[0]])

        self.coords.slice_coords = slice_coords

        #unit vectors for rays
        thetas = np.linspace(0, 2*math.pi, self.geometry.n_seg, endpoint=False)
        thetas = thetas + (thetas[1] - thetas[0]) / 2.
        thetas = np.flip(thetas) + math.pi

        ux = np.cos(thetas)
        uy = np.sin(thetas)

        if len(slice_coords) != thetas.shape[0]:
            print("centroid", self.geometry.centroid)

        assert len(slice_coords) == thetas.shape[0]

        # exit(1)

        self.ray.edge_x = np.int32(ux * self.geometry.range + cx)
        self.ray.edge_y = np.int32(uy * self.geometry.range + cy)

        #calculate rays
        ray_disc = np.linspace(0, self.geometry.range, self.geometry.ray_seg, endpoint=True)

        ray_disc = ray_disc.reshape((1, ray_disc.shape[0]))
        ux = ux.reshape((ux.shape[0], 1))
        uy = uy.reshape((uy.shape[0], 1))

        self.ray.ray_x = np.multiply(ux, ray_disc).astype(np.int32) + cx
        self.ray.ray_y = np.multiply(uy, ray_disc).astype(np.int32) + cy

        slice_keep_coords = []
        slice_obs_coords = []
        see_obs_list = [] #"viewed" pixels which are obstacles

        pts_obs_list = [] #list of points which are detected to hit an obstacle first
        slice_coords = slice_coords[::-1]
        for row_x, row_y, slice in zip(self.ray.ray_x, self.ray.ray_y, slice_coords):

            if len(slice) == 0:
                continue

            x_keep = np.where((0 < row_x) & (row_x < obs_map.shape[1]))[0]  # which indexes to keep from that row for x
            row_x = row_x[x_keep]
            row_y = row_y[x_keep]
            y_keep = np.where((0 < row_y) & (row_y < obs_map.shape[0]))[0]
            row_x = row_x[y_keep]
            row_y = row_y[y_keep]

            obs_pixels = np.where((obs_map[row_y, row_x, :] == (0, 0, 0)).all(axis=1))[0]

            if obs_pixels.shape[0] == 0:
                slice_keep_coords.append(slice)
                continue
            else:
                obs_dist = LA.norm((np.array([cy - row_y[obs_pixels[0]], cx - row_x[obs_pixels[0]]]))) # distance from centroid to first obstacle pixel

                pts_obs_list.append([row_y[obs_pixels[0]], row_x[obs_pixels[0]]])

                slice_pts_dist = LA.norm((slice - np.array([cy, cx])), axis=1) #distance for each point in the slice to the centroid
                slice_keep = np.where(slice_pts_dist < obs_dist)[0]
                # pixels labeled as obstacle, use sqrt(2) for diagonal distance
                slice_obs = np.where(
                    np.logical_and(obs_dist - math.sqrt(2) < slice_pts_dist, slice_pts_dist < obs_dist + math.sqrt(2)))[
                    0]

                slice_obs_check = slice[slice_obs]
                obs_coords = np.where((obs_map[slice_obs_check[:, 0], slice_obs_check[:, 1], :] == (0, 0, 0)).all(axis=1))[0] #where slice obs coordinates are actually an obstacle
                if len(obs_coords) != 0:
                    slice_obs_coords.append(slice_obs_check[obs_coords])

                slice_check = slice[slice_keep]

                obs_coords = np.where((obs_map[slice_check[:, 0], slice_check[:, 1], :] == (0, 0, 0)).all(axis=1))[0] #where slice keep coordinates are actually an obstacle
                if len(obs_coords) != 0:
                    see_obs_list.append(slice_check[obs_coords])

                slice_keep_coords.append(slice[slice_keep]) #[[row, column]......]


        self.coords.see_obs_list = see_obs_list
        self.coords.pts_obs_list = pts_obs_list
        self.coords.slice_keep_coords = slice_keep_coords
        self.coords.slice_obs_coords = slice_obs_coords


class node_astar:
    def __init__(self):
        self.parent = None
        self.type = None
        self.position = None

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar_map(img):

    white = [255, 255, 255]
    black = [0, 0, 0]

    rows = img.shape[0]
    cols = img.shape[1]

    my_map = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            if img[i, j, :].tolist() != white:
                my_map[i, j] = 1

    return my_map


def refine_map(img):
    white = [255, 255, 255]
    black = [0, 0, 0]

    rows = img.shape[0]
    cols = img.shape[1]

    for i in range(rows):
        for j in range(cols):

            if img[i, j, :].tolist() != white:
                img[i, j, :] = black

    return img


def child_cost_greater(child, open_node):
    my_flag = False

    if child == open_node and child.g >= open_node.g:
        my_flag = True

    return my_flag


def find_positions(current_node, new_position):
    # Get node position
    node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

    return node_position


def check_astar_position(node_position, n_cols, n_rows, my_map):

    # make sure within range
    if node_position[0] > (n_cols - 1) or node_position[0] < 0 or node_position[1] > (n_rows - 1) or \
            node_position[1] < 0:
        return False

    # make sure map is traversable
    if my_map[node_position[1], node_position[0]] == 1:
        return False

    return True


def make_child(node_position, current_node):

    child = node_astar()
    child.parent = current_node
    child.position = node_position

    return child


def check_children(child, closed_list, current_node, goal_node, open_list):
    if child in closed_list:  # if child is in closed list continue
        return False

    # create the f, g, and h value
    dist = ((child.position[0] - current_node.position[0]) ** 2 + (
                child.position[1] - current_node.position[1]) ** 2) ** 0.5

    child.g = current_node.g + dist  # distance between current node and child
    child.h = ((child.position[0] - goal_node.position[0]) ** 2 + (
                child.position[1] - goal_node.position[1]) ** 2) ** 0.5
    child.f = child.g + child.h

    flags = [child_cost_greater(child, o) for o in open_list]

    if sum(flags) != 0:
        return False

    return True


def astar(my_map, start, goal):

    #positive directions: x-right, y-down
    #map_dimensions
    n_rows = my_map.shape[0]
    n_cols = my_map.shape[1]

    start_node = node_astar()
    start_node.position = start #(x, y)
    start_node.g = start_node.h = start_node.f = 0

    goal_node = node_astar()
    goal_node.position = goal #(x, y)
    goal_node.g = goal_node.h = goal_node.f = 0

    positions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_list = []
    closed_list = []

    open_list.append(start_node)

    path = []

    print("starting A* loop")
    while len(open_list) > 0:

        # get the current node, set as node with lowest f value
        current_node = min(open_list, key=attrgetter('f'))

        open_list = [node for node in open_list if node != current_node]
        closed_list.append(current_node)


        if current_node == goal_node:
            print("goal reached!")
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        new_positions = [find_positions(current_node, p) for p in positions]
        children = [make_child(node_position, current_node) for node_position in new_positions if
                    check_astar_position(node_position, n_cols, n_rows, my_map)]

        children_keep = [child for child in children if
                    check_children(child, closed_list, current_node, goal_node, open_list)]

        open_list += children_keep

    print("failed to find goal :(")
    return path


def replan_loop(all_obs, path, goal_loc, resize_params, time_step, seq, save_dir, output_dir, gt_map):

    # running the actual astar replanning loop
    current_step = 0
    step_disc = 2
    times_replan = 0
    path_step = 0
    real_path = []

    # for the measurement
    my_range = 30  # radius of measurement
    n_seg = 30
    slice_tol = 1e-9  # to prevent rounding errors when assigning points to slices
    ray_seg = my_range + 1  # resolution of ray discretization
    img_coords = utils_demo.get_img_coords(all_obs)

    while 1:

        if current_step % step_disc == 0:  # print the current step
            print("current step", current_step)

        if path_step % step_disc == 0:
            print("path step", path_step)
            centroid = path[path_step]

            # take the measurement
            measure_params = Measure_params(my_range, n_seg, (centroid[1], centroid[0]), slice_tol, ray_seg)
            measurement = Measurement()
            measurement.init(measure_params, time_step, seq, img_coords, output_dir, resize=resize_params)

            # repopulate the obstacle map
            slice_obs = measurement.coords.slice_obs_coords
            slice_keep = measurement.coords.slice_keep_coords
            see_obs = measurement.coords.see_obs_list

            for coords in slice_obs:
                all_obs[coords[:, 0], coords[:, 1], :] = red
            for coords in see_obs:
                all_obs[coords[:, 0], coords[:, 1], :] = red
            for coords in slice_keep:
                for coord in coords:
                    if all_obs[coord[0], coord[1], :].tolist() != red:
                        all_obs[coord[0], coord[1], :] = white

            # check if we need to replan
            replan = utils_demo.check_replan(path, path_step, all_obs)
            if replan:
                print("Need to replan now")
                start_loc = centroid
                my_map = astar_map(all_obs)
                path = astar(my_map, start_loc, goal_loc)
                path_step = 0
                times_replan += 1

                if len(path) == 0:
                    print("No path can be found after replanning")
                    return times_replan, path, all_obs


        #NOTE: fix for edge obstacle traversal bug
        check_step = path[path_step]
        if gt_map[check_step[1], check_step[0], :].tolist() != white: #if it traverses through an obstacle

            all_obs[check_step[1], check_step[0], :] = red #label that pixel as an obstacle

            print("Need to replan now, went through obstacle")
            start_loc = real_path[-1] #replan from previous step
            my_map = astar_map(all_obs)
            path = astar(my_map, start_loc, goal_loc)
            path_step = 1 #starting from one here since planner moved back a step
            times_replan += 1

            if len(path) == 0:
                print("No path can be found after replanning")
                return times_replan, path, all_obs

        my_loc = path[path_step]
        real_path.append(my_loc)

        if my_loc == goal_loc: #if the goal is reached
            print("goal reached!")
            print("total_steps", current_step)
            print("times_replan", times_replan)

            return times_replan, real_path, all_obs

        path_step += 1
        current_step += 1