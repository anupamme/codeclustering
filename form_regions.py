from sklearn import cluster, preprocessing
import ujson as json
import numpy as np
import itertools
from operator import itemgetter
import cv2
from utils import colormap
from sklearn.metrics import pairwise_distances

def get_centroids(boxes):
    centroids = []
    for box in boxes:
        cen = [(box['x0'] + box['x2']) / 2, (box['y0'] + box['y2']) / 2]
        centroids.append(cen)
    return np.array(centroids)

def assign_labels(boxes, labels):
    for i, box in enumerate(boxes):
        box['region'] = labels[i]
    return boxes

count = 0
max_count = 20

def vertical_more_imp(u, v):
    global count
    global max_count
    coeff_x = 0
    coeff_y = 1
#    print('coeff x, y: ' + str(coeff_x) + ' : ' + str(coeff_y))
    return np.sqrt(coeff_x * 2 * np.power(u[0] - v[0],2) + coeff_y * 2 * np.power(u[1] - v[1],2))

def balanced(u, v):
    global count
    global max_count
    coeff_x = 0.25
    coeff_y = 0.75
#    print('coeff x, y: ' + str(coeff_x) + ' : ' + str(coeff_y))
    return np.sqrt(coeff_x * 2 * np.power(u[0] - v[0],2) + coeff_y * 2 * np.power(u[1] - v[1],2))

def do_db_scan(boxes, index):
    centroids = get_centroids(boxes)
    X = preprocessing.StandardScaler().fit_transform(centroids)
    img = cv2.imread("img/input.png")
    if index == 0:
        distances = pairwise_distances(X, metric = vertical_more_imp)
    elif index == 1:
        distances = pairwise_distances(X, metric = balanced)
    db = cluster.DBSCAN(eps=0.15, min_samples=1, n_jobs=-3, metric='precomputed').fit(distances)
    return img, db

def print_rectangle(value, color_list, mask_color_id, cv2, img):
    left_xs = min(map(lambda x: int(np.round(x['x0'])), value))
    right_xs = max(map(lambda x: int(np.round(x['x2'])), value))
    top_ys = min(map(lambda x: int(np.round(x['y0'])), value))
    bottom_ys = max(map(lambda x: int(np.round(x['y2'])), value))
    tl = (left_xs, top_ys)
    br = (right_xs, bottom_ys)
    color = color_list[mask_color_id % len(color_list), 0:3]
    color = list(map(int, color))
    cv2.rectangle(img, tl, br, color, 3)

def call(input_data: dict, img_name: str):
    color_list = colormap(rgb=True)
    boxes = input_data['text_lines']
    img, db = do_db_scan(boxes, 0)
    labels = db.labels_
    # print(labels)
    boxes = assign_labels(boxes, labels)
    boxes = sorted(boxes, key=itemgetter('region'))
    mask_color_id = 0
    print("Number of cluster is {:2d}".format(len(set(db.labels_))))
    for key, value in itertools.groupby(boxes, key=itemgetter('region')):
        value = list(value)
        print_rectangle(value, color_list, mask_color_id, cv2, img)        
        mask_color_id += 1

        # do next clustering
        img2, db2 = do_db_scan(value, 1)
        labels2 = db2.labels_
        value = assign_labels(value, labels2)
        value = sorted(value, key=itemgetter('region'))
        print("Number of cluster is {:2d}".format(len(set(db2.labels_))))
        for subkey, subvalue in itertools.groupby(value, key=itemgetter('region')):
            subsubvalue = list(subvalue)
            print_rectangle(subsubvalue, color_list, mask_color_id, cv2, img2) 
            mask_color_id += 1
        cv2.imwrite(img_name + str(mask_color_id) + '.png', img2)
    cv2.imwrite(img_name + '.png', img)