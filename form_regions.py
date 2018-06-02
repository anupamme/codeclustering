from sklearn import cluster, preprocessing
import ujson as json
import numpy as np
import itertools
from operator import itemgetter
import cv2
from utils import colormap
from sklearn.metrics import pairwise_distances
import random

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

def vertical_more_imp(u, v):
    coeff_x = 0
    coeff_y = 1
#    print('coeff x, y: ' + str(coeff_x) + ' : ' + str(coeff_y))
    return np.sqrt(coeff_x * 2 * np.power(u[0] - v[0],2) + coeff_y * 2 * np.power(u[1] - v[1],2))

def balanced(u, v):
    coeff_x = 0.25
    coeff_y = 0.75
#    print('coeff x, y: ' + str(coeff_x) + ' : ' + str(coeff_y))
    return np.sqrt(coeff_x * 2 * np.power(u[0] - v[0],2) + coeff_y * 2 * np.power(u[1] - v[1],2))

def do_db_scan(boxes, index, img_name):
    centroids = get_centroids(boxes)
    X = preprocessing.StandardScaler().fit_transform(centroids)
    img = cv2.imread(img_name)
    if index == 0:
        distances = pairwise_distances(X, metric = vertical_more_imp)
    elif index == 1:
        distances = pairwise_distances(X, metric = balanced)
    db = cluster.DBSCAN(eps=0.15, min_samples=2, n_jobs=-3, metric='precomputed').fit(distances)
    return img, db

def calculate_boundaries(value: list):
    left_xs = min(map(lambda x: int(np.round(x['x0'])), value))
    right_xs = max(map(lambda x: int(np.round(x['x2'])), value))
    top_ys = min(map(lambda x: int(np.round(x['y0'])), value))
    bottom_ys = max(map(lambda x: int(np.round(x['y2'])), value))
    tl = (left_xs, top_ys)
    br = (right_xs, bottom_ys)
    return tl, br

def print_rectangle(value, color_list, mask_color_id, img):
    tl, br = calculate_boundaries(value)
    color = color_list[mask_color_id % len(color_list), 0:3]
    color = list(map(int, color))
    cv2.rectangle(img, tl, br, color, 3)
    
def print_clusters(img_file, clusters):
    img = cv2.imread(img_file)
    color_list = colormap(rgb=True)
    for cl in clusters:
        tl, br = cl['top_left'], cl['bottom_right']
        color = color_list[random.randint(0, len(color_list) - 1), 0:3]
        color = list(map(int, color))
        cv2.rectangle(img, tl, br, color, 3)
    cv2.imwrite(img_file + '.png', img)
    
    
    
def create_obj(tl, br, value: list):
    return {
        'top_left': tl,
        'bottom_right': br,
        'polygons': value
    }

def call(input_data: dict, img_name: str):
    color_list = colormap(rgb=True)
    boxes = input_data['text_lines']
    img, db = do_db_scan(boxes, 0, img_name)
    labels = db.labels_
    # print(labels)
    boxes = assign_labels(boxes, labels)
    boxes = sorted(boxes, key=itemgetter('region'))
    mask_color_id = 0
    outer_list = []
    inner_list = []
    print("Number of cluster is {:2d}".format(len(set(db.labels_))))
    for key, value in itertools.groupby(boxes, key=itemgetter('region')):
        value = list(value)
        tl, br = calculate_boundaries(value)
        outer_list.append(create_obj(tl, br, value))
        print_rectangle(value, color_list, mask_color_id, img)
        mask_color_id += 1
        
        # do next clustering
        img2, db2 = do_db_scan(value, 1, img_name)
        labels2 = db2.labels_
        value = assign_labels(value, labels2)
        value = sorted(value, key=itemgetter('region'))
        print("Number of cluster is {:2d}".format(len(set(db2.labels_))))
        for subkey, subvalue in itertools.groupby(value, key=itemgetter('region')):
            subsubvalue = list(subvalue)
            tl_i, br_i = calculate_boundaries(subsubvalue)
            inner_list.append(create_obj(tl_i, br_i, subsubvalue))
            print_rectangle(subsubvalue, color_list, mask_color_id, img2) 
            mask_color_id += 1
        cv2.imwrite(img_name + str(mask_color_id) + '.png', img2)
    cv2.imwrite(img_name + '.png', img)
    return outer_list, inner_list
    
if __name__ == "__main__":
    data = json.loads(open('4.json', 'r').read())
    outer, inner = call(data, '4.jpg')
#    f = open('outer.json', 'w')
#    f.write(json.dumps(outer))
#    f.close()
#    f = open('inner.json', 'w')
#    f.write(json.dumps(inner))
#    f.close()
