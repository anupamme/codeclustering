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
    coeff_x = (float)(count/max_count)
    coeff_y = 1 - (float)(count/max_count)
    print('coeff x, y: ' + str(coeff_x) + ' : ' + str(coeff_y))
    return coeff_x * 2 * np.power(u[0] - v[0],2) + coeff_y * 2 * np.power(u[1] - v[1],2)

if __name__ == '__main__':
    boxes = json.load(open('result.json', 'r'))['text_lines']
    centroids = get_centroids(boxes)
    X = preprocessing.StandardScaler().fit_transform(centroids)
    global count
    global max_count
    while count < max_count:
        img = cv2.imread("img/input.png")
        distances = pairwise_distances(X, metric = vertical_more_imp)
        db = cluster.DBSCAN(eps=0.15, min_samples=1, n_jobs=-3, metric='precomputed').fit(distances)
        labels = db.labels_
        # print(labels)
        boxes = assign_labels(boxes, labels)
        boxes = sorted(boxes, key=itemgetter('region'))
        regions = []
        mask_color_id = 0
        color_list = colormap(rgb=True)
        print("Number of cluster is {:2d}".format(len(set(db.labels_))))
        for key, value in itertools.groupby(boxes, key=itemgetter('region')):
            value = list(value)
            left_xs = min(map(lambda x: int(np.round(x['x0'])), value))
            right_xs = max(map(lambda x: int(np.round(x['x2'])), value))
            top_ys = min(map(lambda x: int(np.round(x['y0'])), value))
            bottom_ys = max(map(lambda x: int(np.round(x['y2'])), value))
            tl = (left_xs, top_ys)
            br = (right_xs, bottom_ys)
            color = color_list[mask_color_id % len(color_list), 0:3]
            color = list(map(int, color))
            mask_color_id += 1
            cv2.rectangle(img, tl, br, color, 3)
        cv2.imwrite('output_groups_' + str(count) + '.png', img)
        count += 1
    # img = cv2.resize(img, (1024, 1024))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
