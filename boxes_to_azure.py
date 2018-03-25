'''
Input: Pally OCR Output :: [rectangle]

Output: Azure Style Output

Process:
    1. Do verical clustering and identify main regions
    2. Do composite clustering on each regions to identify regions within regions
    3. Follow set of region combining rules 
        1. If a region is very less height (2 line height) and is within a region -> merge them
        2. If two regions' vertical distance is less than line gap height -> merge them
        3. If two regions' horizontal distance is less than word gap width -> merge them

'''

import sys
import glob
import os
import requests
import json

import form_regions as form

font_dim = [0, 29]

'''
1. sort the clusters vertically
2. start from top and inspect each region
3. if a region' bottom right is less than the bottom right of the parent then:
    if the region's left and right are also within parent:
        merge: add its polygons to the parent and mark it to be deleted.
4. if the regions top is too close vertically merge the two clusters
    
1. sort the clusters horizontally
2. start from left and move right incrementally
3. if the two regions are too close horizontally merge them
'''
def merge_composite(clusters):
    parent_cl_top = None
    parent_cl_bottom = None
    res = []
    for cl in clusters:
        top_l = cl['top_left']
        bottom_r = cl['bottom_right']
        if parent_cl_top != None and parent_cl_bottom != None:
            if top_l[1] > parent_cl_top[1] and bottom_r[1] < parent_cl_bottom[1]:
                # check for horizontal comp
                if top_l[0] > parent_cl_top[0] and bottom_r[0] < parent_cl_bottom[0]:
                    last_cl = res[-1]
                    last_cl['polygons'] = last_cl['polygons'] + cl['polygons']
                else:
                    res.append(cl)
            else:
                res.append(cl)
        else:
            res.append(cl)
        parent_cl_top = top_l
        parent_cl_bottom = bottom_r
    return res
            
def merge_adjacent (clusters, is_vertical):
    parent_cl_bottom = None
    res = []
    for cl in clusters:
        point_t = cl['top_left'][is_vertical]
        point_b = cl['bottom_right'][is_vertical]
        if parent_cl_bottom != None:
            if abs(point_t - parent_cl_bottom) < font_dim[is_vertical]:
                last_cl = res[-1]
                last_cl['polygons'] = last_cl['polygons'] + cl['polygons']
                last_cl['bottom_right'] = (max(last_cl['bottom_right'][0], cl['bottom_right'][0]), 
                                            max(last_cl['bottom_right'][1], cl['bottom_right'][1]))
                point_b = last_cl['bottom_right'][is_vertical]
            else:
                res.append(cl)
        else:
            res.append(cl)
        parent_cl_bottom = point_b
    return res

def merge_cluster(clusters: list):
    # sort vertically
    
    clusters.sort(key=lambda x: x['top_left'][1])
    # merge composite
    clusters_composite = merge_composite(clusters)
    # merge adjacent
    clusters_adjacent_v = merge_adjacent(clusters_composite, 1)
    # sort horizontally
    clusters_adjacent_v.sort(key=lambda x: x['top_left'][0])
    # merge adjacent
    clusters_adjacent_h = merge_adjacent(clusters_adjacent_v, 0)
    return clusters_adjacent_h


def do_clustering(boxes, img_name):
    outer, inner = form.call(boxes, img_name)
    inner_compact = merge_cluster(inner)
    return inner_compact
    
    
def call_ocr_api(img_name):
    url = 'http://ocr.callup.ai/ocr'
    response = requests.post(url, files={'image': ('tmp', open(img_name, 'rb'), 'image/jpeg', {'Expires': '0'})})
    res_json = json.loads(response.text)
    return res_json

if __name__ == "__main__":
    data = json.loads(open('4.json', 'r').read())
    outer, inner = form.call(data, '4.jpg')
    inner_compact = merge_cluster(inner)
    form.print_clusters('4.jpg', inner_compact)
#    img_dir = 'img/*.jpg'
#    images_list = glob.glob(img_dir)
#    for img in images_list:
#        boxes = call_ocr_api(img)
#        do_clustering(boxes, img)
    
