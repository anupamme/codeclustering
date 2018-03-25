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

def merge_cluster_subsume(clusters: list):
    pass

def merge_cluster_too_near(clusters: list):
    pass

def convert_into_azure(clusters: list):
    pass

def do_clustering(boxes, img_name):
    form.call(boxes, img_name)
    
def call_ocr_api(img_name):
    url = 'http://ocr.callup.ai/ocr'
    response = requests.post(url, files={'image': ('tmp', open(img_name, 'rb'), 'image/jpeg', {'Expires': '0'})})
    res_json = json.loads(response.text)
    return res_json

if __name__ == "__main__":
    img_dir = 'img/*.jpg'
    images_list = glob.glob(img_dir)
    for img in images_list:
        boxes = call_ocr_api(img)
        do_clustering(boxes, img)
    