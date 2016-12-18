#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 16:39:42 2016

@author: Klemens
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
import pickle

def resize_image(image_name, enable_plot = False):
    image = mpimg.imread(image_name)
    
    image_resized = scipy.misc.imresize(image, (32, 32))
    if enable_plot:
        plt.imshow(image_resized)
    return image_resized

if __name__ == "__main__":
    dict_traffic_signs = {
                          'traffic_signs_images_square/einbahnstrasse_17_1.png': 17,
                          'traffic_signs_images_square/einbahnstrasse_17.png': 17,
                          'traffic_signs_images_square/vorfahrtgewaehren_13_1.png': 13,
                          'traffic_signs_images_square/vorfahrtgewaehren_13.png': 13,
                          'traffic_signs_images_square/vorfahrtsschild_12_1.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_2.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_3.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_4.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_5.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12.png': 12,
                          }
    
    image_list_web = []
    label_list_web = []

    for image_name, label in dict_traffic_signs.items():        
        image = resize_image(image_name)
        image_list_web.append(image)
        label_list_web.append(label)
    
    X_web = np.array(image_list_web)
    y_web = np.array(label_list_web)
    