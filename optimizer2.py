# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:22:02 2025

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def draw_activation_hist(init_type='xavier'):
    x = np.random.randn(1000, 100)
    activations = {}
    node_num = 100
    hidden_layer_size = 5

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]

        if init_type == 'xavier':
            w = np.random.randn(node_num, node_num) * np.sqrt(1.3 / node_num)
        elif init_type == 'he':
            w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
        else:
            w = np.random.randn(node_num, node_num) * 0.21  # 기본값

        a = np.dot(x, w)
        z = relu(a)
        activations[i] = z

        plt.subplot(1, hidden_layer_size, i+1)
        plt.title(f"{i+1}층")
        plt.hist(z.flatten(), bins=30, range=(0, 1))

    plt.suptitle(f"{init_type} 초기화 활성화 분포")
    plt.show()

# 실험 실행
draw_activation_hist('xavier')
draw_activation_hist('he')
draw_activation_hist('default')
