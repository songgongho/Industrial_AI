# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:37:14 2025

@author: User
"""
'''
import numpy as np
import matplotlib.pylab as plt



def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
    x = np.array([-1.0, 1.0, 2.0])
    sigmoid(x)
array([ 0.26894142, 0.73105858, 0.88079708])

    t = np.array([1.0, 2.0, 3.0])
    1.0 + t
array([2., 3., 4.])
    1.0/t
array([1.          , 0.5       , 0.3333333])


x = np.arange(-0.5, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show() 
      
      '''
"""  
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid 함수 테스트용 코드
x_test = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x_test))  # [0.26894142 0.73105858 0.88079708]

# 예시로 보여준 기타 연산 테스트
t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)  # [2. 3. 4.]
print(1.0 / t)  # [1.         0.5        0.33333333]

# 시그모이드 함수의 그래프 그리기
x = np.arange(-0.5, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


"""

import numpy as np

A = np.array([1, 2, 3, 4])
print("A:", A)
print("Shape of A:", A.shape)
print("Length of A (A.shape[0]):", A.shape[0])




B = np.array([[1,2], [3,4], [5,6]])
print(B)

np.ndim(B)


B.shape




