#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/10/2 16:18
# @Author   : Kt Qiu
# @Email    : kitty666ball@gmail.com
# @GitHub   : https://github.com/KtQiu

import pandas as pd
import numpy as np

def lfm(user_item,k,alpha = 0.01,lamda = 0.01):
    """user_item is matrix of user item,k is the number of latent number"""
    if not isinstance(user_item,list):
        raise("user item:{user_item} is not a matrix list!".format(user_item=user_item))
    mat = np.array(user_item)
    # print(user_item)
    # print(k)
    # print(mat.shape)
    user_number,item_number = mat.shape
    # init the user and item latent matrix
    u = np.random.rand(user_number,k)
    v = np.random.rand(item_number,k)
    # print(u)
    # print('\n')
    # print(v)
    for it in range(1000):
        for i in range(user_number):
            for j in range(item_number):
                # err
                err = user_item[i][j] - np.dot(u[i], v[j])
                # print(user_item[i][j])
                # print(np.dot(u[i], v[j]))
                # print(err)
                # print('\n\n')
                for r in range(k):
                    gu = err * v[j][r] + lamda * u[i][r]
                    gv = err * u[i][r] + lamda * v[j][r]
                    u[i][r] += alpha * gu
                    v[j][r] += alpha * gv
    return u,v

mat = [[5,5,0,5],[5,0,3,4],[3,4,0,3],[0,0,5,3],[5,4,4,5],[5,4,5,5],[10,2,10,1],[1,23,12,1]]
u,v = lfm(mat,2)
print(u)    
print('\n')
print(v)
print('\n')
print(np.dot(u,v.T))