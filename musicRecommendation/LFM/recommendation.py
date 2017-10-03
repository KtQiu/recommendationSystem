#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/10/3 14:52
# @Author   : Kt Qiu
# @Email    : kitty666ball@gmail.com
# @GitHub   : https://github.com/KtQiu

import random
import math
import numpy as np

def dataProcess(num):
    if num <= 1:
        num = -num
    else:
        num = math.sqrt(num)
    return num

data = []

for line in open('train_triplets.txt'):
    substrs = line.split('	')
    # data.append((substrs[0], substrs[1], float(substrs[2])/5))
    data.append((substrs[0], substrs[1], dataProcess(float(substrs[2]))))
print('Load data successfully!\n')
# data = data[:20000]


def InitLFM(datas, F):
    p = dict()
    q = dict()
    for u, i, rui in datas:
        if not u in p:
            p[u] = np.random.rand(F) / math.sqrt(F)
        if not i in q:
            q[i] = np.random.rand(F) / math.sqrt(F)
    print('Init data successfully!')
    return (p, q)


def Predict(u, i, p, q, F):
    return np.dot(p[u], q[i])


def LearningLFM(train, F, n, alpha, lam):
    (p, q) = InitLFM(data, F)
    for step in range(0, n):
        totalerror = 0.0
        for u, i, rui in train:
            pui = Predict(u, i, p, q, F)
            # print('Predict successfully!')
            eui = rui - pui
            totalerror += np.abs(eui)
            for f in range(0,F):

            # p[u] += alpha * (q[i] * eui - lam * p[u])
            # q[i] += alpha * (p[u] * eui - lam * q[i])
        print(step)
        print(totalerror)
        # print(totalerror)
        # print('%d'%n + totalerror)
        alpha *= 0.9
    return (p, q)


# print(len(data))
# print(data[:10])
(p, q) = LearningLFM(data, 100, 10000, 0.01, 0.1)
for u, i, rui in data[:10]:
    pui = Predict(u, i, p, q, 10)
    print(rui - pui)
