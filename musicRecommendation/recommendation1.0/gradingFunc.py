#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/9/29 10:08
# @Author   : Kt Qiu
# @File     : gradingFunc.py
# @Email    : kitty666ball@gmail.com
import math


# the cost function
# root mean square error
def RMSE(records):
    return math.sqrt(sum([rui - pui] * (rui - pui) for u, i, rui, pui in records) / float(len(records)))


# mean absolute error
def MAE(records):
    return sum(abs(rui - pui) for u, i, rui, pui in records) / float(len(records))


# accuracy TopN
def PrecisionRecall(test, N):
    hit = 0
    n_recall = 0
    n_precision = 0
    for user, item in test.items():
        rank = Recommend(user, N)
        hit += len(rank & items)
        n_recall += len(items)
        n_precision += N
    return [hit / (1.0 * n_recall), hit / (1.0 * n_precision)]


def GiniIndex(p):
    j = 1
    n = len(p)
    G = 0
    for item, weight in sorted(p.item(), key=itemgetter(1)):
        G += (2 * j - n - 1) * weight
    return G / float(n - 1)
