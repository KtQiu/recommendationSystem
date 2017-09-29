#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2017/9/28 22:37
# @Author   : Kt Qiu
# @File     : loadData.py
# @Email    : kitty666ball@gmail.com
import numpy as np
import pandas as pd
# Load the data including the information of music, users and the data of performace
users = pd.read_table('users.txt', header = None, delim_whitespace =True)
trainTriplets = pd.read_table('train_triplets.txt', header=None, delim_whitespace= True)
data = pd.read_table('songs.txt',header=None,delim_whitespace=True)
songName = data[0]
songID = data[1]
# print(trainTriplets.shape, songName.shape, songID.shape, users.shape)
print("Load the data sussessfully!\n")
