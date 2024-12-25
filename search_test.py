# coding=utf-8
from os.path import dirname, join
import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
import os





search_id = '21489'
search_path = "/media/gdh-95/AE1ADC271ADBEA7B/petct肺癌/"

folder_list0 = []
for folder in sorted(os.listdir(search_path)):
    if not '.' in folder:
        folder_list0.append(folder)
# print folder_list0

folder_list1 = []

for folder in folder_list0:
    folders = sorted(os.listdir(search_path+folder))
    folder_list1.append(folders)


for k in range(len(folder_list0)):
    folder0 = folder_list0[k]
    for kk in range(len(folder_list1[k])):
        folder1 = folder_list1[k][kk]
        if search_id in  folder1:
            print(search_path+folder0+'/'+folder1)


