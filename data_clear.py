# coding=utf-8
from os.path import dirname, join
import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
import os
import csv
import random

# csv_path0 = '../Ⅰ、Ⅱ期肺癌预后已随访1.csv'
# csv_path1 = '../Ⅰ、Ⅱ期肺癌预后已随访1_clear.csv'
csv_path0 = '../Ⅲ、Ⅳ期肺癌预后已随访1.csv'
csv_path1 = '../Ⅲ、Ⅳ期肺癌预后已随访1_clear.csv'
dcm_path = "/media/gdh-95/data/petct肺癌/"

out_list = []
csv_reader = csv.reader(open(csv_path0))
csv_writer = csv.writer(open(csv_path1, 'w'))

folder_list0 = []
for folder in sorted(os.listdir(dcm_path)):
    if not '.' in folder:
        folder_list0.append(folder)
# print folder_list0

folder_list1 = []

for folder in folder_list0:
    folders = sorted(os.listdir(dcm_path + folder))
    folder_list1.append(folders)

line_num = -2
for item in csv_reader:
    line_num += 1
    if line_num<= 0:
        out_list.append(item)
    if line_num > 0 and len(item[27]):
        image_id = item[2]
        live_date = item[27]
        find_flag = 0
        for k in range(len(folder_list0)):
            folder0 = folder_list0[k]
            for kk in range(len(folder_list1[k])):
                folder1 = folder_list1[k][kk]
                if image_id in folder1:
                    find_flag = 1
                    out_list.append(item)
                    break
            if find_flag:
                break



print(len(out_list))
for item in out_list:
    csv_writer.writerow(item)