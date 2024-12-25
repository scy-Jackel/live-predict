# coding=utf-8
from os.path import dirname, join
import numpy as np

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir
import os
import csv
import random


class DATA_READER():
    def __init__(self):
        csv_path0 = './csvfiles/Ⅰ、Ⅱ期肺癌预后已随访1.csv'
        csv_path1 = './csvfiles/Ⅲ、Ⅳ期肺癌预后已随访1.csv'
        dcm_path = "/media/cuiyang/WD_ELEMENT/petct肺癌/"

        self.dcm_path = dcm_path

        folder_list0 = []
        for folder in sorted(os.listdir(dcm_path)):
            if not '.' in folder:
                folder_list0.append(folder)
        # print folder_list0

        folder_list1 = []

        for folder in folder_list0:
            folders = sorted(os.listdir(dcm_path + folder))
            folder_list1.append(folders)

        self.live_list = []
        self.dead_list = []
        csv_reader = csv.reader(open(csv_path0, encoding="utf-8"))
        line_num = -2
        for item in csv_reader:
            line_num += 1
            if line_num > 0 and len(item[27]):
                image_id = item[2]
                live_date = item[27]  # 存活时间（月数）
                # print image_id,live_date
                live = item[25]  # 是否存活

                dur_str = item[12]  # 分期
                # print dur_str
                if '2' in dur_str:
                    dur = 2
                else:
                    dur = 1

                if '死' in live:
                    live = 0
                else:
                    live = 1
                find_flag = 0
                for k in range(len(folder_list0)):
                    folder0 = folder_list0[k]
                    for kk in range(len(folder_list1[k])):
                        folder1 = folder_list1[k][kk]
                        if image_id in folder1:
                            find_flag = 1
                            if live:
                                self.live_list.append([folder0 + '/' + folder1, live_date, image_id, live, dur])
                            else:
                                self.dead_list.append([folder0 + '/' + folder1, live_date, image_id, live, dur])
                            break
                    if find_flag:
                        break
                # if find_flag == 0:
                #     print 'error:',image_id,live_date

        csv_reader = csv.reader(open(csv_path1, encoding="utf-8"))
        line_num = -2
        for item in csv_reader:
            line_num += 1
            if line_num > 0 and len(item[27]):
                image_id = item[2]
                live_date = item[27]
                # print image_id,live_date
                live = item[25]

                dur_str = item[12]
                # print dur_str
                if '4' in dur_str:
                    dur = 4
                else:
                    dur = 3

                if '死' in live:
                    live = 0
                else:
                    live = 1
                find_flag = 0
                for k in range(len(folder_list0)):
                    folder0 = folder_list0[k]
                    for kk in range(len(folder_list1[k])):
                        folder1 = folder_list1[k][kk]
                        if image_id in folder1:
                            find_flag = 1
                            if live:
                                self.live_list.append([folder0 + '/' + folder1, live_date, image_id, live, dur])
                            else:
                                self.dead_list.append([folder0 + '/' + folder1, live_date, image_id, live, dur])
                            break
                    if find_flag:
                        break
                # if find_flag == 0:
                #     print 'error:',image_id,live_date

        self.len = len(self.live_list) + len(self.dead_list)
        print("live list len:", len(self.live_list))
        print("dead list len:", len(self.dead_list))

        # for line in self.path_list:
        #     print line[3]
        print(self.len)

    def __len__(self):
        return self.len

    def shuff(self):
        random.shuffle(self.live_list)
        random.shuffle(self.dead_list)

    def seed(self, seed):
        random.seed(seed)

    def get_path(self, index0):
        index0 = index0 % self.len
        if index0 >= len(self.live_list):
            index0 = index0 - len(self.live_list)
            path0 = self.dead_list[index0][0]
            path = self.dcm_path + self.dead_list[index0][0]
            id = self.dead_list[index0][2]
        else:
            path0 = self.live_list[index0][0]
            path = self.dcm_path + self.live_list[index0][0]
            id = self.live_list[index0][2]
        return path, id

    def get_one(self, index0, train=True):
        if train:
            if index0 % 2 == 0:
                index0 = index0 // 2 % len(self.live_list)
                path0 = self.live_list[index0][0]
                path = self.dcm_path + self.live_list[index0][0] + '/DICOMDIR'
                date = self.live_list[index0][1]

                id = self.live_list[index0][2]
                live = self.live_list[index0][3]
                dur = self.live_list[index0][4]

                # print path,date,id, live
                dicom_dir = read_dicomdir(path)
                base_dir = dirname(path)
            else:
                index0 = index0 // 2 % len(self.dead_list)
                path0 = self.dead_list[index0][0]
                path = self.dcm_path + self.dead_list[index0][0] + '/DICOMDIR'
                date = self.dead_list[index0][1]

                id = self.dead_list[index0][2]
                live = self.dead_list[index0][3]
                dur = self.dead_list[index0][4]

                # print path,date,id, live
                dicom_dir = read_dicomdir(path)
                base_dir = dirname(path)
        else:
            index0 = index0 % self.len
            if index0 >= len(self.live_list):
                index0 = index0 - len(self.live_list)

                path0 = self.dead_list[index0][0]
                path = self.dcm_path + self.dead_list[index0][0] + '/DICOMDIR'
                if not os.path.exists(path):
                    return np.array([]), np.array([]), -1, -1, -1
                date = self.dead_list[index0][1]

                id = self.dead_list[index0][2]
                live = self.dead_list[index0][3]
                dur = self.dead_list[index0][4]

                # print path,date,id, live
                dicom_dir = read_dicomdir(path)
                base_dir = dirname(path)
            else:
                path0 = self.live_list[index0][0]
                path = self.dcm_path + self.live_list[index0][0] + '/DICOMDIR'
                if not os.path.exists(path):
                    return np.array([]), np.array([]), -1, -1, -1
                date = self.live_list[index0][1]

                id = self.live_list[index0][2]
                live = self.live_list[index0][3]
                dur = self.live_list[index0][4]
                # print path,date,id, live
                dicom_dir = read_dicomdir(path)
                base_dir = dirname(path)

        npy_list = []

        for patient_record in dicom_dir.patient_records:

            if (id not in str(patient_record.PatientName)) and (id not in patient_record.PatientID):
                continue

            for study in patient_record.children:
                for series in study.children:
                    image_records = series.children
                    if len(image_records) < 100:
                        continue

                    # date0 = str(series[0x0008, 0x0021])
                    # date0 = date0[date0.find('\'')+1:]
                    # date0 = date0[:date0.find('\'')]
                    # time0 = str(series[0x0008, 0x0031])
                    # time0 = time0[time0.find('\'') + 1:]
                    # time0 = time0[:time0.find('\'')]
                    # print '                   ',date0,time0

                    image_filenames = [join(base_dir, *image_rec.ReferencedFileID)
                                       for image_rec in image_records]

                    if '增加' in path:
                        sub_len = len(self.dcm_path + path0)
                        for k in range(len(image_filenames)):
                            filename = image_filenames[k]
                            name_add = filename[sub_len:]
                            name_clear = ''
                            for char_one in name_add:
                                if not char_one == '/':
                                    name_clear += char_one
                            name_clear = str(name_clear)
                            image_filenames[k] = self.dcm_path + path0 + \
                                                 '/' + name_clear

                    # get the pixel array
                    datasets = [pydicom.dcmread(image_filename).pixel_array
                                for image_filename in image_filenames]

                    # convert to numpy array
                    npa = np.array(datasets)
                    # print npa.shape
                    npy_list.append(npa)
        pet_img = np.array([])
        ct_img = np.array([])
        for k in range(len(npy_list)):
            if npy_list[k].shape[1] == 128:
                pet_img = npy_list[k] * 1
                del npy_list[k]
                break
        pet_dep = pet_img.shape[0]

        for npy in npy_list:
            if pet_dep == npy.shape[0] and npy.shape[1] == 512:
                ct_img = npy
                break

        return ct_img, pet_img, float(date), live, dur

    def date_slice(self, start=0., prop=0.9):
        end = start + prop
        start_live = int(start * len(self.live_list))
        end_live = int(end * len(self.live_list))
        self.live_list = self.live_list[start_live:end_live]
        start_dead = int(start * len(self.dead_list))
        end_dead = int(end * len(self.dead_list))
        self.dead_list = self.dead_list[start_dead:end_dead]
        self.len = len(self.live_list) + len(self.dead_list)
