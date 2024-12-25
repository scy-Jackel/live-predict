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


class INFER_READER():
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 1

    def get_data(self, id):

        path = os.path.join(self.path, 'DICOMDIR')

        if not os.path.exists(path):
            print(path, "is not exist")
            return np.array([]), np.array([])

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

                    image_filenames = [join(base_dir, *image_rec.ReferencedFileID)
                                       for image_rec in image_records]

                    # if '增加' in path:
                    #     sub_len = len(self.dcm_path + path0)
                    #     for k in range(len(image_filenames)):
                    #         filename = image_filenames[k]
                    #         name_add = filename[sub_len:]
                    #         name_clear = ''
                    #         for char_one in name_add:
                    #             if not char_one == '/':
                    #                 name_clear += char_one
                    #         name_clear = str(name_clear)
                    #         image_filenames[k] = self.dcm_path + path0 + \
                    #                              '/' + name_clear

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

        return ct_img, pet_img
