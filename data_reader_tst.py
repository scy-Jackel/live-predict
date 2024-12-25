import os
import numpy as np

from data_reader import DATA_READER
from infer_reader import INFER_READER

if __name__ == '__main__':
    data_reader = DATA_READER()
    data_reader.seed(27)
    data_reader.shuff()
    data_reader.date_slice(0., 0.6)

    step0 = len(data_reader)

    detail = True

    for k in range(step0):
        # k=2
        ct_img_1, pet_img_1, date, live, dur = data_reader.get_one(k, train=False)
        path, pet_id= data_reader.get_path(k)
        infer_reader = INFER_READER(path=path)
        ct_img_2, pet_img_2 = infer_reader.get_data(pet_id)
        is_correct = np.all(ct_img_1==ct_img_2) and np.all(pet_img_1==pet_img_2)
        if is_correct:
            print(k, "is ok")
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(k, "is wrong")
            print("path is:", path)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # infer_reader = INFER_READER(path=r"E:\petct肺癌\肺癌资料petct\8695~8696")
    # ct_img, pet_img = infer_reader.get_data()
    # print(ct_img.shape)
    # print(pet_img.shape)

    print("test OK!")
