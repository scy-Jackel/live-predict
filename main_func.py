# coding=utf-8
import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import cv2
import shutil
import torch.nn.functional as F
import math
import random
import util
from data_reader import DATA_READER
from infer_reader import INFER_READER
from model import LIVE_NET
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from lifelines.statistics import logrank_test
from loss_record import RECORDER


def train():
    live_net = LIVE_NET()
    live_net.cuda().train()

    data_reader = DATA_READER()
    data_reader.seed(27)
    data_reader.shuff()
    data_reader.date_slice(0., 0.6)
    batch = 10
    lr0 = 1e-4
    weight_decay0 = 1
    optimizer = torch.optim.Adam(util.models_parameters([live_net]), lr0, weight_decay=weight_decay0)

    ckpt_path = "./ckpt_live/"
    data_path, step = util.load_weight(ckpt_path)
    if step:
        live_net.load_state_dict(torch.load(data_path))

    ckpt_path_opt = "./ckpt_opt/"
    data_path_opt, step = util.load_weight(ckpt_path_opt)
    if step:
        optimizer.load_state_dict(torch.load(data_path_opt))

    step0 = 1000001

    recoder = RECORDER("./loss_record.csv")

    for k in range(step + 1, step0):
        print(k)
        ct_img, pet_img, date, live, _= data_reader.get_one(k)

        if random.randint(0, 1):
            ct_img = ct_img[::-1, :, :]
            pet_img = pet_img[:, ::-1, :]

        ct_img = ct_img.astype('float32')
        ct_img = torch.from_numpy(ct_img)
        ct_img = ct_img.cuda()

        pet_img = pet_img.astype('float32')
        pet_img = torch.from_numpy(pet_img)
        pet_img = pet_img.cuda()

        date_out, live_out = live_net(ct_img, pet_img)
        if live == 0:
            error = (date_out / date - date / date_out).abs().sum() + ((live_out - live) * (live_out - live)).sum()
        else:
            error = F.relu(date / date_out - date_out / date).sum() + ((live_out - live) * (live_out - live)).sum()

        error.backward()
        if k % batch == 0:
            optimizer.step()
            optimizer.zero_grad()

        recoder.write_date([k, float(error.cpu().data.numpy()), \
                            float(live_out[0][0].cpu().data.numpy()), \
                            live, \
                            float(date_out[0][0].cpu().data.numpy()), \
                            date])
        print("         loss:", error.cpu().data.numpy(), \
              '   live_out:', live_out[0][0].cpu().data.numpy(), \
              '   live:', live, \
              '   date_out:', date_out[0][0].cpu().data.numpy(), \
              '   date:', date)

        if k % 5000 == 0 and k > 0:
            print("save weight at step:%d" % (k))
            util.save_weight(live_net, k, ckpt_path)
            util.save_weight(optimizer, k, ckpt_path_opt)


def test():
    torch.set_grad_enabled(False)
    live_net = LIVE_NET()
    live_net.cuda().eval()

    data_reader = DATA_READER()
    data_reader.seed(27)
    data_reader.shuff()
    data_reader.date_slice(0.9, 0.1)
    # data_reader.date_slice(0., 0.9)

    ckpt_path = "./ckpt_live/"
    data_path, step = util.load_weight(ckpt_path)
    if step:
        live_net.load_state_dict(torch.load(data_path))

    # step0 = len(data_reader)
    step0 = 10

    corr_live = 0
    live_num = 0

    corr_dead = 0
    dead_num = 0

    corr_all = 0
    all_num = 0

    date_error = 0

    for k in range(0, step0):
        print(k)
        ct_img, pet_img, date, live, _= data_reader.get_one(k, train=False)
        ct_img = ct_img.astype('float32')
        ct_img = torch.from_numpy(ct_img)
        ct_img = ct_img.cuda()

        pet_img = pet_img.astype('float32')
        pet_img = torch.from_numpy(pet_img)
        pet_img = pet_img.cuda()

        date_out, live_out = live_net(ct_img, pet_img)
        if live == 0:
            error = (date_out / date - date / date_out).abs().sum() + ((live_out - live) * (live_out - live)).sum()
        else:
            error = F.relu(date / date_out - date_out / date).sum() + ((live_out - live) * (live_out - live)).sum()

        all_num += 1
        if live:
            live_num += 1
            if live_out > 0.5:
                corr_live += 1
                corr_all += 1
        else:
            dead_num += 1
            if live_out < 0.5:
                corr_dead += 1
                corr_all += 1
                date_error += (date_out - date).abs().sum().cpu().data.numpy()

        print("         loss:", error.cpu().data.numpy(), \
              '   live_out:', live_out[0][0].cpu().data.numpy(), \
              '   live:', live, \
              '   date_out:', date_out[0][0].cpu().data.numpy(), \
              '   date:', date)

    print("corr_live:", float(corr_live) / float(live_num+1e-5))
    print("corr_dead:", float(corr_dead) / float(dead_num+1e-5))
    print("corr_all:", float(corr_all) / float(all_num))
    print("date_err:", float(date_error) / float(dead_num+1e-5))


def view():
    live_net = LIVE_NET()
    live_net.cuda().train()

    data_reader = DATA_READER()
    data_reader.seed(27)
    data_reader.shuff()
    data_reader.date_slice(0., 0.9)

    ct_img, pet_img, date, live, _= data_reader.get_one(0)

    save_num = 0

    for npy in [ct_img, pet_img]:
        npy_max = npy.max()
        npy = npy / float(npy_max) * 255

        depth = npy.shape[0]
        out_path = './view' + str(save_num) + '/'
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for k in range(depth):
            slice0 = npy[k]
            cv2.imwrite(out_path + 'a_' + str(k) + '.jpg', slice0)

        save_num += 1


def view_single():
    live_net = LIVE_NET()
    live_net.cuda().eval()

    ckpt_path = "./ckpt_live/"
    data_path, step = util.load_weight(ckpt_path)
    print('DEBUG: data_path, step:', data_path, step)
    if step:
        live_net.load_state_dict(torch.load(data_path))

    path = r"E:\pet肺癌\肺癌资料刻盘\2953~2954"
    pet_id = "2953"
    infer_reader = INFER_READER(path)
    ct_img, pet_img = infer_reader.get_data(id=pet_id)
    ct_img = ct_img.astype('float32')
    ct_img = torch.from_numpy(ct_img)
    ct_img = ct_img.cuda()

    pet_img = pet_img.astype('float32')
    pet_img = torch.from_numpy(pet_img)
    pet_img = pet_img.cuda()
    # print('DEBUG IMG SHAPE:', ct_img.cpu().data.numpy().shape, pet_img.cpu().data.numpy().shape)

    date_out, live_out = live_net(ct_img, pet_img)
    print("date_out:", float(date_out[0][0].cpu().data.numpy()))
    print("live_out:", float(live_out[0][0].cpu().data.numpy()))
    print('view single ok')


def view_all():
    live_net = LIVE_NET()
    live_net.cuda().eval()

    ckpt_path = "./ckpt_live/"
    data_path, step = util.load_weight(ckpt_path)
    if step:
        live_net.load_state_dict(torch.load(data_path))

    data_reader = DATA_READER()
    data_reader.seed(27)
    data_reader.shuff()
    data_reader.date_slice(0., 0.6)
    step0 = len(data_reader)
    result_list = []
    for k in range(0, step0):
        print(k)
        ct_img_ori, pet_img_ori, date, live, dur = data_reader.get_one(k, train=False)
        if date == -1:
            continue
        path, pet_id = data_reader.get_path(k)
        infer_reader = INFER_READER(path)
        ct_img_infer, pet_img_infer = infer_reader.get_data(pet_id)

        ct_img_ori = ct_img_ori.astype('float32')
        ct_img_ori = torch.from_numpy(ct_img_ori)
        ct_img_ori = ct_img_ori.cuda()

        pet_img_ori = pet_img_ori.astype('float32')
        pet_img_ori = torch.from_numpy(pet_img_ori)
        pet_img_ori = pet_img_ori.cuda()

        ct_img_infer = ct_img_infer.astype('float32')
        ct_img_infer = torch.from_numpy(ct_img_infer)
        ct_img_infer = ct_img_infer.cuda()

        pet_img_infer = pet_img_infer.astype('float32')
        pet_img_infer = torch.from_numpy(pet_img_infer)
        pet_img_infer = pet_img_infer.cuda()

        date_out_ori, live_out_ori = live_net(ct_img_ori, pet_img_ori)
        date_out_infer, live_out_infer = live_net(ct_img_infer, pet_img_infer)

        live_out_ori = float(live_out_ori[0][0].cpu().data.numpy())
        date_out_ori = float(date_out_ori[0][0].cpu().data.numpy())
        live_out_infer = float(live_out_infer[0][0].cpu().data.numpy())
        date_out_infer = float(date_out_infer[0][0].cpu().data.numpy())
        # result_one = [live_out, live, date_out, date]
        result_one = [live_out_ori, live_out_infer, live, date_out_ori, date_out_infer, date]
        result_list.append(result_one)
        flag = False
        if result_one[0] == result_one[1] and result_one[3] == result_one[4]:
            flag = True
        print(result_one, flag)

    print('view all ok')


def draw_km():
    torch.set_grad_enabled(False)
    live_net = LIVE_NET()
    live_net.cuda().eval()

    data_reader = DATA_READER()
    data_reader.seed(27)
    data_reader.shuff()

    data_reader.date_slice(0., 0.6)

    ckpt_path = "./ckpt_live/"
    data_path, step = util.load_weight(ckpt_path)

    if step:
        live_net.load_state_dict(torch.load(data_path))

    step0 = len(data_reader)

    result_list = []
    for k in range(0, step0):
        print(k)
        ct_img, pet_img, date, live, dur = data_reader.get_one(k, train=False)
        if date == -1:
            continue
        # if dur != 3:
        #     continue

        ct_img = ct_img.astype('float32')
        ct_img = torch.from_numpy(ct_img)
        ct_img = ct_img.cuda()

        pet_img = pet_img.astype('float32')
        pet_img = torch.from_numpy(pet_img)
        pet_img = pet_img.cuda()

        date_out, live_out = live_net(ct_img, pet_img)

        live_out = float(live_out[0][0].cpu().data.numpy())
        date_out = float(date_out[0][0].cpu().data.numpy())
        result_one = [live_out, live, date_out, date]
        # print k,result_one
        result_list.append(result_one)

    result_list.sort(reverse=True, key=lambda x: x[0])
    result_sort = []
    for result_one in result_list:
        if result_one[0] > 0.5:
            result_sort.append(result_one)
        else:
            break
    result_list.sort(reverse=True, key=lambda x: x[2])
    for result_one in result_list:
        if result_one[0] <= 0.5:
            result_sort.append(result_one)

    # for result_one in result_sort:
    #     print result_one
    low_risk = result_sort[:int(len(result_sort) / 2)]
    high_risk = result_sort[int(len(result_sort) / 2):]

    low_risk.sort(key=lambda x: x[1] * 1000 + x[3])
    high_risk.sort(key=lambda x: x[1] * 1000 + x[3])

    # print 'low_risk:'
    # for result_one in low_risk:
    #     print result_one
    #
    # print 'high_risk:'
    # for result_one in high_risk:
    #     print result_one

    low_index = 0
    high_index = 0

    low_num = len(low_risk)
    high_num = len(high_risk)

    # mounth_max = 120
    print(high_num, low_num)

    low_dura = []
    low_event = []
    high_dura = []
    high_event = []

    for one_data in low_risk:
        low_dura.append(one_data[3])
        if one_data[1]:
            low_event.append(False)
        else:
            low_event.append(True)

    for one_data in high_risk:
        high_dura.append(one_data[3])
        if one_data[1]:
            high_event.append(False)
        else:
            high_event.append(True)
    ax = plt.subplot(111)
    t = np.linspace(0, 100, 101)
    kmf = KaplanMeierFitter()

    kmf.fit(low_dura, low_event, label='low risk group', timeline=t)
    ax = kmf.plot(ax=ax)
    kmf.fit(high_dura, high_event, label='high risk group', timeline=t)
    ax = kmf.plot(ax=ax)

    res = logrank_test(high_dura, low_dura, high_event, low_event)
    p_value = res.p_value

    if p_value < 0.00001:
        str_out = 'p<0.00001'
    else:
        str_out = 'p=' + str(int(p_value * 100000) / 100000.)

    plt.annotate(s=str_out, xy=(75, 0.8))
    # plt.annotate(s=str_out, xy=(65, 0.64))
    plt.ylim(0., 1.1)
    plt.xlim(0, 100)
    plt.ylabel('Survival possibility')
    plt.xlabel('Time(mounths)')
    plt.legend(loc='lower left', shadow=True, )
    # plt.legend(loc='upper right', shadow=True, )
    plt.grid(True)
    plt.savefig("./km.png")
    plt.show()

    # for k in range(mounth_max):
    #     if low_index<low_num:
    #         if low_risk[low_index][1]==0 and low_risk[low_index][3]<k:
    #             low_index+=1
    #     if high_index<high_num:
    #         if high_risk[high_index][1]==0 and high_risk[high_index][3]<k:
    #             high_index+=1
    #     low_plot.append(float(low_num-low_index)/float(low_num))
    #     high_plot.append(float(high_num - high_index) / float(high_num))
    # plt.ylabel('Survival possibility')
    # plt.xlabel('Time(mounths)')
    # plt.xlim(0, 120)
    # plt.ylim(0., 1.)
    # plt.plot(low_plot,label= 'low risk group')
    # plt.plot(high_plot,label= 'high risk group')
    # plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    # plt.grid(True)
    # plt.savefig("./km.png")
    # plt.show()


def test_year():
    torch.set_grad_enabled(False)
    live_net = LIVE_NET()
    live_net.cuda().eval()

    data_reader = DATA_READER()
    data_reader.seed(27)
    data_reader.shuff()
    data_reader.date_slice(0.875, 0.125)
    # data_reader.date_slice(0., 0.9)

    ckpt_path = "./ckpt_live/"
    data_path, step = util.load_weight(ckpt_path)
    if step:
        live_net.load_state_dict(torch.load(data_path))

    step0 = len(data_reader)

    corr_6 = 0
    corr_12 = 0
    corr_24 = 0
    corr_36 = 0
    corr_48 = 0
    corr_60 = 0
    corr_72 = 0
    corr_84 = 0
    corr_fin = 0
    for k in range(0, step0):
        print(k)
        ct_img, pet_img, date, live = data_reader.get_one(k, train=False)
        ct_img = ct_img.astype('float32')
        ct_img = torch.from_numpy(ct_img)
        ct_img = ct_img.cuda()

        pet_img = pet_img.astype('float32')
        pet_img = torch.from_numpy(pet_img)
        pet_img = pet_img.cuda()

        date_out, live_out = live_net(ct_img, pet_img)
        live_out = float(live_out[0][0].cpu().data.numpy())
        date_out = float(date_out[0][0].cpu().data.numpy())

        if date > 6 or live == 1:
            if date_out > 6 or live_out == 1:
                corr_6 += 1
        else:
            if date_out <= 6 and live_out == 0:
                corr_6 += 1

        if date > 12 or live == 1:
            if date_out > 12 or live_out == 1:
                corr_12 += 1
        else:
            if date_out <= 12 and live_out == 0:
                corr_12 += 1

        if date > 24 or live == 1:
            if date_out > 24 or live_out >= 0.5:
                corr_24 += 1
        else:
            if date_out <= 24 and live_out < 0.5:
                corr_24 += 1

        if date > 36 or live == 1:
            if date_out > 36 or live_out >= 0.5:
                corr_36 += 1
        else:
            if date_out <= 36 and live_out < 0.5:
                corr_36 += 1

        if date > 48 or live == 1:
            if date_out > 48 or live_out >= 0.5:
                corr_48 += 1
        else:
            if date_out <= 48 and live_out < 0.5:
                corr_48 += 1

        if date > 60 or live == 1:
            if date_out > 60 or live_out >= 0.5:
                corr_60 += 1
        else:
            if date_out <= 60 and live_out < 0.5:
                corr_60 += 1

        if date > 72 or live == 1:
            if date_out > 72 or live_out >= 0.5:
                corr_72 += 1
        else:
            if date_out <= 72 and live_out < 0.5:
                corr_72 += 1

        if date > 84 or live == 1:
            if date_out > 84 or live_out >= 0.5:
                corr_84 += 1
        else:
            if date_out <= 84 and live_out < 0.5:
                corr_84 += 1

        if live == 1:
            if live_out >= 0.5:
                corr_fin += 1
        else:
            if live_out < 0.5:
                corr_fin += 1

    print("corr_6:", float(corr_6) / step0)
    print("corr_12:", float(corr_12) / step0)
    print("corr_24:", float(corr_24) / step0)
    print("corr_36:", float(corr_36) / step0)
    print("corr_48:", float(corr_48) / step0)
    print("corr_60:", float(corr_60) / step0)
    print("corr_72:", float(corr_72) / step0)
    print("corr_84:", float(corr_84) / step0)
    print("corr_fin:", float(corr_fin) / step0)
