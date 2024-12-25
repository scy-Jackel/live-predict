# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_num_2d(input, output_num):
    input_num = input.size(1)
    rest = output_num%input_num
    cop = output_num/input_num

    if rest:
        con_list = [input[:,0:rest,:,:]]
    else:
        con_list = []
    for i in range(int(cop)):
        con_list.append(input)
    out_put = torch.cat(con_list,dim=1)

    return out_put

def adjust_num_3d(input, output_num):
    input_num = input.size(1)
    rest = output_num%input_num
    cop = output_num/input_num
    if rest:
        con_list = [input[:,0:rest,:,:,:]]
    else:
        con_list = []
    for i in range(int(cop)):
        con_list.append(input)
    out_put = torch.cat(con_list,dim=1)
    return out_put


class LIVE_NET(nn.Module):

    def __init__(self):
        super(LIVE_NET, self).__init__()
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.avg2d = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg3d = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        chan_list = [1,8,16,32,48,64,96,128,192,256]
        self.chan_list = chan_list
        # ct
        self.ct_2d_1 = nn.Sequential(
            nn.BatchNorm2d(chan_list[0], momentum=0.05),
            nn.Conv2d(chan_list[0], chan_list[1], kernel_size=3, stride=2, padding=1, bias=True),
        )
        self.ct_2d_2 = nn.Sequential(
            nn.BatchNorm2d(chan_list[1], momentum=0.05),
            nn.Conv2d(chan_list[1], chan_list[2]-1, kernel_size=3, stride=2, padding=1, bias=True),
        )
        self.ct_2d_3 = nn.Sequential(
            nn.BatchNorm2d(chan_list[2], momentum=0.05),
            nn.Conv2d(chan_list[2], chan_list[3], kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.ct_3d_4 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=chan_list[3]),
            nn.Conv3d(chan_list[3], chan_list[4], kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.ct_3d_5 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=chan_list[4]),
            nn.Conv3d(chan_list[4], chan_list[5], kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.ct_3d_6 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=chan_list[5]),
            nn.Conv3d(chan_list[5], chan_list[6], kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.ct_3d_7 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=chan_list[6]),
            nn.Conv3d(chan_list[6], chan_list[7], kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.ct_3d_8 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=chan_list[7]),
            nn.Conv3d(chan_list[7], chan_list[8], kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.ct_3d_9 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=chan_list[8]),
            nn.Conv3d(chan_list[8], chan_list[9], kernel_size=3, stride=2, padding=1, bias=True),
        )

        self.fc0 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(64, 1)
    def forward(self, ct_img,pet_img):
        ct_img_min = ct_img.min()
        ct_img_max = ct_img.max()
        ct_img = (ct_img-ct_img_min)/(ct_img_max-ct_img_min+1e-9)


        pet_img_min = pet_img.min()
        pet_img_max = pet_img.max()
        pet_img = (pet_img - pet_img_min) / (pet_img_max - pet_img_min + 1e-9)
        ct_img = ct_img.unsqueeze(1)

        ct_img = F.relu(self.ct_2d_1(ct_img) +
                        adjust_num_2d(self.avg2d(ct_img),self.chan_list[1]))


        ct_img = F.relu(self.ct_2d_2(ct_img) +
                        adjust_num_2d(self.avg2d(ct_img), self.chan_list[2]-1))


        pet_img = pet_img.unsqueeze(1)
        ct_img = torch.cat([ct_img,pet_img],dim = 1)



        ct_img = F.relu(self.ct_2d_3(ct_img) +
                        adjust_num_2d(self.avg2d(ct_img), self.chan_list[3]))


        ct_img = torch.transpose(ct_img, dim0=0, dim1=1)
        ct_img = ct_img.unsqueeze(0)

        ct_img = F.relu(self.ct_3d_4(ct_img) +
                        adjust_num_3d(self.avg3d(ct_img), self.chan_list[4]))
        ct_img = F.relu(self.ct_3d_5(ct_img) +
                        adjust_num_3d(self.avg3d(ct_img), self.chan_list[5]))
        ct_img = F.relu(self.ct_3d_6(ct_img) +
                        adjust_num_3d(self.avg3d(ct_img), self.chan_list[6]))
        ct_img = F.relu(self.ct_3d_7(ct_img) +
                        adjust_num_3d(self.avg3d(ct_img), self.chan_list[7]))
        ct_img = F.relu(self.ct_3d_8(ct_img) +
                        adjust_num_3d(self.avg3d(ct_img), self.chan_list[8]))
        ct_img = F.relu(self.ct_3d_9(ct_img))


        ct_img = ct_img.squeeze(3)
        ct_img = ct_img.squeeze(3)
        ct_img = ct_img.sum(2)


        ct_img = self.lrelu(self.fc0(ct_img))
        ct_img = self.lrelu(self.fc1(ct_img)+ct_img)
        ct_img = self.lrelu(self.fc2(ct_img)+ct_img)
        date = F.softplus(self.fc3(ct_img))
        live = F.sigmoid(self.fc4(ct_img))


        return date,live