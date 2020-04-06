from conv_cf import HCF
import torch
import numpy as np
import cv2
from models.vgg import vgg19_bn
import torchvision.transforms as T

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = vgg19_bn(pretrained=True, progress=True)
model = model.to(device)
conv3 = torch.zeros((1, 256, 16, 16), device=device)
conv4 = torch.zeros((1, 512, 8, 8), device=device)
conv5 = torch.zeros((1, 512, 4, 4), device=device)


def get_conv(model, extracted_roi):
    with torch.no_grad():
        global conv3, conv4, conv5  # can be remove
        for i in range(53):
            extracted_roi = model.features[i](extracted_roi)
            if i == 26:
                conv3 = extracted_roi
            if i == 39:
                conv4 = extracted_roi
        conv5 = extracted_roi

    return conv3, conv4, conv5


def get_border_roi(x1, y1, x2, y2, frame):
    h, w = frame.shape[0], frame.shape[1]
    d_x1, d_x2, d_y1, d_y2 = [0]*4
    # in_roi
    if x1 < 0:
        # x2 -= x1
        x1 = 0
        d_x1 = -x1
    if y1 < 0:
        # y2 -= y1
        y1 = 0
        d_y1 = -y1
    if x2 > w:
        # x1 -= w
        x2 = w
        d_x2 = x2 - w
    if y2 > h:
        # y1 -= h
        y2 = h
        d_y2 = y2 - h
    in_roi = frame[y1:y2+1, x1:x2+1, :]

    bordertype = cv2.BORDER_REFLECT
    final_roi = cv2.copyMakeBorder(in_roi, d_y1, d_y2, d_x1, d_x2, bordertype)

    return final_roi


class Tracker(object):

    def __init__(self, lamda):
        super(Tracker,self).__init__()
        self.hcf_ori = HCF(lamda=lamda)
        self.hcf_3 = HCF(lamda=lamda)
        self.hcf_4 = HCF(lamda=lamda)
        self.hcf_5 = HCF(lamda=lamda)

        self.pad = 1.21
        self.gamma = 2.0
        self.tmpl = [256]*2
        self.region = [0, 0, 0, 0]
        self.region_size = [0, 0]
        self.frame = 0.0

        self.x = 0.0

    def init_frame(self, frame, region):
        self.region = region
        self.region_size = [region[2]-region[0] + 1, region[3]-region[1] + 1]
        self.frame = frame

        self.x, nothing = self.get_featuremap(frame, *region, scale=1.0, pad=self.pad)
        conv3, conv4, conv5 = get_conv(model=model, extracted_roi=self.x)
        self.hcf_ori.init_conv(self.x, D=3, layers=6)
        self.hcf_3.init_conv(conv3, D=256, layers=3)
        self.hcf_4.init_conv(conv4, D=512, layers=4)
        self.hcf_5.init_conv(conv5, D=512, layers=5)

    def get_featuremap(self, frame, new_x1, new_y1, new_x2, new_y2, scale, pad):
        """Important slice operation"""
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pad_scale_roi = get_border_roi(new_x1, new_x2, new_y1, new_y2, frame)

        if scale != 1.0:
            half_w = (new_x2 - new_x1 + 1) / 2 * (scale - 1)
            half_h = (new_y2 - new_y1 + 1) / 2 * (scale - 1)

            new_x1 = int(np.ceil(new_x1 - half_w))
            new_y1 = int(np.ceil(new_y1 - half_h))
            new_x2 = int(np.floor(new_x2 + half_w))
            new_y2 = int(np.floor(new_y2 + half_h))

            """Important slice operation"""
            #pad_scale_roi = frame[max(new_y1, 0):new_y2+1, max(new_x1, 0):new_x2+1]  # frame follow H x W sequenc
            pad_scale_roi = get_border_roi(new_x1, new_y1, new_x2, new_y2, frame)

        if pad != 0:
            half_w = (new_x2 - new_x1 + 1) / 2 * (self.pad - 1)
            half_h = (new_y2 - new_y1 + 1) / 2 * (self.pad - 1)

            tmp_x1 = int(np.ceil(new_x1 - half_w))
            tmp_y1 = int(np.ceil(new_y1 - half_h))
            tmp_x2 = int(np.floor(new_x2 + half_w))
            tmp_y2 = int(np.floor(new_y2 + half_h))

            """Important slice operation"""
            #pad_scale_roi = frame[max(tmp_y1, 0):tmp_y2+1, max(tmp_x1, 0):tmp_x2+1]  # frame fol
            pad_scale_roi = get_border_roi(tmp_x1, tmp_y1, tmp_x2, tmp_y2, frame)

        fix_roi = cv2.resize(pad_scale_roi, dsize=(self.tmpl[1], self.tmpl[1]))  # dsize follow W x H (64x64)
        fix_roi = np.asarray(fix_roi, dtype=np.float32)  # np.array
        fix_roi = fix_roi / 255.0 - 0.5         # normalize
        fix_roi = np.power(fix_roi, self.gamma)  # gamma correct

        fix_roi = np.transpose(fix_roi, axes=(2, 0, 1))   # zip to Torch.Tensor
        fix_roi = torch.as_tensor(fix_roi, dtype=torch.float, device=device)  # zip to Torch.Tensor.CUDA
        #fix_roi = T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(fix_roi)
        #fix_roi = T.Normalize((0., 0., 0.),(0.229, 0.224, 0.225))(fix_roi)
        fix_roi = fix_roi[None, :, :, :]

        return fix_roi, [new_x1, new_y1, new_x2, new_y2]

    def get_peak(self, response , l):
        pos = int(np.argmax(response) + 1)
        pos_y = int(np.ceil(pos/l))   # row
        pos_x = int(pos % l)   # line
        if pos_x == 0:
            pos_x = l
        peak = response[pos_y - 1, pos_x - 1]

        return pos_y, pos_x, peak

    def update(self, cur_frame):
        # Decode axis and set img_size
        x1 = self.region[0]
        y1 = self.region[1]
        x2 = self.region[2]
        y2 = self.region[3]

        z_roi, [new_x1, new_y1, new_x2, new_y2] = \
            self.get_featuremap(cur_frame, x1, y1, x2, y2, scale=1.0, pad=self.pad)
        conv3, conv4, conv5 = get_conv(model, extracted_roi=z_roi)


        N = 64
        response_ori = self.hcf_ori.get_response_map(cur_conv=z_roi)
        # response_3 = self.hcf_3.get_response_map(cur_conv=conv3)
        response_3 = cv2.resize(self.hcf_3.get_response_map(cur_conv=conv3), dsize=(N, N))
        response_4 = cv2.resize(self.hcf_4.get_response_map(cur_conv=conv4), dsize=(N, N))
        response_5 = cv2.resize(self.hcf_5.get_response_map(cur_conv=conv5), dsize=(N, N))

        final_response = 1.*response_3 + 0.5 * response_4 + 0.2 * response_5

        pos_y, pos_x, peak = self.get_peak(final_response, N)

        center = N / 2 + 0.5
        float_y = (pos_y - 0.5 - center) / N * (self.region_size[0])
        float_x = (pos_x - 0.5 - center) / N * (self.region_size[1])
        '''
        
        pos_y, pos_x, peak = [0]*6, [0]*6, [0]*6
        N = 256
        response_ori = self.hcf_ori.get_response_map(cur_conv=z_roi)
        response_3 = self.hcf_3.get_response_map(cur_conv=conv3)
        response_4 = self.hcf_4.get_response_map(cur_conv=conv4)
        response_5 = self.hcf_5.get_response_map(cur_conv=conv5)

        pos_y[5], pos_x[5], peak[5] = self.get_peak(response_5, l=8)
        extract_ori_4 = response_4[2*(pos_y[5]-1):2*(pos_y[5]), 2*(pos_x[5]-1):2*(pos_x[5])]
        pos_y[2], pos_x[2], peak[2] = self.get_peak(extract_ori_4, l=2)
        pos_y[4] = (pos_y[5]-1)*2 + pos_y[2]
        pos_x[4] = (pos_x[5]-1)*2 + pos_x[2]

        extract_ori_3 = response_3[2*(pos_y[4]-1):2*(pos_y[4]), 2*(pos_x[4]-1):2*(pos_x[4])]
        pos_y[2], pos_x[2], peak[2] = self.get_peak(extract_ori_3, l=2)
        pos_y[3] = (pos_y[4]-1)*2 + pos_y[2]
        pos_x[3] = (pos_x[4]-1)*2 + pos_x[2]

        center = N / 2 + 0.5
        float_y = (pos_y[3] - 0.5 - center) / N * (self.region_size[0])
        float_x = (pos_x[3] - 0.5 - center) / N * (self.region_size[1])
        '''
        y1 += float_y
        y2 += float_y
        x1 += float_x
        x2 += float_x
        f = np.floor
        self.region = list(map(int, [f(x1), f(y1), f(x2), f(y2)]))

        return self.region




