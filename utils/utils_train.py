import torch
import torch.nn as nn
import math
import numpy as np
from functools import partial

import datetime
import os

import scipy.signal
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0, focal_loss=False, alpha=0.25, gamma=2):
        super(YOLOLoss, self).__init__()
        # -----------------------------------------------------------#
        #   13x13 feature's anchor : [142, 110],[192, 243],[459, 401]
        #   26x26 feature's anchor: [36, 75],[76, 55],[72, 146]
        #   52x52 feature's anchor: [12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.mask = mask
        self.label_smoothing = label_smoothing

        self.ignore_threshold = 0.5

        # -----------------------------------------------------------#
        # loss 가중치 확인 필요
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)

        # -----------------------------------------------------------#
        self.focal_loss = focal_loss
        self.focal_loss_ratio = 10
        self.alpha = alpha
        self.gamma = gamma

    # -----------------------------------------------------------#
    # t_min과 t_max 사이의 값으로 만듬
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    # -----------------------------------------------------------#
    # MSE(Mean square error) 계산
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    # -----------------------------------------------------------#
    # BCE(Binary cross entropy) 계산
    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_ciou(self, b1, b2):
        """
        input：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        transform：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   pbox left-up corner(mins) & right-bottom corner(maxs)
        # ----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   gt left-up corner & right-bottom corner
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   find all ious between b1, b2
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        # center distance between b1 and b2
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        # find smallest box that encloses the two boxes
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

        # calc diagonal distance
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v

        return ciou

    # ---------------------------------------------------#
    #   label smooth : 1
    #   1 => yture * (1-1) + 1/20 = 1/20
    #   0 => ytrue * (1-0) + 0/20 = yture
    #   BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1])
    #   y_ture = y_true[..., 5:][y_true[..., 4] == 1]
    # ---------------------------------------------------#
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None):
        # l : layer 번호
        # input : 8, 3*5*nc, 13, 13

        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 13x13 feature layer -> 1 point = 32x32 pixels
        # stride_h = stride_w = 32, 16, 8
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # scaled_anchors is relative to the feature layer
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # 8, 3, 13, 13, 5+nc
        prediction = input.view(bs, len(self.mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # sigmoid
        cx = torch.sigmoid(prediction[..., 0])
        cy = torch.sigmoid(prediction[..., 1])

        w = prediction[..., 2]
        h = prediction[..., 3]

        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 앵커 박스들 중에서 가장 유사한 박스를 찾음
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        # 탐지결과(3x13x13)와 정답지 간 비교를 통해 50% 이상 교차하는 것만 사용
        # 사용한 박스만 찾음.
        noobj_mask, pred_boxes = self.get_ignore(l, cx, cy, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        y_true = y_true.cuda()
        noobj_mask = noobj_mask.cuda()
        box_loss_scale = box_loss_scale.cuda()

        # -----------------------------------------------------------#
        #   reshape_y_true[...,2:3] and reshape_y_true[...,3:4]
        #   0-1 사이의 실제 상자의 너비와 높이를 나타냅니다.
        #   실제 상자가 클수록 비율이 작아지고 작은 상자의 비율이 커집니다.
        # -----------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale

        loss = 0
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        if n != 0:

            # loss_loc    = torch.mean((1 - ciou)[obj_mask] * box_loss_scale[obj_mask])

            ciou = self.box_ciou(pred_boxes, y_true[..., :4])

            loss_loc = torch.mean((1 - ciou)[obj_mask])
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))

            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        if self.focal_loss:
            ratio = torch.where(obj_mask, torch.ones_like(conf) * self.alpha,
                                torch.ones_like(conf) * (1 - self.alpha)) * torch.where(obj_mask,
                                                                                        torch.ones_like(conf) - conf,
                                                                                        conf) ** self.gamma
            loss_conf = torch.mean((self.BCELoss(conf, obj_mask.type_as(conf)) * ratio)[
                                       noobj_mask.bool() | obj_mask]) * self.focal_loss_ratio
        else:
            loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])

        loss += loss_conf * self.balance[l] * self.obj_ratio

        return loss

    def calculate_iou(self, _box_a, _box_b):

        # GT left top, right bottom
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2

        # pred left top, right bottom
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        # xyxy
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        A = box_a.size(0)
        B = box_b.size(0)

        # 교차 면적 계산
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]

        # 개별 면적 계산
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

        # 유니온 계산
        union = area_a + area_b - inter

        return inter / union

    def get_target(self, l, targets, anchors, in_h, in_w):

        bs = len(targets)

        # [8x3x13x13]
        noobj_mask = torch.ones(bs, len(self.mask[l]), in_h, in_w, requires_grad=False)

        # [8x3x13x13]
        box_loss_scale = torch.zeros(bs, len(self.mask[l]), in_h, in_w, requires_grad=False)

        # [8x3x13x13x(5+nc)]
        y_true = torch.zeros(bs, len(self.mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)

        for b in range(bs):
            if len(targets[b]) == 0:
                continue

            # 7(target), 5(box+label)
            target = torch.zeros_like(targets[b])

            # 해당 레이어로 스케일링 xywh
            target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w  # cx, w
            target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h  # cy, h
            target[:, 4] = targets[b][:, 4]
            target = target.cpu()

            # 실제 박스로 변환 : target x 4(0,0,w,h)
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((target.size(0), 2)), target[:, 2:4]), 1))

            # 앵거 박스로 변환 : 9 x 4 = 총박스 x (0,0,w,h)
            anchor_box = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))

            # 가장 크기가 비슷한 박스찾기
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_box), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.mask[l]:
                    print('%d not in mask layer %d' % (t, l))
                    continue

                # 해당 레이어의 앵커 인덱스
                k = self.mask[l].index(best_n)

                # GT box 그리드에서의 위치
                i = torch.floor(target[t, 0]).long()
                j = torch.floor(target[t, 1]).long()

                c = target[t, 4].long()

                noobj_mask[b, k, j, i] = 0

                y_true[b, k, j, i, 0] = target[t, 0]
                y_true[b, k, j, i, 1] = target[t, 1]
                y_true[b, k, j, i, 2] = target[t, 2]
                y_true[b, k, j, i, 3] = target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1

                #   xywh를 얻는 데 사용되는 척도
                #   큰 목표는 손실 가중치가 작고 작은 목표는 손실 가중치가 큽니다.
                box_loss_scale[b, k, j, i] = target[t, 2] * target[t, 3] / in_w / in_h

        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):

        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

        # 그리드 인덱스 생성 0,1,...,11
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 이전 상자의 너비와 높이 생성
        scaled_anchors_l = np.array(scaled_anchors)[self.mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 탐지결과 디코딩
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):

            # 3*13*13, 4
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) == 0:
                continue

            # 피처 레이어에서 포지티브 샘플의 중심점 계산
            target = torch.zeros_like(targets[b])
            target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            target = target[:, :4]

            # [target, 3x13x13]
            anch_ious = self.calculate_iou(target, pred_boxes_for_ignore)

            # [3x13x13] 가장 높은 오버랩을 가진 것만 다시 계산
            anch_ious_max, _ = torch.max(anch_ious, dim=0)
            anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])

            # 교차 영역이 threshold 보다 큰 것만 사용
            noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0

        return noobj_mask, pred_boxes


# ---------------------------------------------------#
# 모델 가중치 초기화
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def resume_weights(resume_path, optimizer, model_train):
    # load optimizer
    checkpoint = torch.load(resume_path)  # load checkpoint
    optimizer.load_state_dict(checkpoint['optimizer'])

    # load loss results
    loss = checkpoint['loss']

    # load epoch
    start_epoch = checkpoint['epoch'] + 1

    # load model
    model_train.load_state_dict(checkpoint['model'])
    print('Transferred %g/%g items' % (len(checkpoint['model']), len(model_train.state_dict())))

    del checkpoint

    return optimizer, loss, start_epoch, model_train


# ---------------------------------------------------#
# 가중치 불러오기
def load_weights(model, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path.endswith('pth'):
        # ---------------------------------------------------#
        # 백본만 가중치 적용
        backbone = model.backbone
        model_dict = backbone.state_dict()

        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_names = list(pretrained_dict.keys())

        for i, param in enumerate(model_dict):
            if model_dict[param].shape != pretrained_dict[pretrained_names[i]].shape:
                print('%d error', i)
                exit()

            print(i, ' ', model_dict[param].numel(), ' ', param, ' : ', model_dict[param].shape, '-----',
                  pretrained_names[i], pretrained_dict[pretrained_names[i]].shape)

            model_dict[param] = pretrained_dict[pretrained_names[i]]

        backbone.load_state_dict(model_dict)

    else:
        # ---------------------------------------------------#
        # 모델 전체 가중치 적용
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        """
        model_dict = model.state_dict()

        pretrained_dict = torch.load(model_path, map_location=device)

        # 1. filter out unnecessary keys
        cnt = 0
        for k, v in pretrained_dict.items():
            if np.shape(model_dict[k]) == np.shape(v):
                pretrained_dict = {k: v}
                cnt += 1

            if np.shape(model_dict[k]) != np.shape(v):
                print('error count : ', cnt)
                exit()

        print('count : ', cnt)
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        """


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class LossHistory:
    def __init__(self, log_dir, model, input_shape):

        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')

        self.log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        # self.train_loss = []
        # self.val_loss = []

        self.loss = {"train": [], "valid": []}

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

        self.min_train_loss = 9999.0
        self.min_valid_loss = 9999.0

    def load_loss(self, loss):

        self.loss = loss

        for idx, l in enumerate(loss["train"]):
            epoch = idx
            train_loss = loss["train"][idx]
            valid_loss = loss["valid"][idx]

            self.writer.add_scalar('train_loss', train_loss, epoch + 1)
            self.writer.add_scalar('valid_loss', valid_loss, epoch + 1)

    def append_loss(self, epoch, train_loss, valid_loss):
        self.loss["train"].append(train_loss)
        self.loss["valid"].append(valid_loss)

        with open(os.path.join(self.log_dir, "train_loss.txt"), 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "valid_loss.txt"), 'a') as f:
            f.write(str(valid_loss))
            f.write("\n")

        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('valid_loss', valid_loss, epoch)

        self.loss_plot()

        if train_loss < self.min_train_loss:
            self.min_train_loss = train_loss
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss

    def loss_plot(self):
        iters = range(len(self.loss["train"]))

        plt.figure()
        plt.plot(iters, self.loss["train"], 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.loss["valid"], 'coral', linewidth=2, label='valid loss')
        try:
            if len(self.loss["train"]) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


def create_optimizer(configs, model):
    """Create optimizer for training process
    Refer from https://github.com/ultralytics/yolov3/blob/e80cc2b80e3fd46395e8ec75f843960100927ff2/train.py#L94
    """
    if hasattr(model, 'module'):
        params_dict = dict(model.module.named_parameters())
    else:
        params_dict = dict(model.named_parameters())

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in params_dict.items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif ('conv' in k) and ('.weight' in k):
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if configs.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(pg0, lr=configs.lr, momentum=configs.momentum, nesterov=True)
    elif configs.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(pg0, lr=configs.lr)
    else:
        assert False, "Unknown optimizer type"

    optimizer.add_param_group({'params': pg1, 'weight_decay': configs.weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))

    return optimizer

"""
from torch.optim.lr_scheduler import LambdaLR


def create_lr_scheduler(optimizer, type):
    #Create learning rate scheduler for training process

    if type.lr_type == 'multi_step':
        def burnin_schedule(i):
            if i < configs.burn_in:
                factor = pow(i / configs.burn_in, 4)
            elif i < configs.steps[0]:
                factor = 1.0
            elif i < configs.steps[1]:
                factor = 0.1
            else:
                factor = 0.01
            return factor

        lr_scheduler = LambdaLR(optimizer, burnin_schedule)
    elif type.lr_type == 'cosin':
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / type.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
        # plot_lr_scheduler(optimizer, lr_scheduler, configs.num_epochs, save_dir=configs.logs_dir)
    else:
        raise ValueError

    return lr_scheduler
"""