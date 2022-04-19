import torch
from torchvision.ops import nms

import numpy as np
from PIL import Image


# ---------------------------------------------------------#
class DecodeBox:
    def __init__(self, anchors, num_classes, input_shape, anchors_mask):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # prob of each class
        self.input_shape = input_shape  # input image size

        # -----------------------------------------------------------#
        #   52x52 feature layer(P3) anchor : [12, 16],[19, 36],[40, 28] for small objects
        #   26x26 feature layer(P4) anchor : [36, 75],[76, 55],[72, 146]
        #   13x13 feature layer(P5) anchor : [142, 110],[192, 243],[459, 401] for big objects
        # -----------------------------------------------------------#
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []

        # 3-scale 따른 박스값 보정
        # input : bs, 75(3x25), 13, 13
        for i, input in enumerate(inputs):

            bs = input.size(0)
            input_height = input.size(2)  # 그리드 크기 w : 13
            input_width = input.size(3)  # 그리드 크기 h :13

            # stride(step) 값 계산 : 32 -> 16 -> 8
            stride_h = self.input_shape[0] / input_height  # 414/13 = 32
            stride_w = self.input_shape[1] / input_width

            # 앵커의 크기를 그리드에서의 크기로 변환
            scaled_anchors = []
            for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]:
                scaled_anchors += [(anchor_width / stride_w, anchor_height / stride_h)]

            # 예측값의 형태를 변환 : 1, 75, 13, 13 => 1, 3, 13, 13, 25
            prediction = input.view(bs, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width)
            prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()

            # -----------------------------------------------------------------#
            # 박스의 중심 좌표 추출
            bbox_cx = torch.sigmoid(prediction[..., 0])
            bbox_cy = torch.sigmoid(prediction[..., 1])

            # 박스의 높이, 너비 추출
            bbox_w = prediction[..., 2]
            bbox_h = prediction[..., 3]

            # 물체 존재(confidence) 추출
            conf = torch.sigmoid(prediction[..., 4])

            # 클래스 별 확률값 추출
            pred_cls = torch.sigmoid(prediction[..., 5:])

            # -----------------------------------------------------------------#
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor

            # 그리드 좌표값 생성 : 1, 3, 13, 13
            grid_x = torch.linspace(0, input_width - 1, input_width) \
                .repeat(input_height, 1) \
                .repeat(bs * len(self.anchors_mask[i]), 1, 1) \
                .view(bbox_cx.shape).type(FloatTensor)

            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                bs * len(self.anchors_mask[i]), 1, 1).view(bbox_cy.shape).type(FloatTensor)

            # 한 셀당 3개의 박스가 매칭 되도록 생성
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, input_height * input_width).view(bbox_w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, input_height * input_width).view(bbox_h.shape)

            # -----------------------------------------------------------------#
            # 예측 박스 정보를 읽기(decoding)
            pred_boxes = FloatTensor(prediction[..., :4].shape)

            # 원영상에서의 중심 위치 변경
            pred_boxes[..., 0] = bbox_cx.data + grid_x
            pred_boxes[..., 1] = bbox_cy.data + grid_y

            # 원영상에서의 박스 크기로 변경 : e^w + w, e^h + h
            pred_boxes[..., 2] = torch.exp(bbox_w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(bbox_h.data) * anchor_h

            # pred_boxes[..., 0] = x.data*2. - 0.5+grid_x
            # pred_boxes[..., 1]  = (y.data*2)**2 * grid_y

            # io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            # io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            # io[..., :4] *= self.stride

            # 그리드 스케일 값
            scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)

            a = pred_boxes.view(bs, -1, 4) / scale
            b = conf.view(bs, -1, 1)
            c = pred_cls.view(bs, -1, self.num_classes)

            # 결과 형태 변경 : 1, 박스(13*13*3), 좌표(4)
            output = torch.cat((a, b, c), -1)
            outputs.append(output.data)

        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):

        # 편리하게 연산을 위해 y 앞으로
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # 박스 정보 수정
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        # 박스 좌상단, 우하단
        box_LT = box_yx - (box_hw / 2.)
        box_RD = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_LT[..., 0:1], box_LT[..., 1:2], box_RD[..., 0:1], box_RD[..., 1:2]], axis=-1)

        # 영상에서 좌표로 변경
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        return boxes

    def non_max_suppression(self, preds, n_c, input_shape, image_shape, letterbox_image, conf_th=0.5, nms_th=0.4):

        # ----------------------------------------------------------#
        # xywh -> xyxy
        box_corner = preds.new(preds.shape)
        box_corner[:, :, 0] = preds[:, :, 0] - preds[:, :, 2] / 2
        box_corner[:, :, 1] = preds[:, :, 1] - preds[:, :, 3] / 2
        box_corner[:, :, 2] = preds[:, :, 0] + preds[:, :, 2] / 2
        box_corner[:, :, 3] = preds[:, :, 1] + preds[:, :, 3] / 2
        preds[:, :, :4] = box_corner[:, :, :4]

        # ----------------------------------------------------------#
        #
        output = [None for _ in range(len(preds))]
        for i, pred in enumerate(preds):

            #   가장 높은 클래스 확률과 인덱스
            class_conf, class_pred = torch.max(pred[:, 5:5 + n_c], 1, keepdim=True)  # xyxy, c, cls

            #  예측값의 신뢰도를 기준으로 필터링 : pred * conf
            conf_mask = (pred[:, 4] * class_conf[:, 0] >= conf_th).squeeze()

            pred = pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not pred.size(0):
                continue

            # 탐지 결과 : xyxy, obj_conf, class_conf, class_pred
            detections = torch.cat((pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # 탐지된 클래스 가져오기
            unique_labels = detections[:, -1].unique()

            # 동일 클래스부터 중복 계산
            for c in unique_labels:
                # 영상에서 탐지된 클래스
                detections_class = detections[detections[:, -1] == c]

                # torchvision 함수 사용
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_th
                )
                max_detections = detections_class[keep]

                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                # output[i] = output[i].numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return output


# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# ---------------------------------------------------#
def preprocess_input(image):
    image /= 255.0
    return image


import configparser


def read_config():
    # 설정파일 읽기
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')

    # 설정파일의 색션 확인
    # config.sections())
    ver = config['system']['version']
    print('config.ini file loaded(ver. %s)' % ver)

    return config
