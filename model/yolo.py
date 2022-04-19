import colorsys
import time

import torch.nn as nn
from PIL import ImageDraw, ImageFont

from model.model import YoloBody
from model.utils import *

import cv2

class YOLO(object):
    _defaults = {
        "weight_path": '',  # 사전학습 가중치 설정, '' 미사용
        "classes_path": 'dataset/voc_names',  # 클래스 종류 경로

        "anchors_path": 'dataset/yolo_anchors.txt',  # 앵커 박스의 크기(보통 수정 x)
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],  # 스케일에 맞는 앵커 박스 정의(보통 수정 x)

        "input_shape": [416, 416],  # 입력영상의 크기(32배수) : 416, 608, ...
        "confidence": 0.5,  # 예측 결과 값이 0.5 이상만 사용
        "nms_iou": 0.3,  # NMS에서 제거되는 기준

        "letterbox_image": False,  # 입력영상으로 변환 과정에서 기존 비율의 유지 여부(테스트 결과 False가 더 좋음)
        "cuda": True,  # CUDA 사용여부
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    # 초기화
    def __init__(self, **kwargs):

        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():

            if name == 'model_params':
                for key, val in value.items():
                    setattr(self, key, val)
            else:
                setattr(self, name, value)

        self.class_names = [name.strip() for name in self.class_names.split(',')]
        self.num_classes = len(self.class_names)

        self.anchors = np.array([float(x) for x in self.anchors.split(',')]).reshape(-1, 2)
        self.input_shape = np.array([int(x) for x in self.input_shape.split(',')])

        self.confidence = float(self.confidence)
        self.nms_iou = float(self.nms_iou)


        # 앵커의 크기 복호화
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        # 클래스마다 다른색상으로 표시
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.model = self.generate()

    # ---------------------------------------------------#
    # 모델 생성
    def generate(self):

        model = YoloBody(self.anchors_mask, self.num_classes)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict = model.state_dict()

        load = torch.load(self.weight_path, map_location=device)

        new = list(load.items())

        count = 0
        for key, value in state_dict.items():
            layer_name, weights = new[count]
            state_dict[key] = weights
            count += 1

        model.load_state_dict(state_dict)

        model = model.eval()
        print('{} model, anchors, and classes loaded.'.format(self.weight_path))

        if self.cuda:
            model = nn.DataParallel(model)
            model = model.cuda()

        return model

    def set_confidence(self, value):
        self.confidence = value

    def set_nms_iou(self, value):
        self.nms_iou = value

    def set_input_shape(self, value):
        self.input_shape = value

    # ---------------------------------------------------#
    # 이미지 탐지
    def detect_image(self, image):

        image_shape = np.array(np.shape(image)[0:2])
        #image = cvtColor(image)  # RGB
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 예측
            outputs = self.model(images)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_th=self.confidence,
                                                         nms_th=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        # 텍스트 설정
        font = ImageFont.truetype(font='data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # 결과 그리기
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


    def get_FPS(self, image, test_interval):

        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)  # RGB
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():

                outputs = self.model(images)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect(self, image):

        image_shape = np.array(np.shape(image)[0:2])
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 예측
            outputs = self.model(images)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_th=self.confidence,
                                                         nms_th=self.nms_iou)

            if results[0] is None:
                return None

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        target_list = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box

            target = [predicted_class, score[:6], top, left, bottom, right]
            target_list.append(target)

        return target_list