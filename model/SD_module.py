import configparser
import time

import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw, ImageFont

from model.yolo import YOLO

from time import strftime

import colorsys


class ShipDetection:
    def __init__(self):

        # self.generate_config()

        self.config = self.read_config()

        model_pramas = dict(self.config['model'])
        self.model = YOLO(model_params=model_pramas)

    #    def set_confidence(self):

    #    def nms_iou(self):

    def set_confidence(self, value):
        self.model.set_confidence(value)

    def set_nms_iou(self, value):
        self.model.set_nms_iou(value)

    def set_input_shape(self, value):
        self.model.set_input_shape(value)
        self.config['model']['input_shape'] = '%d, %d' % (value[0], value[1])

    def generate_config(self):

        # 설정파일 만들기
        config = configparser.ConfigParser()

        # 설정파일 오브젝트 만들기
        config['system'] = {}
        config['system']['title'] = 'Ship Detection'
        config['system']['version'] = '0.0.0'
        config['system']['update'] = strftime('%Y-%m-%d %H:%M:%S')

        config['model'] = {}
        config['model']['weight_path'] = 'data/yolo4_voc_weights.pth'
        config['model']['classes_path'] = 'dataset/voc_names'
        config['model']['anchors_path'] = 'dataset/yolo_anchors.txt'
        # config['model']['anchors_box'] = '12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401'
        config['model'][
            'class_names'] = 'aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor'

        # config['model']['anchors_mask'] =
        # config['model']['input_shape'] = '416, 416'
        # config['model']['confidence'] = '0.5'
        # config['model']['nms_iou'] = '0.3'

        # 설정파일 저장
        with open('../config.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    def detect_image(self, image):

        image = Image.open(image)
        result = self.model.detect_image(image)
        result.show()

    def detect(self, image, show=False):

        #image = Image.open(image)
        result = self.model.detect(image)

        if show is True:
            self.display_image(image, result)

        return result

    def detect_image_cv2(self, image):

        image = cv2.imread(image)
        result = self.model.detect_image_cv2(image)
        cv2.imshow('cv2_result', result)
        cv2.waitKey()

    def detect_video(self, path, show=True, save_path=''):

        capture = cv2.VideoCapture(path)
        if save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(save_path, fourcc, 25.0, size)

        fps = 0.0
        while True:

            # t1 = time.time()
            ref, frame = capture.read()
            if ref is False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            frame = Image.fromarray(np.uint8(frame))  # frame

            # prediction
            t1 = time.time()
            frame = np.array(self.model.detect_image(frame))
            fps = (fps + (1. / (time.time() - t1))) / 2

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR

            # fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if show is True:
                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff

            if save_path != "":
                out.write(frame)

            # if c == 27:
            #    capture.release()
            #    break

        capture.release()
        cv2.destroyAllWindows()

        if save_path != "":
            out.release()

    def detect_fps(self, image_path=''):

        test_interval = 100
        image_path = '../data/street.jpg'

        img = Image.open(image_path)
        tact_time = self.model.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    def detect_dir(self, origin_path='img/', save_path='img_out/'):

        import os
        from tqdm import tqdm

        img_names = os.listdir(origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(origin_path, img_name)
                image = Image.open(image_path)
                r_image = self.model.detect_image(image)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                r_image.save(os.path.join(save_path, img_name))

    def display_image(self, image, target_list):

        input_shape = self.config['model']['input_shape']
        input_shape = np.array([int(x) for x in input_shape.split(',')])

        class_names = self.config['model']['class_names']
        class_names = [name.strip() for name in class_names.split(',')]
        num_classes = len(class_names)

        # 텍스트 설정
        font = ImageFont.truetype(font='data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        # 결과 그리기
        if target_list is not None:

            for target in target_list:

                top = max(0, int(target[2]))
                left = max(0, int(target[3]))
                bottom = max(0, int(target[4]))
                right = max(0, int(target[5]))

                label = '{} {:.2f}'.format(target[0], float(target[1]))

                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                print(label, top, left, bottom, right)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                class_index = class_names.index(target[0])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[class_index])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[class_index])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

        image.show()


if __name__ == "__main__":
    sd = ShipDetection()


    # 영상 입력(path) & 결과 전시
    sd.detect_image('data/street.jpg')
    #sd.detect('data/street.jpg')

    #sd.set_confidence(0.2)
    #sd.set_nms_iou(0.8)
    #sd.set_input_shape([416, 416])

    #path = 'data/test.avi'
    #sd.set_nms_iou(0.2)
    #path = '/media/add/ETC_DB/youtube/1.mp4'
    #sd.detect_video(path, show=True)

    #img_path = '/media/add/ETC_DB/smd/image/MVI_1619_VIS_frame_0110.jpg'
    #sd.detect_image(img_path)

    # 영상 입력(path) & 좌표 회신
    # result = sd.detect('data/street.jpg')

   # for target in result:
    #    print("%s %s : %d %d %d %d" % (target[0], target[1], target[2], target[3], target[4], target[5]))

    #sd.detect_image_cv2('data/street.jpg')

    # 동영상 입력(path) & 결과 전시
    #sd.detect_video('data/test.avi')

    # fps 측정
    #sd.detect_fps('data/street.jpg')

    # 카메라 입력
    #path = 'data/test.avi'

    """
    CAM_ID = 0
    capture = cv2.VideoCapture(CAM_ID)

    fps = 0
    while True:

        # t1 = time.time()
        ref, frame = capture.read()
        if ref is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
        frame = Image.fromarray(np.uint8(frame))  # frame

        # prediction
        t1 = time.time()
        frame = np.array(sd.model.detect_image(frame))
        fps = (fps + (1. / (time.time() - t1))) / 2

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR

        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
    
    """