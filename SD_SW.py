from model.SD_module import ShipDetection

import enum

import cv2
import time
import numpy as np

from PIL import Image


class Command(enum.Enum):
    VIDEO_DETECT = enum.auto()
    CAMERA_DETECT = enum.auto()
    IMAGE_DETECT = enum.auto()
    PARAM_SET = enum.auto()


if __name__ == "__main__":

    cmd = Command.IMAGE_DETECT
    sd = ShipDetection()

    stream = False
    if cmd == Command.VIDEO_DETECT:
        video_path = 'data/test.avi'
        stream = True
    elif cmd == Command.CAMERA_DETECT:
        video_path = 0
        stream = True
    elif cmd == Command.IMAGE_DETECT:
        image_path = '../USV_SD_project/data/street.jpg'
        sd.detect_image(image_path)
    elif cmd == Command.PARAM_SET:
        sd.set_input_shape([416, 416])
        sd.set_confidence(0.2)
        sd.set_nms_iou(0.8)

    show = True
    if stream is True:

        capture = cv2.VideoCapture(video_path)

        fps = 0.0
        count = 0
        while True:

            # t1 = time.time()
            ref, frame = capture.read()
            if ref is False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            frame = Image.fromarray(np.uint8(frame))  # frame

            count += 1
            if show is True:

                # prediction
                #t1 = time.time()
                #frame = np.array(sd.model.detect_image(frame))
                #fps = (fps + (1. / (time.time() - t1))) / 2

                results = sd.model.detect(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff

                if c == 27 or c == ord('q'):
                    capture.release()
                    break

                continue

            # prediction
            result = sd.detect(frame)

            if result is not None:
                for target in result:
                    print("frame %d : %s %s : %d %d %d %d" % (count, target[0], target[1], target[2], target[3], target[4], target[5]))

