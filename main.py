import cv2
import numpy as np
import time
from tracker import Tracker
import torch
from cal_loss import get_bbox_classes

def get_video():
    # read the default videos
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_all = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("[INFO] FPS: {:.0f}fps".format(fps))
    print("[INFO] frame_all : {:.0f}".format(frame_all))
    print("[INFO] time_last: {:.1f}s".format(frame_all / fps))

    return cap


def get_groundtruth():
    with open("Truth.txt", "r") as f1:
        line = f1.readlines()[0].strip()
        x1, y1, x2, y2 = map(int, line.split(','))

    return x1, y1, x2, y2


def cal_average_pixel(error):
    global axis
    if error > 50:
        pass
    elif error > 40:
        axis[4] += 1
    elif error > 30:
        axis[3] += 1
    elif error > 20:
        axis[2] += 1
    elif error > 10:
        axis[1] += 1
    else:
        axis[0] += 1

def accumulate():
    global  axis
    # accumulate
    for i in range(1, 5):
        axis[i] += axis[i - 1]

    for i in range(5):
        axis[i] /= 273


axis = [0, 0, 0, 0, 0]


def main():
    i = 1
    acu_error = 0
    cap = get_video()
    x1, y1, x2, y2 = get_groundtruth()
    time_last = 0.0
    time_last = 0.0
    init_tracker = True

    while 1:
        ret, frame = cap.read()
        # Set to numpy array and to raw pixels
        if ret == False:
            break

        if (init_tracker):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255))

            # init
            hcf_tracker.init_frame(frame, [x1, y1, x2, y2])
            init_tracker = False

        else:
            # Refresh frame
            t1 = time.time()
            new_region = hcf_tracker.update(cur_frame=frame)
            t2 = time.time()
            duration = (1 - 0.1) * time_last + 0.1 * float(t2 - t1)  # exp mean
            time_last = duration
            print("%d: %.1f fps" % (i, 1.0 / duration), end=",")

            x1, y1, x2, y2 = new_region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thickness=3)
            #frame = cv2.resize(frame,(0,0), fx=0.5, fy=0.5)
            #cv2.imshow("HCF Tracker", frame)

        # write image
        # cv2.imwrite("res/%04d.jpg" % i, frame)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        '''  # loss calculate 
        with open("groundtruth.txt", "r") as f:
            line_all = f.readlines()
            cx, cy = line_all[i-1].strip().split(',')
            cx, cy = int(cx), int(cy)

            error = np.sqrt((center_x - cx)**2 + abs(center_y - cy)**2)
            acu_error += error
            cal_average_pixel(error)
            print(" error pixels: {:.1f}, {:.1f}".format(acu_error, error))
            i += 1
        '''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    accumulate()
    print(acu_error / 273.0)
    print(axis)


hcf_tracker = Tracker(lamda=0.0001)
main()

