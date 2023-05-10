"""
derictions:
FallDetection.py: Human fall detection through YoloPose key point detection
author: yuchuxiu
date:2023-5-4
"""
import os
import cv2
import time
import torch
import argparse
import numpy as np

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.plots import plot_skeleton_kpts
from utils.torch_utils import select_device, time_synchronized

# Support video file detection and image file detection
imagepath = r".\test\test.jpg"  # image folder for images or image-file path for one image.
videopath = r".\test\Falldetection.mp4"  # video file
video_format = ['.avi', '.mp4', '.mkv']
image_format = ['.jpg', '.png', '.jpeg']
if os.path.isdir(imagepath):
    files = os.listdir(imagepath)
    imagefiles = [os.path.join(imagepath, image) for image in files if image.endswith(tuple(image_format))]
else:
    imagefiles = [imagepath]


def FallDetection(weights, interval=1, imgsz=640, opt=None, save_video=True, half_precision=True, mode='video'):
    assert mode in ['video', 'image'], "mode must be video or image"
    # Load model
    device = select_device(opt.device, batch_size=1)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    model.model[-1].flip_test = False
    model.model[-1].flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    nc = 1  # number of classes，Here we only do key point detection on people

    # Test video file
    if mode == "video":
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter('output.mp4', fourcc, 25, (1280, 720))

        cap = cv2.VideoCapture(videopath)
        while (True):
            for _ in range(interval):

                _, org = cap.read()
                if org is None:
                    cv2.destroyAllWindows()
                    if save_video:
                        writer.release()
                    return None

            # resize & letterbox。Aspect Ratio Unchanged， resize to(640,640)
            t0 = time_synchronized()
            h0, w0 = org.shape[:2]
            ratio = imgsz / max(h0, w0)
            frame = cv2.resize(org, (int(w0 * ratio), int(h0 * ratio)))
            frame, _, pad = letterbox(frame, auto=False)
            # Convert
            frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
            frame = np.ascontiguousarray(frame)
            shapes = (h0, w0), ((ratio, ratio), pad)
            frame = torch.FloatTensor([frame])
            frame = frame.to(device, non_blocking=True)
            frame = frame.half() if half else frame.float()  # uint8 to fp16/32
            # frame = frame.astype(np.float32)
            frame /= 255.0  # 0 - 255 to 0.0 - 1.0
            frame_count = 0 # set a frame counter
            #print("data preprocessing time: ", time_synchronized() - t0)

            with torch.no_grad():
                t1 = time_synchronized()
                out, _  = model(frame, augment=False)  # inference
                out = non_max_suppression(out, conf_thres=0.1, iou_thres=0.6, labels=[], multi_label=True, agnostic=True, kpt_label=True, nc=nc)
            #print("inference & NMS time: ", time_synchronized() - t1)
            # statistics per image
            for si, pred in enumerate(out):
                pred[:, 5] = 0  # single class: person
                scale_coords(frame[si].shape[1:], pred[:, :4], shapes[0], ratio_pad=None,
                             kpt_label=False)  # native-space pred
                scale_coords(frame[si].shape[1:], pred[:, 6:], shapes[0], ratio_pad=None, kpt_label=True,
                                 step=3)  # native-space pred
                # print('*'*80)
                # print(pred.shape)
                # print(pred.cpu().numpy())  # A target has 57 outputs(target frame information(xyxy,conf,id) + (17 keypoints*(x,y,conf))



                # # According to the aspect ratio of the target box
                pred_numpy = pred.cpu().numpy()
                obj_num = pred_numpy.shape[0]
                for i in range(obj_num):
                    is_alarm = False  # Detect whether a person has fallen based on key points
                    obj_info = pred_numpy[i]
                    # Draw key points of the human body
                    plot_skeleton_kpts(org, obj_info[6:], steps=3)
                    if ((obj_info[3] - obj_info[1]) / (obj_info[2] - obj_info[0]) < 1):  # Aspect ratio less than 1
                        # Trigger condition 2, the angle between the centerline of the body and the vertical direction
                        obj_kpts = obj_info[6:]
                        face_x_lst, face_y_lst = [], []
                        for i in range(5):  # 5 key points of the face
                            if (obj_kpts[3 * i + 2] > 0.5):
                                face_x_lst.append(obj_kpts[3 * i])
                                face_y_lst.append(obj_kpts[3 * i + 1])
                        if len(face_x_lst) > 0:
                            face_x = np.mean(face_x_lst)
                            face_y = np.mean(face_y_lst)

                        waist_x_lst, waist_y_lst = [], []
                        for i in range(11, 13):  # 2 key points of the waist
                            if (obj_kpts[3 * i + 2] > 0.5):
                                waist_x_lst.append(obj_kpts[3 * i])
                                waist_y_lst.append(obj_kpts[3 * i + 1])
                        if len(waist_x_lst) > 0:
                            waist_x = np.mean(waist_x_lst)
                            waist_y = np.mean(waist_y_lst)


                        # If the body tilts more than 45°, it is judged as a fall

                        if (len(face_x_lst) * len(waist_x_lst) > 0) and (np.abs(waist_x - face_x) > np.abs(waist_y - face_y)):
                            '''
                            frame_count += 1
                        else:
                            frame_count =0
                        print ("frame_count: ", frame_count)

                        if frame_count >= 5:
                        '''
                            ##The test video is faster and does not need to set the counter
                            print('Person Falls Detected！！')
                            is_alarm = True
                    if is_alarm:
                        #The target draws a red detection frame and gives an alarm prompt
                        cv2.rectangle(org, (obj_info[0], obj_info[1]), (obj_info[2], obj_info[3]), color=(0, 0, 255),
                                      thickness=2)
                        cv2.putText(org, "Fall Down!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(org, (obj_info[0], obj_info[1]), (obj_info[2], obj_info[3]), color=(208, 224, 64),
                                      thickness=2)

                # Drawing to see if the result is correct
                cv2.imshow("image", org)
                cv2.waitKey(10)

                # Save result video
                if save_video:
                    writer.write(org)

        # Test image
    elif mode == "image":
        for imagefile in imagefiles:
            org = cv2.imread(imagefile)
            h0, w0 = org.shape[:2]
            ratio = imgsz / max(h0, w0)
            frame = cv2.resize(org, (int(w0 * ratio), int(h0 * ratio)))
            frame, _, pad = letterbox(frame, auto=False)
            # Convert
            frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
            frame = np.ascontiguousarray(frame)
            shapes = (h0, w0), ((ratio, ratio), pad)

            frame = torch.FloatTensor([frame])
            frame = frame.to(device, non_blocking=True)
            frame = frame.half() if half else frame.float()  # uint8 to fp16/32
            # frame = frame.astype(np.float32)
            frame /= 255.0  # 0 - 255 to 0.0 - 1.0

            with torch.no_grad():
                out, _  = model(frame, augment=False)  # inference
                out = non_max_suppression(out, conf_thres=0.1, iou_thres=0.6, labels=[], multi_label=True, agnostic=True, kpt_label=True, nc=nc)
            # statistics per image
            for si, pred in enumerate(out):
                pred[:, 5] = 0  # single class: person
                scale_coords(frame[si].shape[1:], pred[:, :4], shapes[0], ratio_pad=None,
                             kpt_label=False)  # native-space pred
                scale_coords(frame[si].shape[1:], pred[:, 6:], shapes[0], ratio_pad=None, kpt_label=True,
                                 step=3)  # native-space pred
                pred_numpy = pred.cpu().numpy()
                # print(pred_numpy)
                obj_num = pred_numpy.shape[0]
                for i in range(obj_num):
                    obj_info = pred_numpy[i]
                    # Draw key points of the human body
                    plot_skeleton_kpts(org, obj_info[6:], steps=3)
                    cv2.rectangle(org, (int(obj_info[0]), int(obj_info[1])), (int(obj_info[2]), int(obj_info[3])), color=(208, 224, 64),
                                  thickness=2)
                    cv2.rectangle(org, (int(obj_info[0]), int(obj_info[1])-45), (int(obj_info[0])+100, int(obj_info[1])), color=(255,250,87),thickness=-1, lineType=cv2.LINE_AA)  # filled
                    cv2.putText(org, str(round(obj_info[4],1)), (int(obj_info[0])+10, int(obj_info[1])-5), 0, 1.5, (225, 255, 255), thickness=2,
                                lineType=cv2.LINE_AA)
                cv2.imshow("image", org)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FD.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print(opt)
    FallDetection(opt.weights, opt=opt)

