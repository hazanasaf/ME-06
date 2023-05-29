from ultralytics import YOLO
import cv2
import pandas as pd
from os import system
import pygame
from datetime import datetime
import threading

WIDTH = 640
HEIGHT = 480
NORMAL_COLOR = (255, 0, 0) # blue
ALERT_COLOR = (0, 0, 255) # red
ALERT_REGION = [120, 60, 210, 180]
TIME_OF_LAST_DING = datetime.now()
MIN_DING_INTERVAL = 3 # seconds
resize_scale_factor = 1

def point_in_rect(xy,xyxy):
    return xyxy[0] <= xy[0] <= xyxy[2] and xyxy[1] <= xy[1] <= xyxy[3]


def rect_intersects_rect(xyxy1,xyxy2):
    for i in [0,2]:
        for j in [1,3]:
            if point_in_rect((xyxy1[i],xyxy1[j]),xyxy2) or point_in_rect((xyxy2[i], xyxy2[j]), xyxy1):
                return True
    return False

def read_warning_list():
    with open('warnings.txt', 'r') as warnings:
        lines = warnings.read().split('\n')
        return [ l.strip() for l in lines if len(l.strip()) > 0 ]

def rect_intersects_alert_region(xyxy):
    return rect_intersects_rect(xyxy, ALERT_REGION)

def run_model_on_img(yolo, img, results):
    tmp = yolo.predict(img)
    results.clear()
    results.extend(tmp)

def run_object_detection():
    yolo = YOLO("yolov8n.pt")

    cam = cv2.VideoCapture(0) #0=front-cam, 1=back-cam
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    pygame.mixer.init()
    TIME_OF_LAST_DING = datetime.now()
    
    warning_list = read_warning_list()
    
    results = []
    thread = None
    while True:
        ## press q or Esc to quit
        if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
            break

        ## read frames
        ret, img = cam.read()

        if img is None:
            continue

        # predict yolo
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=run_model_on_img, args=(yolo, img, results))
            thread.start()

        alert_list = []
        for r in results:
            for b in r.boxes:

                xyxy = b.xyxy.numpy()[0]
                xyxy = [int(a) for a in xyxy]
                c = ALERT_COLOR if rect_intersects_alert_region(xyxy) else NORMAL_COLOR
                class_name = yolo.names[int(b.cls.numpy()[0])]
                if c == ALERT_COLOR:
                    if class_name in warning_list:
                        alert_list.append(class_name)
                    #else:
                    #    alert_list.append('ding')
                
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), c, 2)
                cv2.putText(img, class_name, (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, c)

        ## display predictions
        cv2.rectangle(img, (ALERT_REGION[0], ALERT_REGION[1]), (ALERT_REGION[2], ALERT_REGION[3]), (0,255,0), 2)

        if len(alert_list) > 0:
            now = datetime.now()
            if (now - TIME_OF_LAST_DING).seconds >= MIN_DING_INTERVAL:
                alert_list = list(set(alert_list))
                TIME_OF_LAST_DING = now
                for alert in alert_list:
                    pygame.mixer.music.load(f"warnings/{alert}.mp3")
                    pygame.mixer.music.play()
        #    system("beep -f 440 -l 5")

        cv2.imshow("", img)

    ## close camera
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_object_detection()
    