from ultralytics import YOLO
import cv2
import pandas as pd
from os import system

WIDTH = 1000
HEIGHT = 850
NORMAL_COLOR = (255, 0, 0) # blue
ALERT_COLOR = (0, 0, 255) # red
ALERT_REGION = [int(WIDTH*4/10),140,int(WIDTH*6/10),460]

def point_in_rect(xy,xyxy):
    return xyxy[0] <= xy[0] <= xyxy[2] and xyxy[1] <= xy[1] <= xyxy[3]


def rect_intersects_rect(xyxy1,xyxy2):
    for i in [0,2]:
        for j in [1,3]:
            if point_in_rect((xyxy1[i],xyxy1[j]),xyxy2) or point_in_rect((xyxy2[i], xyxy2[j]), xyxy1):
                return True
    return False


def rect_intersects_alert_region(xyxy):
    return rect_intersects_rect(xyxy, ALERT_REGION)


yolo = YOLO("yolov8n.pt")

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


while True:
    ## press q or Esc to quit
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

    ## read frames
    ret, img = cam.read()
    if img is None:
        continue
    ## predict yolo
    results = yolo.predict(img)

    has_alert = False
    for r in results:
        for b in r.boxes:

            xyxy = b.xyxy.numpy()[0]
            xyxy = [int(a) for a in xyxy]
            c = ALERT_COLOR if rect_intersects_alert_region(xyxy) else NORMAL_COLOR
            if c == ALERT_COLOR:
                has_alert = True
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), c, 2)
            cv2.putText(img, yolo.names[int(b.cls.numpy()[0])], (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, c)

    ## display predictions
    cv2.rectangle(img, (ALERT_REGION[0], ALERT_REGION[1]), (ALERT_REGION[2], ALERT_REGION[3]), (0,255,0), 2)

    if has_alert:
        system("beep -f 440 -l 5")

    cv2.imshow("", img)

## close camera
cam.release()
cv2.destroyAllWindows()


# hello world
