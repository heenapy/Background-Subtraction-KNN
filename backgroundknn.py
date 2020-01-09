# ____________________Background Subtraction KNN____________________________
import numpy as np
import cv2

cap = cv2.VideoCapture('/home/paython/Videos/4K Video Downloader/Pogba Penalty, 10 seconds walking.mp4')
kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernal)
    cv2.imshow('frame',fgmask)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()