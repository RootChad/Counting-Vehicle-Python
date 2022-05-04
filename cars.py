import cv2
import numpy as np
object_cascade=cv2.CascadeClassifier("./cars.xml")
cap=cv2.VideoCapture("./India.mp4")

while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    object=object_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=6)

    for x, y, w, h in object:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # cv2.imshow('video', frame)
    cv2.imshow('Frame', cv2.resize(frame, (800, 600)))
    cv2.imshow('Mask', gray)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()