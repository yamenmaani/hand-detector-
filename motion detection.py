
import cv2
import time
import numpy as np
import mediapipe as mp
import keyboard


# 
# Wcam,Hcam = 640,480
Wcam,Hcam = 500,500
pTime = 0
mpHands = mp.solutions.mediapipe.python.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
# 
cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                # print(id, lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                print("point",id, cx,cy)
                if id == 0:
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
                if id == 4:
                    cv2.circle(img,(cx,cy),10,(255,255,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handlms, mpHands.HAND_CONNECTIONS)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f"fps: {int(fps)}", (40,20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,(255,0,255),2)
    cv2.imshow("cropped", img)
    cv2.waitKey(1)
    
    if keyboard.is_pressed('q'):
        break