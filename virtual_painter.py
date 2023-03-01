import cv2

from Hands_Detection_Module import *
import numpy as np
import os

designPath = "design"
imgSrc = os.listdir(designPath)
overlayList = [cv2.imread(f'{designPath}/{path}') for path in imgSrc]
header = overlayList[0]
fingers = []
colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0)]
eraserThickness = 50
brushThickness = 15
xp = 0
yp = 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    with mp_hands.Hands() as hands:
        success, img = cap.read()
        # 1. Import the image
        img = cv2.flip(img, 1)  # flipping the image
        img[0:125, 0:1280] = header  # Setting the header image

        # 2. Find hand landmarks
        img, results = mediapipe_detection(img, hands)
        drawing_utilities(img, results)
        lmList = detect_fingers_positions(img, results)

        # 3. Check which fingers are up
        if len(lmList) > 0:
            fingers = fingersUp(lmList)
            x1, y1 = lmList[8]
            x2, y2 = lmList[12]
        # 4. If selection mode - Two fingers are up
        try:
            if fingers[1] and fingers[2]:
                if y1 < 120:
                    if 250 < x1 < 400:
                        header = overlayList[0]
                        drawColor = colors[0]
                    elif 500 < x1 < 700:
                        header = overlayList[1]
                        drawColor = colors[1]
                    elif 830 < x1 < 950:
                        header = overlayList[2]
                        drawColor = colors[2]
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = colors[3]
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
                xp = 0
                yp = 0
        except:
            pass
        # 5. If drawing mode - Index finger is up
        try:
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                if xp == yp == 0:
                    xp, yp = x1, y1
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1
                imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
                imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, imgInv)
                img = cv2.bitwise_or(img, imgCanvas)
        except:
            pass
        cv2.imshow("Live Feed", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
