import cv2
import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

Black = (0, 0, 0)
White = (255, 255, 255)
Pink = (255, 0, 255)

while True:
    ret, frame = cap.read()
    imageRGB = cv.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handsLms in results.multi_hand_landmarks:
            for iD, lm in enumerate(handsLms.landmark):
                # print((iD, lm))
                h, w, c = frame.shape
                cx, cy, = int(lm.x * w), int(lm.y * h)
                print(iD, cx, cy)
                # if iD ==4:
                cv.circle(frame, (cx, cy), 8, Pink, cv.FILLED)

            mpDraw.draw_landmarks(frame, handsLms, mpHands.HAND_CONNECTIONS)

    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)

    cv.putText(frame, str(time_string), (10, 40), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 3)
    # Output: Result: Fri Dec 28 08:44:04 2022

    cv.imshow("handtracking", frame)
    cv.waitKey(1)

