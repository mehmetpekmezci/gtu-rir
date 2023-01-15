import cv2
import imutils
import time
import numpy as np
device="/dev/video2"    


'''
Problem 1 : Different applications use different scales for HSV. For example gimp uses H = 0-360, S = 0-100 and V = 0-100. But OpenCV uses H: 0-179, S: 0-255, V: 0-255. Here i got a hue value of 22 in gimp. So I took half of it, 11, and defined range for that. ie (5,50,50) - (15,255,255).

Problem 2: And also, OpenCV uses BGR format, not RGB. So change your code which converts RGB to HSV as follows:

cv.CvtColor(frame, frameHSV, cv.CV_BGR2HSV)

'''

low_black = np.array([0, 0, 0])
high_black = np.array([350,55,100])

#ORANGE_MIN = cv.Scalar(18, 40, 90)
#ORANGE_MAX = cv.Scalar(27, 255, 255)
ORANGE_MIN = np.array([5, 50, 50],np.uint8)
ORANGE_MAX = np.array([15, 255, 255],np.uint8)
low_blue = np.array([94, 80, 2])
high_blue = np.array([126, 255, 255])
low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])
low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])
low_every_except_white = np.array([0, 42, 0])
high_every_except_white = np.array([179, 255, 255])
#vs = cv2.VideoCapture("input.mp4")
vs = cv2.VideoCapture(device)
time.sleep(2.0)

while True:
    _, frame = vs.read()

    if frame is None:
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    width, height = frame.shape[:2]
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_black, high_black)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # To see the centroid clearly
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 5)
            cv2.imwrite("circled_frame.png", cv2.resize(frame, (int(height / 2), int(width / 2))))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()


