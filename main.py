__author__ = 'David Manouchehri (david@davidmanouchehri.com)'
__license__ = 'MIT'
'''
I am using the following content as input for my experiment, if you use another you may have varying success.

SHA512 (v30.MPG) = 20b65b62cf523b15a157e563e90105e68e3f9b8df64afea5ce3323ad4b69e76bff56a738bc484ca30adee6c01cd0d5b3cbfb5243a35f8e87717159a016d71f01
SHA512 (v40.MPG) = bed9a9af5eae471affd4d3c44c5ec56e7330c0499404f7cdb083fa01a61894336bb2cdfd93ca6b1d1ee2bcac9d8bef12675d0cb957b6c2f6feedc09d5de627be
SHA512 (v50.MPG) = 46194b6fbe2e03bea56ed999e0106fd713930799834fc1a252fcfda036bc99751943181969c3d15b61480bfa8046c3810c649687025ba4d896fc994c7af5235e

From: http://www.engr.mun.ca/~migara/eng1040/Labs/src/Mechanical/Videos.zip

All content from Memorial University is strictly their own property, please speak to them regarding licensing. None of
their content has been included in this project to avoid any question of Intellectual Property.
'''
'''
Resources used include:
http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html
By Alexander Mordvintsev & Abid K.

http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
By the opencv dev team.
'''

import cv2  # I'm using OpenCV 3 with Python 3 support compiled in.
import numpy as np
import argparse  # Needed to accept the file names
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="the filename of the video (MPG only)")
args = parser.parse_args()


# from multiprocessing import Process, Queue  # OpenCV has multithreading itself, still other tricks can be applied.
# I can be a bit lazier when it comes to my OpenCV programming, because Python will be executing it in parallel anyway.

print('Written by ' + __author__ + '. All content is under the ' + __license__ + ' license.')

print('Reading: ' + args.filename)

cap = cv2.VideoCapture(args.filename)  # Load the file

# Grab the first frame
ret, frame = cap.read()

# Not exactly sure what values mean what
c, r, w, h = 325, 200, 50, 100  # QUESTION: This numbers seem to jump around wildly for position, why?
track_window = (c, r, w, h)


# Create a region of interest
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 0, 0)

lower = np.array([110, 0, 0])
upper = np.array([130, 255, 255])
# lower = np.uint8([[[150, 100, 200]]])
# upper = np.uint8([[[255, 255, 255]]])
fgbg = cv2.createBackgroundSubtractorMOG2()


while cap.isOpened():  # Bit redundant
    ret, frame = cap.read()  # I think this might skip the first frame?

    # Only continue if the frame can be read
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, gray = cv2.threshold(gray, 127, 255, 0)
        #gray2 = gray.copy()
        #mask2 = np.zeros(gray.shape, np.uint8)

        # Apply meanshift to find the new location
        ret, track_window = cv2.meanShift(gray, track_window, term_criteria)

        # Draw the results
        x, y, w, h = track_window
        img2 = cv2.rectangle(gray, (x, y), (x+w, y+h), 255, 2)
        #mask = cv2.inRange(hsv, lower, upper)
        fgmask = fgbg.apply(frame)
        #res = cv2.bitwise_and(frame, frame, mask=mask2)

        cv2.imshow('img2', gray)
    else:
        print('End of file, no more frames could be read.')
        break

    # cv2.imshow('frame', frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# If other parts of the program needed cap, then releasing would matter
cap.release()

# Burn everything!
cv2.destroyAllWindows()