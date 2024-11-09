# importing the necessary libraries
import cv2
import numpy as np
import sys

# Creating a VideoCapture object to read the video
if len(sys.argv) == 1:
    filename = input("Video filename: ")
else:
    filename = sys.argv[1]
cap = cv2.VideoCapture(filename)

# Loop until the end of the video
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # using cv2.Gaussianblur() method to blur the video

    # (5, 5) is the kernel size for blurring.
    gaussianblur = cv2.GaussianBlur(frame, (5, 5), 0) 
    cv2.imshow('gblur', gaussianblur)

    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# release the video capture object
cap.release()

# Closes all the windows currently opened.
cv2.destroyAllWindows()

