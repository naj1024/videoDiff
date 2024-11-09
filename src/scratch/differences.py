import sys
import cv2
import numpy as np
from collections import deque

def difference(array1: np.ndarray, array2:np.ndarray, alpha: float, threshold: int, ghost: bool):
    # blend the two images together, they are time shifted N frames
    beta  = 1.0 - alpha   # 2nd image %
    gamma = 30.0
    difference = cv2.addWeighted(array2, alpha, array1, beta, gamma)

    # add back ghost original image
    if ghost:
        alpha2 = 0.8
        beta2 = 1 - alpha2
        difference = cv2.addWeighted(difference, alpha2, array1, beta2, gamma)

    # print(np.min(difference), np.max(difference), np.median(difference), np.mean(difference), np.std(difference))

    # threshold
    difference = np.where(difference >= threshold, difference, 0)

    return difference


# Creating a VideoCapture object to read the video
# if len(sys.argv) == 1:
#     filename = input("Video filename: ")
# else:
#     filename = sys.argv[1]
filename = "videos/lizard.mp4"
cap = cv2.VideoCapture(filename)

gray_scale = False           # gray scale the original frame source
reduce_resolution = False   # resize the frames
ghost_frame = True          # add back the original as a ghost
length_of_delay_buffer = 50 # maximum delay
delay = 40                  # frame delay for difference against
alpha = 0.5                 # % of delayed frame to subtract on difference
threshold = 1               # value below which things are set to black

frame_buffer = deque(maxlen = length_of_delay_buffer)
previous_frame = None

# Loop until the end of the video
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # cut down size, if required
    if reduce_resolution:
        frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)

    if gray_scale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Frame', frame)

    frame_buffer.append(frame)
    if len(frame_buffer) > delay:
            previous_frame = np.invert(frame_buffer[len(frame_buffer) - 1 - delay])
    else:
        previous_frame = np.invert(frame)
   
    diff = difference(frame, previous_frame, alpha, threshold, ghost_frame)
    cv2.imshow('Differences', diff)

    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# release the video capture object
cap.release()

# Closes all the windows currently opened.
cv2.destroyAllWindows()

