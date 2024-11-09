import cv2
import numpy as np
import sys

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize video capture
if len(sys.argv) == 1:
    filename = input("Video filename: ")
else:
    filename = sys.argv[1]
cap = cv2.VideoCapture(filename)

while True:
    # Read the current frame
    ret, frame = cap.read()
    
    # Apply the background subtractor to obtain the foreground mask
    fg_mask = bg_subtractor.apply(frame)
    
    # Perform noise removal on the foreground mask
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("Motion Detection", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
