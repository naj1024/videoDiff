# Example of using cv2 to processes video to highlight differences in time
# 
# Delay a video by N frames and subtract it from the current frame
# static mode - Grab a frame and compare against latest
# loop - on file input loop the file at the end
# ghost - add back original to the output 
# gray - convert frames to gray first
# delay - how many frames ago we compare against
# scale - scale the frame

import sys
import cv2
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QRadioButton, QCheckBox, QPushButton, QFileDialog, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class VideoStreamApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Initialize video capture (0 is the default camera)
        self.file_input = False
        self.filename = ""
        self.video_feed_number = 0

        # Video device
        self.cap = None
        self.grabbed_frame = None

        self.loop_file = False           # on file input do we loop the file
        self.gray_scale = False          # gray scale the original frame source
        self.ghost = False               # add back the original as a ghost
        self.use_static = False          # Use static frame, grabbed
        self.length_of_delay_buffer = 50 # maximum delay
        self.delay = 20                  # frame delay for difference against
        self.alpha = 0.5                 # % of delayed frame to subtract on difference
        self.threshold = 0               # value below which things are set to black
        self.scale = 1.0                 # scale video frames, 0.1 to 3.0

        self.frame_buffer = deque(maxlen = self.length_of_delay_buffer)

        # Set up the user interface
        self.initUI()

        # Start a timer to update frames
        self.timer = QTimer()

        # get the source (which may fail to open)
        self.open_source(not self.file_input)
        self.set_timer()

        # start the updates
        self.timer.timeout.connect(self.update_frame)
        
    def open_source(self, live: bool):
        # assign video device and get video stream (which may fail to open)  
        if self.cap:
            self.video_label.clear()    # remove current image
            self.cap.release()          # destroy source
            self.frame_buffer.clear()   # wipe current time buffer
            self.grabbed_frame = None   # remve any static grabbed frame
        if live:
            self.cap = cv2.VideoCapture(self.video_feed_number)
            self.file_input = False
        else:
            if len(self.filename) > 0:
                self.cap = cv2.VideoCapture(self.filename)
            self.file_input = True  # even if open fails

    def set_timer(self):
        # set the update to the fps of the source
        if(self.cap.isOpened()):
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(1000 / fps) if fps > 0 else 30  # Default to 30 ms if fps is not available
            self.timer.stop()
            self.timer.start(frame_interval)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            #print("video fps", fps, "rate", frame_interval, "msec")

    def initUI(self):
        # Create a label to hold the video
        self.video_label = QLabel(self)
        
        # source, live video and file
        self.fps_label = QLabel("FPS: ")
        self.live_video = QRadioButton("Live")
        self.live_video.setChecked(True)
        self.file_video = QRadioButton("File")
        self.live_video.toggled.connect(self.update_source_toggle)
        self.file_video.toggled.connect(self.update_source_toggle)
        button_name = self.filename
        if button_name == "":
            button_name = "Open File"
        self.open_button = QPushButton(button_name, self)
        self.open_button.setToolTip("Choose file to process for file mode")
        self.open_button.clicked.connect(self.open_file_dialog)

        # button to grab current frame
        self.grab_button = QPushButton("Grab frame", self)
        self.grab_button.setToolTip("Grab current video frame for use in static mode")
        self.grab_button.clicked.connect(self.grab_frame)

        # source in horizontal layout
        self.source_layout = QVBoxLayout()
        self.source_layout.addWidget(self.grab_button)
        self.source_layout.addWidget(self.live_video)
        self.source_layout.addWidget(self.file_video)
        self.source_layout.addWidget(self.open_button)

        # Create a slider to adjust the size of the video
        self.slider_sc = QSlider(Qt.Horizontal, self)
        self.slider_sc.setRange(1, 30)  
        self.slider_sc.setValue(int(self.scale * 10))  
        self.slider_sc.valueChanged.connect(self.update_scale)  # Connect slider to handler
        self.slider_sc.setToolTip("Scale the video")
        self.scale_label = QLabel(f"Scale: {self.scale:.1f}", self)

        # Create a slider to adjust the delay
        self.slider_d = QSlider(Qt.Horizontal, self)
        self.slider_d.setRange(0, self.length_of_delay_buffer - 1)  
        self.slider_d.setValue(self.delay)  
        self.slider_d.valueChanged.connect(self.update_delay)  # Connect slider to handler
        self.slider_d.setToolTip("Frame delay for video difference")
        self.delay_label = QLabel(f"Delay: {self.delay}", self)

        # Create a slider to adjust the threshold
        self.slider_t = QSlider(Qt.Horizontal, self)
        self.slider_t.setRange(0, 255)  
        self.slider_t.setValue(self.threshold)  
        self.slider_t.valueChanged.connect(self.update_threshold)  # Connect slider to handler
        self.slider_t.setToolTip("Threshold for black")
        self.threshold_label = QLabel(f"Threshold: {self.threshold}", self)

        # Arrange the sliders and their labels
        slider_layout1 = QHBoxLayout()
        slider_layout1.addWidget(self.scale_label)
        slider_layout1.addWidget(self.slider_sc)
        slider_layout2 = QHBoxLayout()
        slider_layout2.addWidget(self.delay_label)
        slider_layout2.addWidget(self.slider_d)
        slider_layout3 = QHBoxLayout()
        slider_layout3.addWidget(self.threshold_label)
        slider_layout3.addWidget(self.slider_t)
        slider_layout = QVBoxLayout()
        slider_layout.addLayout(slider_layout1)
        slider_layout.addLayout(slider_layout2)
        slider_layout.addLayout(slider_layout3)

        fps_layout = QVBoxLayout()
        fps_layout.addWidget(self.fps_label)

        # toggle button 
        self.check_g = QCheckBox("Gray", self)
        self.check_g.setCheckable(True)  # Make the button checkable
        self.check_g.setChecked(self.gray_scale)  # Set initial checked state
        self.check_g.stateChanged.connect(self.check_gray)  # Connect toggle signal
        self.check_g.setToolTip("Set frames to gray scale")
        
        self.check_s = QCheckBox("Ghost", self)
        self.check_s.setCheckable(True)  # Make the button checkable
        self.check_s.setChecked(self.ghost)  # Set initial checked state
        self.check_s.stateChanged.connect(self.check_spectre)  # Connect toggle signal
        self.check_s.setToolTip("Overlay original frame as a ghost")
        
        self.check_l = QCheckBox("Loop", self)
        self.check_l.setCheckable(True)  # Make the button checkable
        self.check_l.setChecked(self.loop_file)  # Set initial checked state
        self.check_l.stateChanged.connect(self.check_loop_file)  # Connect toggle signal
        self.check_l.setToolTip("Loop file input")

        self.check_gr = QCheckBox("Static", self)
        self.check_gr.setCheckable(True)  # Make the button checkable
        self.check_gr.setChecked(self.use_static)  # Set initial checked state
        self.check_gr.stateChanged.connect(self.check_grabbed)  # Connect toggle signal
        self.check_gr.setToolTip("Use grabbed frame as difference frame")

        # layout the checkboxes
        check_layout = QVBoxLayout()
        check_layout.addWidget(self.check_g)
        check_layout.addWidget(self.check_s)
        check_layout.addWidget(self.check_l)
        check_layout.addWidget(self.check_gr)

        # Arrange everything together
        layoutV = QVBoxLayout()
        layoutV.addLayout(self.source_layout)
        layoutV.addLayout(check_layout)
        layoutV.addLayout(slider_layout)
        layoutV.addLayout(fps_layout)

        layoutH = QHBoxLayout()
        layoutH.addWidget(self.video_label)
        layoutH.addLayout(layoutV)

        self.setLayout(layoutH)

        # Set the window title and show the app
        self.setWindowTitle("OpenCV video delay effects")
        self.resize(640, 480)
        self.show()

    def update_source_toggle(self):
        # arrive here if a check box changes state
        # have to check what state we are in as well as we arrive here
        # for state active and state inactive
        if self.live_video.isChecked() and self.file_input:
            self.open_source(True)
            self.set_timer()
        elif self.file_video.isChecked() and not self.file_input:
            self.open_source(False)
            self.set_timer()

    def open_file_dialog(self):
        # Open file dialog to select a file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a video file", "", 
                                                   "All Files (*)", 
                                                   options=options)
        if file_path:
            self.filename = file_path
            if self.file_video.isChecked():
                self.open_source(False)
                self.set_timer()

            # Display only the filename (not the full path) in the label
            filename = file_path.split("/")[-1]
            self.open_button.setText(filename)

    def grab_frame(self):
        self.grabbed_frame = self.frame_buffer[len(self.frame_buffer) - 1]

    def update_scale(self, value: int):
        self.scale = value / 10.0
        self.scale_label.setText(f"Scale: {self.scale:.1f}")
        self.frame_buffer.clear()  # drop previous frames as they may not be gray scale
        self.grabbed_frame = None  # ditto

    def update_delay(self, value: int):
        self.delay = value
        self.delay_label.setText(f"Delay: {self.delay}")

    def update_threshold(self, value: int):
        self.threshold = value
        self.threshold_label.setText(f"Threshold: {self.threshold}")
        
    def update_alpha(self, value: int):
        self.alpha = value / 100
        self.alpha_label.setText(f"Alpha: {self.alpha}")
    
    def check_gray(self, state: bool):
        self.gray_scale = state == Qt.Checked
        self.frame_buffer.clear()  # drop previous frames as they may not be gray scale
        self.grabbed_frame = None  # ditto

    def check_spectre(self, state: bool):
        self.ghost = state == Qt.Checked

    def check_loop_file(self, state: bool):
        self.loop_file = state == Qt.Checked

    def check_grabbed(self, state: bool):
        self.use_static = state == Qt.Checked

    def update_frame(self):
        if self.cap.isOpened():
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                if self.file_input and self.loop_file:
                    # Rewind / loop the file input
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.filename)
                else:
                    self.cap.release()
                    self.video_label.clear()
            else:
                if self.scale != 1.0:
                    height  = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
                    width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    frame = cv2.resize(frame, (int(width * self.scale), int(height * self.scale)), 
                                       fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)

                if self.gray_scale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                self.frame_buffer.append(frame)

                if self.use_static and isinstance(self.grabbed_frame, np.ndarray):
                    previous_frame = np.invert(self.grabbed_frame)
                else:
                    if len(self.frame_buffer) > self.delay:
                            previous_frame = np.invert(self.frame_buffer[len(self.frame_buffer) - 1 - self.delay])
                    else:
                        previous_frame = np.invert(frame)
            
                diff = self.difference(frame, previous_frame)
                
                # Convert the frame to QImage for PyQt5
                if self.gray_scale:
                    h, w = diff.shape
                    bytes_per_line = w
                    q_image = QImage(diff.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                else:
                    h, w, ch = diff.shape
                    bytes_per_line = ch * w
                    q_image = QImage(diff.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Display the image on the label
                self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        # Release the video capture when the window is closed
        self.cap.release()
        event.accept()

    def difference(self, array1: np.ndarray, array2:np.ndarray):
        # blend the two images together, they are time shifted N frames
        beta  = 1.0 - self.alpha   # 2nd image %
        gamma = 30.0
        diff = cv2.addWeighted(array2, self.alpha, array1, beta, gamma)

        # add back ghost original image
        if self.ghost:
            alpha2 = 0.8
            beta2 = 1 - alpha2
            diff = cv2.addWeighted(diff, alpha2, array1, beta2, gamma)

        # print(np.min(diff), np.max(diff), np.median(diff), np.mean(diff), np.std(diff))

        # threshold
        diff = np.where(diff >= self.threshold, diff, 0)

        return diff

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_stream_app = VideoStreamApp()
    sys.exit(app.exec_())
