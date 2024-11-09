import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class VideoStreamApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Initialize video capture and red_intensity
        self.cap = cv2.VideoCapture(0)
        self.red_intensity = 128

        self.initUI()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        self.video_label = QLabel(self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 255)
        self.slider.setValue(self.red_intensity)
        self.slider.valueChanged.connect(self.update_intensity)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.setWindowTitle("OpenCV Video Stream with Slider")
        self.resize(640, 480)
        self.show()

    def update_intensity(self, value):
        self.red_intensity = value

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame[:, :, 2] = np.clip(frame[:, :, 2] * (self.red_intensity / 128), 0, 255)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_stream_app = VideoStreamApp()
    sys.exit(app.exec_())
