import cv2
import face_recognition
import glob
import datetime
import logging
import sys
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QPushButton
from database import create_database_connection, close_database_connection, insert_employee, update_employee, record_attendance

# Ensure the Haar Cascade XML file is in your project directory or specify the correct path
haar_cascade_path = 'C:/Project/personal_project1/xmls/haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

last_recognition_times = {}

"""
The function `load_known_faces` loads known faces, encodes them, fits a Support Vector Machine
classifier, and returns the encoded faces, labels, classifier, and label encoder.
    
:param known_faces: The `load_known_faces` function takes a dictionary `known_faces` as input, where
the keys are file paths to folders containing images of faces, and the values are the names
associated with those faces. The function loads the images, extracts the face encodings using
face_recognition library, encodes
:return: The function `load_known_faces` returns four values:
1. `known_face_encodings`: A list of face encodings for the known faces.
2. `encoded_labels`: Numerical labels encoded from the string labels of the known faces.
3. `clf`: An SVM classifier trained on the known face encodings and encoded labels.
4. `label_encoder`: A `LabelEncoder` object
"""
    
def load_known_faces(known_faces):
    logging.info("Loading known faces...")
    known_face_encodings = []
    known_face_names = []
    label_encoder = LabelEncoder()

    for face_folder, name in known_faces.items():
        if not face_folder.endswith('/'):
            face_folder += '/'
        image_path = f"{face_folder}*"
        image_files = glob.glob(image_path)
        if not image_files:
            logging.warning(f"No files found for path: {image_path}")
            continue
        for image_file in image_files:
            try:
                known_image = face_recognition.load_image_file(image_file)
                known_encoding = face_recognition.face_encodings(known_image)[0]
                known_face_encodings.append(known_encoding)
                known_face_names.append(name)
                logging.info(f"Loaded image: {image_file}")
            except Exception as e:
                logging.error(f"Error loading image {image_file}: {e}")

    # Encode the string labels to numerical labels
    encoded_labels = label_encoder.fit_transform(known_face_names)

    clf = svm.SVC(gamma='scale')
    clf.fit(known_face_encodings, encoded_labels)

    return known_face_encodings, encoded_labels, clf, label_encoder
"""
The function `recognize_faces` takes a frame, detects faces using Haar Cascade, recognizes faces
using a classifier, and updates attendance records for recognized faces in the IT department.
    
:param frame: The `frame` parameter is the current frame captured from a video stream or camera
feed. It is typically a numpy array representing an image in OpenCV format. This frame will be used
for face recognition and drawing bounding boxes around detected faces
:param clf: The `clf` parameter in the `recognize_faces` function seems to be a classifier model
used for predicting the identity of faces based on their encodings. It is likely a machine learning
model that has been trained to recognize faces. The function uses this classifier to predict the
identity of the detected faces
:param known_face_encodings: The `known_face_encodings` parameter in the `recognize_faces` function
is a list of known face encodings that are used for comparing and recognizing faces in the input
frame. These face encodings are typically generated during a training phase where faces are detected
and encoded into numerical representations for comparison during
:param encoded_labels: The `encoded_labels` parameter in the `recognize_faces` function seems to be
used for storing the encoded labels of known faces. These encoded labels are likely used for
recognizing and matching faces during the face recognition process. The function uses these encoded
labels along with the classifier (`clf`) to predict the
:param label_encoder: The `label_encoder` parameter in the `recognize_faces` function is used to
encode and decode string labels to numerical labels and vice versa. It is used to transform the
labels for training the classifier and then decoding the predicted numerical labels back to their
original string labels for display purposes. This helps in
:param connection: The `connection` parameter in the `recognize_faces` function is likely a database
connection object that allows the function to interact with a database. It is used to execute SQL
queries to retrieve and update employee information and record attendance. The function uses this
connection to query the database for employee information, update
:return: The function `recognize_faces` returns the modified `frame` with bounding boxes and labels
drawn around recognized faces, as well as the `recognized_name` of the person detected in the frame.
"""
def recognize_faces(frame, clf, known_face_encodings, encoded_labels, label_encoder, connection):
    
    global last_recognition_times
    recognized_name = 'Unknown' # Initialize recognized_name with a default value
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the frame to RGB color space
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    gray_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_RGB2GRAY)
    
    # Using Haar Cascade for face detection
    faces_rect = haar_cascade.detectMultiScale(gray_small_frame, scaleFactor=1.1, minNeighbors=9)
    
    for (x, y, w, h) in faces_rect:
        top = y
        right = x + w
        bottom = y + h
        left = x
        
        # Adjusting the coordinates for the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Cropping the frame to the detected face and converting to RGB
        face_region = frame[top:bottom, left:right]
        rgb_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Ensure the face region is in RGB color space
        rgb_face_region = np.ascontiguousarray(rgb_face_region[:, :, ::-1])
        
        # Check if the face_region contains a face before attempting to get encodings
        face_encodings = face_recognition.face_encodings(rgb_face_region)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            matches = clf.predict([face_encoding])

            # Decode numerical labels back to string labels
            recognized_name = label_encoder.inverse_transform(matches)[0]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{recognized_name}"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.35, (255, 255, 255), 1)

            if recognized_name not in last_recognition_times or (datetime.datetime.now() - last_recognition_times[recognized_name]).total_seconds() > 60:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logging.info(f"Recognized: {recognized_name} at {current_time}")
                last_recognition_times[recognized_name] = datetime.datetime.now()
                
                try:
                    cursor = connection.cursor()
                    query = "SELECT employee_id FROM Employee WHERE employee_name = ? AND department = ?;"
                    cursor.execute(query, (recognized_name, "IT Department"))
                    existing_employee = cursor.fetchone()

                    if existing_employee:
                        employee_id = existing_employee[0]
                        update_employee(connection, employee_id, recognized_name, "IT Department")
                    else:
                        insert_employee(connection, recognized_name, "IT Department")
                    
                    record_attendance(connection, recognized_name, "IT Department", current_time)
                except Exception as e:
                    logging.error(f"Error recording attendance: {e}")
                
    return frame, recognized_name

class VideoStreamThread(QThread):
    ImageUpdate = pyqtSignal(QImage, str)

    def __init__(self, clf, known_face_encodings, encoded_labels, label_encoder, connection):
        super().__init__()
        self.clf = clf
        self.known_face_encodings = known_face_encodings
        self.encoded_labels = encoded_labels
        self.label_encoder = label_encoder
        self.connection = connection
        self.cap = cv2.VideoCapture(0)
        self.ThreadActive = True

    def run(self):
        while self.ThreadActive:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame, recognized_name = recognize_faces(frame, self.clf, self.known_face_encodings, self.encoded_labels, self.label_encoder, self.connection)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.ImageUpdate.emit(qImg, recognized_name)

    def stop(self):
        self.ThreadActive = False
        self.cap.release()
        close_database_connection(self.connection)
        self.quit()
       
class FaceRecogScreen(QDialog):
    def __init__(self, clf, known_face_encodings, encoded_labels, label_encoder, connection, parent=None):
        super(FaceRecogScreen, self).__init__(parent)
        self.setWindowTitle("Face Recognition Screen")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        self.label_cam = QLabel(self)
        layout.addWidget(self.label_cam)

        self.label_info = QLabel(self)
        layout.addWidget(self.label_info)

        self.pushButton_cancel = QPushButton("Cancel", self)
        layout.addWidget(self.pushButton_cancel)
        self.pushButton_cancel.clicked.connect(self.stop_video)

        self.setLayout(layout)

        self.clf = clf
        self.known_face_encodings = known_face_encodings
        self.encoded_labels = encoded_labels
        self.label_encoder = label_encoder
        self.connection = connection

        # Corrected instantiation with all required arguments
        self.video_thread = VideoStreamThread(self.clf, self.known_face_encodings, self.encoded_labels, self.label_encoder, self.connection)
        self.video_thread.ImageUpdate.connect(self.ImageUpdateSlot)
        self.video_thread.start()

    def ImageUpdateSlot(self, qImg, recognized_name):
        pixmap = QPixmap.fromImage(qImg)
        self.label_cam.setPixmap(pixmap)
        self.label_info.setText(f"Recognized: {recognized_name} at {datetime.datetime.now().strftime('%H:%M:%S')}")

    def stop_video(self):
        self.video_thread.stop()
        self.close()

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 200, 100)

        self.faceRecogButton = QPushButton("Start Face Recognition", self)
        self.faceRecogButton.clicked.connect(self.openFaceRecogScreen)

        layout = QVBoxLayout()
        layout.addWidget(self.faceRecogButton)
        self.setLayout(layout)

        self.clf = svm.SVC()
        self.known_face_names = []

        known_faces = {
            "C:/Project/personal_project1/training/IT Department/Reynier Abito/": "Reynier Abito",
            "C:/Project/personal_project1/training/IT Department/Keane Farol/": "Keane Farol",
            "C:/Project/personal_project1/training/IT Department/Justin Juson/": "Justin Juson",
        }
        self.known_face_encodings, self.encoded_labels, self.clf, self.label_encoder = load_known_faces(known_faces)

    def openFaceRecogScreen(self):
        # Ensure the connection is established before opening the face recognition screen
        self.connection = create_database_connection()
        self.faceRecogScreen = FaceRecogScreen(self.clf, self.known_face_encodings, self.encoded_labels, self.label_encoder, self.connection, self)
        self.faceRecogScreen.show()

if __name__ == "__main__":
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
