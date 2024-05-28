import datetime
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFormLayout, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal
from sklearn import svm
from backend import load_known_faces, VideoStreamThread
from database import close_database_connection, create_database_connection, insert_employee, update_employee, record_attendance

# The EmployeeForm class in Python creates a dialog window for entering employee data and submitting
# it to a database.
class EmployeeForm(QDialog):
    def __init__(self, connection, is_update=False, employee_id=None, parent=None):
        super(EmployeeForm, self).__init__(parent)
        self.setWindowTitle("Employee Data")
        self.setGeometry(100, 100, 300, 200)

        self.connection = connection
        self.is_update = is_update
        self.employee_id = employee_id

        layout = QVBoxLayout()
        self.setLayout(layout)

        form_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.department_input = QLineEdit()

        form_layout.addRow("Employee Name:", self.name_input)
        form_layout.addRow("Department:", self.department_input)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_data)

        layout.addLayout(form_layout)
        layout.addWidget(self.submit_button)

    def submit_data(self):
        employee_name = self.name_input.text().strip()
        department = self.department_input.text().strip()

        if employee_name and department:
            try:
                if self.is_update:
                    update_employee(self.connection, self.employee_id, employee_name, department)
                else:
                    insert_employee(self.connection, employee_name, department)
                self.close()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Invalid Input", "Please fill in all fields.")

# The `FaceRecogScreen` class is a QDialog subclass that implements a face recognition screen with
# video streaming and recognition updates.
class FaceRecogScreen(QDialog):
    def __init__(self, clf, known_face_encodings, encoded_labels, label_encoder, connection, parent=None):
        super(FaceRecogScreen, self).__init__(parent)
        self.connection = connection # Set the connection as an instance variable
        self.setWindowTitle("Face Recognition Screen")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        self.label_cam = QLabel(self)
        layout.addWidget(self.label_cam)

        self.label_info = QLabel(self)
        layout.addWidget(self.label_info)

        self.pushButton_cancel = QPushButton("Exit App", self)
        layout.addWidget(self.pushButton_cancel)
        self.pushButton_cancel.clicked.connect(self.stop_video)

        self.setLayout(layout)

        self.clf = clf
        self.known_face_encodings = known_face_encodings
        self.encoded_labels = encoded_labels
        self.label_encoder = label_encoder

        # Corrected instantiation with all required arguments
        self.video_thread = VideoStreamThread(self.clf, self.known_face_encodings, self.encoded_labels, self.label_encoder, self.connection)
        self.video_thread.ImageUpdate.connect(self.ImageUpdateSlot)
        self.video_thread.start()

    def ImageUpdateSlot(self, qImg, recognized_name):
        pixmap = QPixmap.fromImage(qImg)
        self.label_cam.setPixmap(pixmap)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self.label_info.setText(f"Recognized: {recognized_name} at {current_time}")

    def stop_video(self):
        self.video_thread.stop()
        self.close()

# The `MainWindow` class in Python sets up a GUI window for a face recognition system with buttons to
# start face recognition, add new employees, and update employee information.
class MainWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FaReSys")
        self.setGeometry(100, 100, 200, 100)

        self.faceRecogButton = QPushButton("Start Face Recognition", self)
        self.faceRecogButton.clicked.connect(self.openFaceRecogScreen)

        self.employeeFormButton = QPushButton("Add New Employee", self)
        self.employeeFormButton.clicked.connect(self.openNewEmployeeForm)

        self.updateEmployeeFormButton = QPushButton("Update Employee", self)
        self.updateEmployeeFormButton.clicked.connect(self.openUpdateEmployeeForm)

        layout = QVBoxLayout()
        layout.addWidget(self.faceRecogButton)
        layout.addWidget(self.employeeFormButton)
        layout.addWidget(self.updateEmployeeFormButton)
        self.setLayout(layout)
        
        self.connection = create_database_connection()

        self.clf = svm.SVC()
        self.known_face_names = []
        
        known_faces = {
            "C:/Project/personal_project1/training/IT Department/Reynier Abito/": "Reynier Abito",
            "C:/Project/personal_project1/training/IT Department/Keane Farol/": "Keane Farol",
            "C:/Project/personal_project1/training/IT Department/Justin Juson/": "Justin Juson",
        }
        # Corrected unpacking to include all four return values
        self.known_face_encodings, self.encoded_labels, self.clf, self.label_encoder = load_known_faces(known_faces)

    def openFaceRecogScreen(self):
        # Ensure the connection is established before opening the face recognition screen
        self.connection = create_database_connection()
        self.faceRecogScreen = FaceRecogScreen(self.clf, self.known_face_encodings, self.encoded_labels, self.label_encoder, self.connection)
        self.faceRecogScreen.show()

    def openEmployeeForm(self, is_update=False, employee_id=None):
        self.employeeForm = EmployeeForm(self.connection, is_update, employee_id)
        self.employeeForm.show()
        
    def openNewEmployeeForm(self):
        self.employeeForm = EmployeeForm(self.connection, is_update=False)
        self.employeeForm.show()

    def openUpdateEmployeeForm(self, employee_id):
        self.employeeForm = EmployeeForm(self.connection, is_update=True, employee_id=employee_id)
        self.employeeForm.show()
        
    def closeEvent(self, event):
        close_database_connection(self.connection)
        event.accept()
        
if __name__ == "__main__":
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
