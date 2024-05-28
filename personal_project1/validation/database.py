import datetime
import pyodbc
import logging

logging.basicConfig(filename='C:/Project/personal_project1/output/face_recognition.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def create_database_connection():
    server = 'facerecognitionproject.database.windows.net'
    database = 'employees'
    username = 'user'
    password = '@Test123'
    driver = '{ODBC Driver 18 for SQL Server}'
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    
    try:
        connection = pyodbc.connect(connection_string)
        logging.info("Database connection successful")
        return connection
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return None

def close_database_connection(connection):
    if connection and not connection.closed:
        connection.close()
        logging.info("Database connection closed")
    else:
        logging.info("Database connection is already closed or invalid")

def insert_employee(connection, employee_name, department):
    try:
        cursor = connection.cursor()
        query = "INSERT INTO Employee (employee_name, department, date_created, date_edited) VALUES (?, ?, GETDATE(), GETDATE()); SELECT SCOPE_IDENTITY();"
        cursor.execute(query, (employee_name, department))
        connection.commit()
        employee_id = cursor.fetchone()[0]
        logging.info("Employee inserted successfully")
        return employee_id
    except Exception as e:
        logging.error(f"Error inserting employee: {e}")
        return None

def update_employee(connection, employee_id, employee_name, department):
    try:
        cursor = connection.cursor()
        query = "UPDATE Employee SET employee_name = ?, department = ?, date_edited = ? WHERE employee_id = ?;"
        current_time = datetime.datetime.now()
        cursor.execute(query, (employee_name, department, current_time, employee_id))
        connection.commit()
        logging.info("Date edited successfully")
    except Exception as e:
        logging.error(f"Error date edited: {e}")

def record_attendance(connection, employee_name, department, recognition_time):
    try:
        cursor = connection.cursor()
        query_employee_id = "SELECT employee_id FROM Employee WHERE employee_name = ? AND department = ?;"
        cursor.execute(query_employee_id, (employee_name, department))
        employee_id = cursor.fetchone()

        if employee_id:
            query_insert_attendance = "INSERT INTO Attendance (employee_id, recognition_time) VALUES (?, ?);"
            cursor.execute(query_insert_attendance, (employee_id[0], recognition_time))
            connection.commit()
            logging.info("Attendance recorded successfully")
        else:
            logging.error("Employee not found")
    except Exception as e:
        logging.error(f"Error recording attendance: {e}")
