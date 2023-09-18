#Facial Recognition

This Python script is a Face Recognition System that uses the face_recognition, OpenCV, NumPy, Pandas, and TensorFlow libraries to recognize known faces from a live video feed. 
The system can identify known faces and add new faces to its database for recognition. Below, we will provide an overview of the script's functionality and how to use it.

Prerequisites
Before using this Face Recognition System, make sure you have the following installed:

Python (3.6 or higher)
Required Python libraries: face_recognition, OpenCV, NumPy, Pandas, and TensorFlow
Usage
Clone or download the script and ensure that you have the necessary libraries installed.

Create a CSV file named database.csv to store known face encodings and their corresponding names. The file should have two columns: 'Name' and 'Encoding'.


Important Functions
Facerec.load_encoding_images(): Loads known face encodings and names from the database.csv file.
Facerec.save_encoding_image(name, encoding): Saves a new face encoding along with the corresponding name to the database.
Facerec.detect_known_faces(frame): Detects known faces in a video frame, recognizes them, and returns their locations and names.
Facerec.update_known_faces(frame, face_encoding): Prompts the user to enter the name for an unknown face and saves it to the database.
main(): The main function initializes the Face Recognition System, loads known faces, and starts capturing video frames for recognition.
Known Issues
This system may not work well in low-light conditions or if faces are not well-lit.
Recognizing faces with accessories (e.g., glasses, hats) may be less accurate.
