# ReconhecimentoFacial

This is a Python application for face recognition using the Face Recognition library, OpenCV, and TensorFlow. The application captures video from a webcam, detects faces in real-time, and recognizes known faces based on previously saved face encodings. If an unknown face is detected, it prompts the user to enter their name and saves the encoding for future recognition.

#Getting Started

Make sure you have the required libraries installed.
-face_recognition 
-opencv-python 
-numpy 
-pandas
-tensorflow


The application will start capturing video from your webcam and attempt to recognize faces in the video stream.

If a known face is detected, the person's name will be displayed above their face.

If an unknown face is detected, the application will prompt you to enter the person's name. Once you enter the name, the face encoding will be saved for future recognition.

Press the Esc key to exit the application.

#Important Notes
The application uses the Face Recognition library to perform face detection and recognition. Make sure to provide good lighting conditions and clear frontal images of known faces for better recognition accuracy.

Face encodings are saved in the database.csv file, allowing the application to recognize known faces in future sessions.

You can customize the number of training epochs for the neural network model by modifying the epochs parameter in the load_encoding_images and save_encoding_image methods of the Facerec class.

You may need to adjust the frame resizing factor (self.frame_resizing) to optimize the application's performance based on your webcam's capabilities.

The application uses OpenCV for video capture and display. Ensure that your webcam is properly configured and connected to your computer.

TensorFlow is used to create a simple neural network model for encoding faces. You can experiment with different models and architectures to improve recognition accuracy.
