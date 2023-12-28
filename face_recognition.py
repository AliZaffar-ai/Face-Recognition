import cv2
import pickle

# Load the trained model and label dictionary
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/apple/NEWFOLDER/Ali/Face-Detection-Recognition/trainer/trainer.yml')

with open('/Users/apple/NEWFOLDER/Ali/Face-Detection-Recognition/label_dict.txt', 'rb') as f:
    label_dict = pickle.load(f)
id_to_name = {v: k for k, v in label_dict.items()}

# Initialize the webcam and the face detector
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Get the name from the id, if not Ali or Bisma, display Unknown
        name = id_to_name.get(id, "Unknown")
        

        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cam.release()
cv2.destroyAllWindows()
