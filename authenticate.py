import cv2
import pickle

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    # capturing frames
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 85:

            font = cv2.FONT_HERSHEY_PLAIN
            name = labels[id_]
            color_red = (0, 0, 255)
            color_green = (0, 255, 0)
            stroke = 2
            if name == 'aayush-dua':
                cv2.putText(frame, 'Verified', (x, y), font, 2, color_green, stroke, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Invalid', (x, y), font, 2, color_red, stroke, cv2.LINE_AA)

        color = (0, 255, 0)
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)

    # to show image

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
