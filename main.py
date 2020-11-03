import cv2
import pickle

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

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
        if 45 <= conf <= 90:

            font = cv2.FONT_HERSHEY_PLAIN
            name = labels[id_]
            output = name + "  " + str(round(conf, 2))
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, output, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        color = (0, 255, 0)
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)

    # to show image

    cv2.imshow('frame', frame)
    if cv2.waitKey(29) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
