import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import dlib
from math import hypot
from multiprocessing import Process
from playsound import playsound


cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
model = load_model("mask_recog1.h5")
video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
nose_image = cv2.imread("pignose.png")
alarm_file = "alarm_tone.mp3"
play_alarm = 0


def detect_faces():
    global play_alarm
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        faces_list = []
        preds = []
        for (x, y, w, h) in faces:
            face_frame = frame[y:y + h, x:x + w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list) > 0:
                preds = model.predict(faces_list)
            for pred in preds:
                (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"

            if mask > withoutMask:
                print('mask')
                play_alarm = 0

            else:
                draw_mask(frame)
                play_alarm += 1
                play_sound()

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def draw_mask(frame):
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        # Nose coordinates
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)

        nose_width = int(hypot(left_nose[0] - right_nose[0],
                               left_nose[1] - right_nose[1]) * 1.7)
        nose_height = int(nose_width * 0.77)

        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2),
                    int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2),
                        int(center_nose[1] + nose_height / 2))
        # Adding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)

        frame[top_left[1]: top_left[1] + nose_height,
        top_left[0]: top_left[0] + nose_width] = final_nose


def sound_alarm():
    global play_alarm
    if play_alarm is 1:
        playsound(alarm_file, False)


def play_sound():
    audio_process = Process(target=sound_alarm)
    if not audio_process.is_alive():
        audio_process.run()
    else:
        pass


if __name__ == '__main__':
    detect_faces()
