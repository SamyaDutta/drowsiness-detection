import cv2
import numpy as np
import dlib
from imutils import face_utils

cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Setting up STATUS parameters
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

face_frame = []


# Computes distance between 2 points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


# Checks whether blinked or not
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label_position = (x1, y1 - 10)  # ensures that the text always stays on top of the frame

        landmarks = predict(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # setting up LANDMARKS
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Setting up BASE CONDITIONS for prediction
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "Sleeping"
                color = (255, 255, 0)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy"
                color = (0, 255, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active"
                color = (0, 255, 0)

        cv2.putText(frame, status, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Plotting the Landmarks on screen
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # OUTPUT WINDOWS
    cv2.imshow("Drowsiness Detection", frame)
    cv2.imshow("Facemap", face_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
