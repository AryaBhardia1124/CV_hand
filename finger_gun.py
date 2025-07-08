import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

base_options = python.BaseOptions(model_asset_path = 'gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options = base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb)
    result = recognizer.recognize(mp_image)

    if result.gestures:
        for gesture in result.gestures:
            top_gesture = gesture[0]
            gesture_name = top_gesture.category_name
            confidence = top_gesture.score

            cv2.putText(img, f'{gesture_name}: {confidence:.2f}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

