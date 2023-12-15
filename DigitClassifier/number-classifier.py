from imutils.video import FPS
import tensorflow as tf
import cv2
import time
import os
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = "digit_recognizer.model"

print("[INFO] Starting video stream...")
video_capture = cv2.VideoCapture(0)

print("[INFO] Importing model {}".format(model_name))
model = tf.keras.models.load_model(model_name)

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
s = time.time()
fps = FPS().start()

print("[INFO] Displaying video")
while True:

    ret, frame = video_capture.read()

    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    IMG_SIZE = 28
    small_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    new_array = cv2.resize(gray_small_frame, (IMG_SIZE, IMG_SIZE))

    prediction = model.predict([new_array.reshape(1, IMG_SIZE * IMG_SIZE)])

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, CATEGORIES[int(np.argmax(prediction[0]))], (int(0), int(video_width * .6)), font, 5.0, (0, 255, 0), 1)

    cv2.imshow('Video', frame)
    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print("[INFO] Process finished")
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

video_capture.release()
cv2.destroyAllWindows()
