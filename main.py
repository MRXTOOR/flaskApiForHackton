from flask import Flask, request
import cv2
import dlib
from imutils import face_utils
import numpy as np
import random
import os
import urllib.request

app = Flask(__name__)

database = {}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def extract_face_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        return None

    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)

    return shape.flatten()


@app.route('/upload', methods=['POST'])
def upload_image():
    image_url = request.form['image_url']
    image_path = 'temp_image.jpg'

    urllib.request.urlretrieve(image_url, image_path)

    image = cv2.imread(image_path)
    metrics = extract_face_metrics(image)
    if metrics is None:
        return 'No face found in the image!'

    filename = str(random.randint(1, 1000000000))
    database[filename] = metrics

    os.remove(image_path)

    return 'Metrics saved for image {}'.format(filename)


@app.route('/check', methods=['POST'])
def check_image():
    image_url = request.form['image_url']
    image_path = 'temp_image.jpg'

    urllib.request.urlretrieve(image_url, image_path)

    image = cv2.imread(image_path)
    query_metrics = extract_face_metrics(image)
    if query_metrics is None:
        return 'No face found in the image!'

    for filename, metrics in database.items():
        if np.array_equal(metrics, query_metrics):
            os.remove(image_path)
            return 'Face found in the database for image {}'.format(filename)

    os.remove(image_path)
    return 'Face not found in the database'


if __name__ == '__main__':
    app.run()