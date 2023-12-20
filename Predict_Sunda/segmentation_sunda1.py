import os
from google.cloud import storage
import cv2
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'caraka-credentials.json'
storage_client = storage.Client()

def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req

model = load_model('model_sunda.h5', custom_objects={'req': req})

def preprocess_input_image(image, target_size=(150, 150)):
    resized_image = cv2.resize(image, target_size)
    if resized_image.shape[-1] == 1:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    return resized_image


def predict_character(model, character_image, class_labels, target_size=(150, 150)):
    input_image = preprocess_input_image(character_image, target_size)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    # print('Predictions:', predictions)

    predicted_label_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_label_index]

    return predicted_label

def predict_characters(model, image_path, list_of_boxes, class_labels):
    predicted_labels = []

    for box in list_of_boxes:
        x1, y1, x2, y2 = box
        character_image = image_path[y1:y2, x1:x2]
        predicted_label = predict_character(model, character_image, class_labels)
        predicted_labels.append(predicted_label)

        print(f'Bounding Box Coordinates: ({x1}, {y1}, {x2}, {y2})')

    return predicted_labels


def segment_characters(image_path):
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    threshold = cv2.threshold(blur, 0.5, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
        cv2.rectangle(image_path, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    boxes.sort(key=lambda x: (x[0], x[1]))

    return boxes

def predict_words(labels):
    for id, label in enumerate(labels):
        print(f'Predicted Class {id}: {label}')

    words = []

    for id, label in enumerate(labels):
        if label == 'vowels_o':
            word = words.pop()
            label = word.replace(list(word)[-1], 'o')

        words.append(label)

    return ''.join(word for word in words).lower()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            image_bucket = storage_client.get_bucket(
                'aksara-bucket1')
            filename = request.json['filename']
            img_blob = image_bucket.blob('predict_uploads/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
            img_data = img_blob.download_as_bytes()
            image_path = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond

        # find the max prediction of the image
        class_labels = ['a', 'ba', 'ca', 'da', 'e',
                'ee', 'eu', 'fa', 'ga', 'ha',
                'i', 'ja', 'ka', 'la', 'ma', 'na',
                'nga', 'nya', 'ou', 'pa', 'qa', 'ra',
                'sa', 'ta', 'u', 'va', 'vowels_e',
                'vowels_ee', 'vowels_eu', 'vowels_h',
                'vowels_i', 'vowels_la', 'vowels_ng',
                'vowels_o', 'vowels_r', 'vowels_ra',
                'vowels_u', 'vowels_x', 'vowels_ya',
                'wa', 'xa', 'ya', 'za']
        
        list_of_boxes = segment_characters(image_path)
        predicted_labels = predict_characters(model, image_path, list_of_boxes, class_labels)
        predicted_words = predict_words(predicted_labels)
        print(predicted_words)

        result = {
            "hasil": predicted_words
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
