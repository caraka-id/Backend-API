#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
from google.cloud import storage
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input


app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'caraka-credentials.json'
storage_client = storage.Client()

def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req

model = load_model('model_trained_efficientNetV2B09998A.h5', custom_objects={'req': req})

def preprocess_input_image(image, target_size=(150, 150)):
    resized_image = cv2.resize(image, target_size)
    if resized_image.shape[-1] == 1:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    return resized_image


def predict_character(model, class_labels, character_image, target_size=(150, 150)):
    input_image = preprocess_input_image(character_image, target_size)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    # print('Predictions:', predictions)

    predicted_label_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_label_index]

    return predicted_label


def segment_characters(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    threshold = cv2.threshold(blur, 0.5, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    boxes.sort(key=lambda x: (x[0], x[1]))

    return boxes


def predict_characters(model, img_np, list_of_boxes, class_labels):
    predicted_labels = []

    for box in list_of_boxes:
        x1, y1, x2, y2 = box
        character_image = img_np[y1:y2, x1:x2]
        predicted_label = predict_character(model, class_labels, character_image)
        predicted_labels.append(predicted_label)

        print(f'Bounding Box Coordinates: ({x1}, {y1}, {x2}, {y2})')

    return predicted_labels


def predict_words(labels):
    for id, label in enumerate(labels):
        print(f'Predicted Class {id}: {label}')

    words = []

    for id, label in enumerate(labels):
        if label == 'i':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'i')
            
        elif label == 'ee':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'è')

        elif label == 'e':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'e')
            
        elif label == 'o':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'o')

        elif label == 'au':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'au')
        
        elif label == 'ai':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'ai')
            
        elif label == 'u':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'u')
            
        elif label == 'h':
            label = 'h'
        
        elif label == 'n':
            label = 'n'
        
        elif label == 'r':
            label = 'r'
            
        elif label == 'ng':
            label = 'ng'
        
        elif label == 'nengen':
            label =  labels[id - 1].replace(list(labels[id - 1])[1], '')

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
            img_np = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond

        # find the max prediction of the image
        class_labels = ['a', 'ai', 'au', 'ba', 'ca', 'da', 'e', 'ee', 'ga', 'gha', 'h', 'ha', 'i', 'ja', 'ka', 'la', 
                'ma', 'n', 'na', 'nengen', 'ng', 'nga', 'nya', 'o', 'pa', 'r', 'ra', 'sa', 'ta', 'u', 'wa', 'ya']


        list_of_boxes = segment_characters(img_np)
        predicted_labels = predict_characters(model, img_np, list_of_boxes, class_labels)
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


