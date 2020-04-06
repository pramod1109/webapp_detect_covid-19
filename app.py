from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app=Flask(__name__)

model = load_model('Covid_Binary.h5')

img_width, img_height = 150, 150

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("uploaded")

        # Make prediction
        img = image.load_img(f, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        classes = model.predict_proba(img, batch_size=1, verbose=1)
        print(classes[0][0])
        if classes[0][0]>1.0:
            r="normal"
        elif classes[0][0]==1:
            r="re do the test after 1 or 2 days,there may be a chance"
        else:
            r="covid-19"
        return r
    return None

if __name__=="__main__":
    app.run(debug=True)
