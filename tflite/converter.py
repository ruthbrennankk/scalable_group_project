#!/usr/bin/env python3
# Converter

import tensorflow as tf
import tensorflow.keras as keras

def main(model_name):
    with open(f'/content/{model_name}.json') as f:
        model = f.read()
    model = keras.models.model_from_json(model)
    model.load_weights(f'/content/{model_name}.h5')
    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = cvt.convert()
    with open(f'/content/converted_{model_name}.tflite', 'wb') as f:
        f.write(tflite)

if __name__ == '__main__':
    model_name = 'test'
    main(model_name)
