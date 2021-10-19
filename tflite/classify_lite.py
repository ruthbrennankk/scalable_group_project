# Classify - Lite
# !/usr/bin/env python3
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def decode(characters, y):
    y = np.argmax(np.array(y), axis=1)
    # y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


def main():
    model_name = '/content/converted_test.tflite'
    captcha_dir = '/content/testers_4_characters_no_lowercase'
    output = '/content/converted_4_output.csv'
    symbols = '/content/drive/MyDrive/Year V/Scalable Computing/Practical 1/symbols.txt'

    symbols_file = open(symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    correct = 0
    incorrect = 0

    with open(output, 'w') as output_file:

        tf_interpreter = tflite.Interpreter(model_name)
        tf_interpreter.allocate_tensors()
        input_tf = tf_interpreter.get_input_details()
        output_tf = tf_interpreter.get_output_details()

        files = os.listdir(captcha_dir)
        files = sorted(files)

        for x in files:
            # Load & Preprocess Image - Currently setup to work with test (the sample model)
            raw_data = cv2.imread(os.path.join(captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = np.array(rgb_data, dtype=np.float32) / 255.0 # update
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])

            tf_interpreter.set_tensor(input_tf[0]['index'], image)
            tf_interpreter.invoke()
            prediction = []
            for output_node in output_tf:
                prediction.append(tf_interpreter.get_tensor(output_node['index']))
            prediction = np.reshape(prediction, (len(output_tf), -1))
            decoded_symbol = decode(captcha_symbols, prediction)
            output_file.write(x + "," + decoded_symbol + "\n")

            print('Classified ' + x + '///' + decoded_symbol)

            answer = x[:-4]
            if (answer == decoded_symbol):
                correct = correct + 1
            else:
                incorrect = incorrect + 1

    print('correct = ' + str(correct))
    print('incorrect =' + str(incorrect))


if __name__ == '__main__':
    main()