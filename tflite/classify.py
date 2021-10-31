#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import argparse
import tflite_runtime.interpreter as tflite

def decode(characters, y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()
    captcha_symbols = captcha_symbols + ' '
    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    count = 0

    with open(args.output, 'w') as output_file:
        tf_interpreter = tflite.Interpreter(args.model_name)
        tf_interpreter.allocate_tensors()
        input_tf = tf_interpreter.get_input_details()
        output_tf = tf_interpreter.get_output_details()

        files = os.listdir(args.captcha_dir)
        files = sorted(files)

        for x in files:
            # Load & Preprocess Image
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = np.array(rgb_data, dtype=np.float32) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])

            tf_interpreter.set_tensor(input_tf[0]['index'],image)
            tf_interpreter.invoke()
            prediction = []
            for output_node in output_tf:
                prediction.append(tf_interpreter.get_tensor(output_node['index']))
            prediction = np.reshape(prediction,(len(output_tf),-1))
            predictedAnswer = decode(captcha_symbols, prediction)
            predictedAnswer = predictedAnswer.replace(" ", "")
            output_file.write(x + "," + predictedAnswer + "\n")

            file = x[:-4]
            print('Classified (count ' + str(count) + ') ' + file + '///' + predictedAnswer)
            count = count + 1

if __name__ == '__main__':
    main()
