#!/usr/bin/venv python3

import os
import numpy as np
import random
import cv2
import argparse
import captcha.image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output_dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()
    captcha_symbols_g = captcha_symbols

    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height)
    print("Generating captchas with symbol set {" + captcha_symbols_g + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    filenames = []
    index = 0

    for i in range(args.count):
        caplength = np.random.randint(1, 7)
        random_str = ''.join([random.choice(captcha_symbols_g) for j in range(caplength)])

        img_name = 'img_' + str(i) + '.png'
        image_path = os.path.join(args.output_dir, img_name)
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')

        image = np.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

        with open(args.outputFile, '+a') as f:
            f.write(img_name + "," + random_str + "\n")

if __name__ == '__main__':
    main()