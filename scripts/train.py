# Train
# !/usr/bin/env python3
# Imports
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Build a Keras model given some parameters
def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = keras.layers.Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same',
                                    kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d' % (i + 1))(x) for i in
         range(captcha_length)]
    model = keras.Model(inputs=input_tensor, outputs=x)

    return model


# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, label_file, batch_size, captcha_length, captcha_symbols, captcha_width,
                 captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        with open(label_file, "+r") as f:
            labelList = f.readlines()
            self.labels = dict(
                zip(map(lambda x: x.split(",")[0], labelList), map(lambda x: x.split(",")[1].strip(), labelList)))
        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)
        # print('dir name ' + self.directory_name)
        # print(self.count)

    def __len__(self):
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, len(self.captcha_symbols)), dtype=np.uint8) for i in range(self.captcha_length)]

        for i in range(min(self.batch_size, len(self.files))):
            random_image_name = random.choice(list(self.labels.keys()))
            random_image_label = self.labels[random_image_name]
            random_image_file = self.files[random_image_name[:-4]]

            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_name[:-4]))
            self.labels.pop(random_image_name)

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            processed_data = preprocess(raw_data)
            X[i] = processed_data

            if len(random_image_label) < self.captcha_length:
                balance = self.captcha_length - len(random_image_label)
                for x in range(0, balance):
                    random_image_label = random_image_label + ' '

            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y


def main():
    # Inputs
    width = 128
    height = 64
    length = 6
    batch_size = 32
    epochs = 2  # 8

    train_dataset = '/content/train_set/'
    train_labels_file = '/content/train_labels.txt'
    validate_dataset = '/content/val_set/'
    val_labels_file = '/content/val_labels.txt'

    output_model_name = '/content/model_19_e7'
    input_model = '/content/model_19_e5'

    captcha_symbols_t = captcha_symbols + ' '

    with tf.device('/device:GPU:0'):
        model = create_model(length, len(captcha_symbols_t), (height, width, 3))
        if input_model is not None:
            model.load_weights(input_model + '.h5')

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])
        model.summary()

        training_data = ImageSequence(train_dataset, train_labels_file, batch_size, length, captcha_symbols_t, width,
                                      height)
        validation_data = ImageSequence(validate_dataset, val_labels_file, batch_size, length, captcha_symbols_t, width,
                                        height)

        callbacks = [keras.callbacks.EarlyStopping(patience=3),
                     # keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(output_model_name + '.h5', save_best_only=False)]

        # Save the model architecture to JSON
        with open(output_model_name + ".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit_generator(generator=training_data,
                                validation_data=validation_data,
                                epochs=epochs,
                                callbacks=callbacks,
                                use_multiprocessing=True)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + output_model_name + '_resume.h5')
            model.save_weights(output_model_name + '_resume.h5')


if __name__ == '__main__':
    main()