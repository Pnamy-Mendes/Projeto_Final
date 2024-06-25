import os
import logging
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization,
    Concatenate, MaxPooling2D, GlobalAveragePooling2D, Add
)
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

tf.keras.utils.set_random_seed(42)  # For reproducibility, if necessary

class RobustCNN:
    def __init__(self, model_path, input_shape_images=(256, 256, 3), input_shape_landmarks=(68, 2), num_moods=7, num_races=5):
        self.model_path = model_path
        self.input_shape_images = input_shape_images
        self.input_shape_landmarks = input_shape_landmarks
        self.num_moods = num_moods
        self.num_races = num_races
        self.model = self.load_or_create_model()

    def residual_block(self, x, filters, increase_filter=False):
        try:
            shortcut = x
            stride = 1
            if increase_filter:
                stride = 2
                shortcut = Conv2D(filters, (1, 1), strides=stride)(shortcut)

            x = Conv2D(filters, (3, 3), padding='same', strides=stride, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Add()([x, shortcut])
            x = Dropout(0.2)(x)
            return x
        except Exception as e:
            logging.error(f"Error in residual block: {e}")
            raise

    def build_model(self):
        try:
            image_input = Input(shape=self.input_shape_images, name='image_input')
            landmark_input = Input(shape=self.input_shape_landmarks, name='landmark_input')

            x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
            x = BatchNormalization()(x)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = self.residual_block(x, 64, increase_filter=True)
            x = self.residual_block(x, 128, increase_filter=True)
            x = self.residual_block(x, 256, increase_filter=True)
            x = GlobalAveragePooling2D()(x)

            y = Flatten()(landmark_input)
            y = Dense(512, activation='relu')(y)
            y = BatchNormalization()(y)
            y = Dense(256, activation='relu')(y)
            y = BatchNormalization()(y)

            concatenated = Concatenate()([x, y])
            z = Dense(1024, activation='relu')(concatenated)
            z = Dropout(0.5)(z)

            mood_output = Dense(self.num_moods, activation='softmax', name='mood_output')(z)
            age_output = Dense(1, activation='linear', name='age_output')(z)
            gender_output = Dense(1, activation='sigmoid', name='gender_output')(z)
            race_output = Dense(self.num_races, activation='softmax', name='race_output')(z)

            model = Model(inputs=[image_input, landmark_input], outputs=[mood_output, age_output, gender_output, race_output])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss={
                              'mood_output': 'categorical_crossentropy', 
                              'age_output': 'mse', 
                              'gender_output': 'binary_crossentropy', 
                              'race_output': 'categorical_crossentropy'
                          },
                          metrics={'mood_output': 'accuracy', 'gender_output': 'accuracy', 'race_output': 'accuracy'})

            model.summary()
            return model
        except Exception as e:
            logging.error(f"Error building the model: {e}")
            raise

    def load_or_create_model(self):
        try:
            if os.path.exists(self.model_path):
                logging.info(f"Loading model from {self.model_path}")
                return tf.keras.models.load_model(self.model_path)
            else:
                logging.info("Model not found, creating a new one.")
                model = self.build_model()
                model.save(self.model_path)
                return model
        except Exception as e:
            logging.error(f"Error loading or creating the model: {e}")
            raise
