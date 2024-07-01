import os
import logging
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization,
    Concatenate, MaxPooling2D, GlobalAveragePooling2D, Add, LeakyReLU, LRN
)
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

tf.keras.utils.set_random_seed(42)  # For reproducibility

class RobustCNN:
    def __init__(self, model_path, input_shape_images=(256, 256, 3), input_shape_landmarks=(68, 2), num_moods=7, num_races=5, feature_length=20):
        self.model_path = model_path
        self.input_shape_images = input_shape_images
        self.input_shape_landmarks = input_shape_landmarks
        self.num_moods = num_moods
        self.num_races = num_races
        self.feature_length = feature_length
        self.model = self.load_or_create_model()

    def residual_block(self, x, filters, increase_filter=False):
        shortcut = x
        stride = 1
        if increase_filter:
            stride = 2
            shortcut = Conv2D(filters, (1, 1), strides=stride)(shortcut)

        x = Conv2D(filters, (3, 3), padding='same', strides=stride)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Dropout(0.2)(x)
        return x

    def build_model(self):
        image_input = Input(shape=self.input_shape_images, name='image_input')
        landmark_input = Input(shape=self.input_shape_landmarks, name='landmark_input')
        feature_input = Input(shape=(self.feature_length,), name='feature_input')

        # Enhanced model with more convolutional layers and LRN
        x = Conv2D(96, (7, 7), strides=(4, 4), padding='same', activation='relu')(image_input)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = LRN()(x)
        x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = LRN()(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = GlobalAveragePooling2D()(x)

        y = Flatten()(landmark_input)
        y = Dense(512, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Dense(256, activation='relu')(y)
        y = BatchNormalization()(y)

        concatenated = Concatenate()([x, y, feature_input])
        z = Dense(1024, activation='relu')(concatenated)
        z = Dropout(0.5)(z)

        mood_output = Dense(self.num_moods, activation='softmax', name='mood_output')(z)
        age_output = Dense(1, name='age_output')(z)
        gender_output = Dense(1, activation='sigmoid', name='gender_output')(z)
        race_output = Dense(self.num_races, activation='softmax', name='race_output')(z)

        model = Model(inputs=[image_input, landmark_input, feature_input], outputs=[mood_output, age_output, gender_output, race_output])
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

    def load_or_create_model(self):
        if os.path.exists(self.model_path):
            logging.info(f"Loading model from {self.model_path}")
            return tf.keras.models.load_model(self.model_path)
        else:
            logging.info("Model not found, creating a new one.")
            model = self.build_model()
            model.save(self.model_path)
            return model

    def train(self, train_data, val_data, epochs=100, batch_size=16):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)

        x_train, y_train = train_data
        x_val, y_val = val_data

        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, checkpoint]
        )
        return history
