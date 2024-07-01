# mood_model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from models.mood.attention_layer import add_attention_layer

def get_advanced_model(input_shape_features, leaky_relu_slope, dropout_rate, regularization_rate, n_classes, dense_units_1, dense_units_2, dense_units_3, dense_units_4, dense_units_5):
    input_features = Input(shape=input_shape_features)
    x = Flatten()(input_features)
    x = Dense(dense_units_1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leaky_relu_slope)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units_2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leaky_relu_slope)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units_3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leaky_relu_slope)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units_4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leaky_relu_slope)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units_5)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leaky_relu_slope)(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_features, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
