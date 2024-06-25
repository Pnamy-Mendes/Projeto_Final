import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def get_vgg16_teacher_model(input_shape=(48, 48, 3), num_classes=7):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=x)

    # Freeze the layers of VGG16 except the last block
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
