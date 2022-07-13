import keras.backend as K
import tensorflow as tf
import keras
from tf_model.focal_loss import BinaryFocalLoss

def xception(image_size=256):
    model = keras.models.Sequential([
        keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)),
        keras.layers.Flatten(),
        #     keras.layers.Dense(128,activation = 'relu',kernel_initializer='random_normal'),
        #     keras.layers.Dense(16,activation = 'relu',kernel_initializer='random_normal'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def inception(image_size=256):
    model = keras.models.Sequential([
        keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)),
        keras.layers.Flatten(),
        #     keras.layers.Dense(128,activation = 'relu',kernel_initializer='random_normal'),
        #     keras.layers.Dense(16,activation = 'relu',kernel_initializer='random_normal'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

if __name__ == "__main__":
    model = xception()
    model.compile(optimizer="adam", loss=BinaryFocalLoss(gamma=2), metrics=['accuracy'])
