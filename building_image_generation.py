import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Sequential

# Load CIFAR-10 dataset
(x_train, _), (_, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0

# Define the generator model architecture
def build_generator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))

    return model

# Build and compile the generator model
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Train the generator model (example)
# Replace this with your actual training code
# generator.fit(x_train, epochs=10, batch_size=32)
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Sequential

# Load CIFAR-10 dataset
(x_train, _), (_, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0

# Define the generator model architecture
def build_generator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))

    return model

# Build and compile the generator model
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Train the generator model (example)
# Replace this with your actual training code
# generator.fit(x_train, epochs=10, batch_size=32)
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Sequential

# Load CIFAR-10 dataset
(x_train, _), (_, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0

# Define the generator model architecture
def build_generator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))

    return model

# Build and compile the generator model
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Train the generator model (example)
# Replace this with your actual training code
# generator.fit(x_train, epochs=10, batch_size=32)
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Sequential

# Load CIFAR-10 dataset
(x_train, _), (_, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0

# Define the generator model architecture
def build_generator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))

    return model

# Build and compile the generator model
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Train the generator model (example)
# Replace this with your actual training code
# generator.fit(x_train, epochs=10, batch_size=32)
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Sequential

# Load CIFAR-10 dataset
(x_train, _), (_, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0

# Define the generator model architecture
def build_generator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))

    return model

# Build and compile the generator model
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Train the generator model (example)
# Replace this with your actual training code
# generator.fit(x_train, epochs=10, batch_size=32)
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Sequential

# Load CIFAR-10 dataset
(x_train, _), (_, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0

# Define the generator model architecture
def build_generator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))

    return model

# Build and compile the generator model
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Train the generator model (example)
# Replace this with your actual training code
# generator.fit(x_train, epochs=10, batch_size=32)
