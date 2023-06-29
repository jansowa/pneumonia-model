from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from matplotlib import image
import random
from keras.utils import np_utils
from keras import callbacks

# batch_size = higher batch size could be better, but I haven't got enough gpu memory
batch_size = 32
# epochs = 50
epochs = 20

train_dir = 'chest_xray/chest_xray/train/'
val_folder = 'chest_xray/chest_xray/val/'
test_folder = 'chest_xray/chest_xray/test/'

pneumonia_str = "PNEUMONIA/"
normal_str = "NORMAL/"

train_p_dir = train_dir + pneumonia_str
train_n_dir = train_dir + normal_str

val_p_dir = val_folder + pneumonia_str
val_n_dir = val_folder + normal_str

test_p_dir = test_folder + pneumonia_str
test_n_dir = test_folder + normal_str

filenames = tf.io.gfile.glob(str('chest_xray/chest_xray/train/*/*'))
filenames.extend(tf.io.gfile.glob(str('chest_xray/chest_xray/val/*/*')))

# train/NORMAL:
# IM-XXXX-0001.jpeg
# NORMAL2-IM-XXXX-0001.jpeg

# train/PNEUMONIA:
# personX-bacteria_1.jpeg
# personX_virus_201.jpeg

print("filenames:")
print(filenames)
print("type(filenames):")
print(type(filenames))

train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)

COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len([filename for filename in train_filenames if "PNEUMONIA" in filename])
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))

train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

# Print few file names
for f in train_list_ds.take(5):
    print(f.numpy())


# Check if there is good ratio between train and val photos number
TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))
VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))


CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                        for item in tf.io.gfile.glob(str("chest_xray/chest_xray/train/*"))])
print(CLASS_NAMES)

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == "PNEUMONIA"

IMAGE_SIZE = [180, 180]
input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def process_path_img_only(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
X_train = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_list_ds = tf.data.Dataset.list_files(str('chest_xray/chest_xray/test/*/*'))
TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(batch_size)

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)
# test_ds = prepare_for_training(test_ds)

# COPY FROM CIFAR-10
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# print("jed")
# X_train = np.concatenate([x for x, y in train_ds], axis=0)
# print("dwa")
# y_train = np.concatenate([y for x, y in train_ds], axis=0)
# print("trz")
#
# X_val = np.concatenate([x for x, y in val_ds], axis=0)
# print("czt")
# y_val = np.concatenate([y for x, y in val_ds], axis=0)
# print("pie")
#
# img_rows = IMAGE_SIZE[0]
# img_cols = IMAGE_SIZE[1]
#
# if K.image_data_format() == 'channels_first':
#     X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
#     X_val = X_val.reshape(X_val.shape[0], 3, img_rows, img_cols)
#     input_shape = (3, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
#     X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 3)
#     input_shape = (img_rows, img_cols, 3)

# train_ds = (X_train, y_train)
# val_ds = (X_val, y_val)

model = Sequential()

# https: // datascience.stackexchange.com / questions / 102483 / difference - between - relu - elu - and -leaky - relu - their - pros - and -cons - majorly
# relu -> x if x>0 else 0
# leaky relu -> x if x>0 else 0.01x
# elu -> x if x>0 else 0.01 * (exp(x) - 1)
activation_function = "leaky_relu"
if activation_function == "leaky_relu":
    activation_function = keras.layers.LeakyReLU(alpha=0.01)

# https://stackoverflow.com/questions/36243536/what-is-the-number-of-filter-in-cnn
# "filters" param sets the maximum number of filters in layer
# each filter creates new feature map (much higher complexity!)
conv_two_exp = 4
first_layer_filters = 2 ** conv_two_exp
kernel_unit = 3
kernel_size = (kernel_unit, kernel_unit)
kernel_initializer = 'he_uniform'

model.add(Conv2D(first_layer_filters, kernel_size=kernel_size,
                     activation=activation_function,
                     input_shape=input_shape, kernel_initializer = kernel_initializer))
# https://www.baeldung.com/cs/batch-normalization-cnn
# BatchNormalization - normalizes data between mini-batches. It allows using higher learning rate.
# Substract mean of neurons output and divide by standard deviation.
# "each feature map will have a single mean and standard deviation, used on all the features it contains"
model.add(BatchNormalization())
model.add(Conv2D(first_layer_filters, kernel_size, activation=activation_function, kernel_initializer = kernel_initializer))
model.add(BatchNormalization())
dropout1 = 0.5
model.add(Dropout(dropout1))

# 2x more filters in each next Conv2D layer
model.add(Conv2D(first_layer_filters*2, kernel_size, activation=activation_function, kernel_initializer = kernel_initializer))

# MaxPooling2D Reduces number of trainable parameters
model.add(MaxPooling2D(pool_size=(2, 2)))
dropout2 = 0.5
model.add(Dropout(dropout2))
model.add(Conv2D(first_layer_filters*4, kernel_size, activation=activation_function, kernel_initializer = kernel_initializer))
model.add(BatchNormalization())
model.add(Conv2D(first_layer_filters*8, kernel_size, activation=activation_function, kernel_initializer = kernel_initializer))
model.add(BatchNormalization())
dropout3 = 0.5
model.add(Dropout(dropout3))
model.add(Flatten())
dense_two_exp = 4
dense_units = 2 ** dense_two_exp
model.add(Dense(dense_units, activation=activation_function))
model.add(BatchNormalization())
dropout4 = 0.5
model.add(Dropout(dropout4))
model.add(Dense(dense_units/4, activation=activation_function))
model.add(BatchNormalization())
dropout5 = 0.5
model.add(Dropout(dropout5))
model.add(Dense(1, activation='sigmoid'))

# Adam optimizer tunes learning rate in next epochs
learning_rate = 0.0009766910468362844

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=METRICS)

# datagen = ImageDataGenerator(
#          rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
#          zoom_range=0.2851945624723854, # Randomly zoom image
#          shear_range=0.42664072982759615,# shear angle in counter-clockwise direction in degrees
#          width_shift_range=0.14234862104307688,  # randomly shift images horizontally (fraction of total width)
#          height_shift_range=0.12363487226276715,  # randomly shift images vertically (fraction of total height)
#          vertical_flip=False,
#          horizontal_flip=True)  # randomly flip images
#
# datagen.fit(X_train)

checkpoint_filepath = "xray_model.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                   save_best_only=True, monitor="val_accuracy", mode="max")
earlystopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                            # mode="max", patience=20,
                                            mode="max", patience=100,
                                            restore_best_weights=True)
print(model.summary())

initial_bias = np.log([COUNT_PNEUMONIA/COUNT_NORMAL])
weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0
weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0
class_weight = {0: weight_for_0, 1: weight_for_1}

# hist = model.fit(datagen.flow(train_ds, batch_size = batch_size),
hist = model.fit(train_ds, batch_size = batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=val_ds,
                     callbacks=[checkpoint_cb, earlystopping],
                    steps_per_epoch=TRAIN_IMG_COUNT // batch_size,
                    validation_steps=VAL_IMG_COUNT // batch_size)
                 # class_weight=class_weight)


model.load_weights(checkpoint_filepath)
loss, acc, prec, rec = model.evaluate(test_ds)
# score = model.evaluate(val_ds, verbose=0) # TODO: use test_ds
print('Test loss:', loss)
print('Test accuracy:', acc)
# 0.766 without class weights 180x180
#  without class weights 180x180 and deleted duplications in "normal"
# 0.764 with class weights 180x180
# 0.769 with class weights 256x256
# 0.700 without class weights 256x256
print('Test prec:', prec)
print('Test rec:', rec)