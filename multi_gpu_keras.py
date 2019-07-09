import argparse
import time
import os

import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model

import wandb
from wandb.keras import WandbCallback

# training config
#--------------------------------
img_width, img_height = 299, 299

def load_optimizer(optimizer, learning_rate):
  optimizer_path = "optimizers." + optimizer
  optimizer_module = __import__(optimizer_path, fromlist=[''])
  return optimizer_module(lr=learning_rate)  

def build_core_model(dropout, num_classes):
  """ Build core 7-layer model """
  if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
  else:
    input_shape = (img_width, img_height, 3)

  model = Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=input_shape, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(dropout))
  model.add(Dense(num_classes, activation='softmax'))
  return model

def build_parallel_gpu_model(args):
  """ Build a parallel model over several gpus """
  with tf.device('/cpu:0'):
    model = build_core_model(args.dropout, args.num_classes)

  optimizer = load_optimizer(args.optimizer, args.learning_rate)
  # log model here before we parallelize it
  log_model_params(model, args)

  parallel_model = multi_gpu_model(model, gpus=gpus)
  parallel_model.compile(loss='categorical_crossentropy',
                optimizer=lr_optimizer,
                metrics=['accuracy'])
  print(parallel_model.summary())
  return parallel_model

def build_single_gpu_model(args):
  """ Build a regular model """
  model = build_core_model(args.dropout, args.num_classes)

  lr_optimizer = load_optimizer(args.optimizer, args.learning_rate)
  model.compile(loss='categorical_crossentropy',
                optimizer=lr_optimizer,
                metrics=['accuracy'])
  print(model.summary())
  return model

def build_distributed_model(args):
  if args.gpus == 1:
    model = build_core_model(args)
  else:
    with tf.device('/cpu:0'):
      core_model = build_core_model(args)
    model = multi_gpu_model(core_model, gpus=args.gpus)
  
  lr_optimizer = load_optimizer(args.optimizer, args.learning_rate)
  model.compile(loss='categorical_crossentropy',
                optimizer=lr_optimizer,
                metrics=['accuracy'])
  print(model.summary())
  return model

def log_params(args):
  """ Extract params of interest about the model (e.g. number of different layer types).
      Log these and any experiment-level settings to wandb """
  wandb.config.update({
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "n_train" : args.num_train,
    "n_valid" : args.num_valid,
    "optimizer" : args.optimizer,
    "dropout" : args.dropout,
    "GPU" : args.distrib
  })

def train_multi_gpu_model(args):

  wandb.init(project="estuary")
   # data generator from Keras finetuning tutorial
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)
  
#  if args.distrib == 1:
    # this is not a distributed model
#    model = build_single_gpu_model(args)
#    log_model_params(model, args)
#  else:
#    model = build_parallel_gpu_model(args.optimizer, arg None, wandb.config, args)
  model = build_distributed_model(args)
  log_params(args)

  train_generator = train_datagen.flow_from_directory(
    args.train_data,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  validation_generator = test_datagen.flow_from_directory(
    args.val_data,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  callbacks = [WandbCallback()]

  model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.epochs,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  #save_model_filename = args.model_name + ".h5"
  #model.save_weights(save_model_filename)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  #############################
  # Frequently updated params
  #----------------------------
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default="",
    help="Name of this model/run (model will be saved to this file)")
  #############################
  # Mostly fixed params
  #--------------------------
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=64,
    help="Batch size")
  parser.add_argument(
    "-c",
    "--num_classes",
    type=int,
    default=10,
    help="Number of classes to predict")
  parser.add_argument(
    "-d",
    "--dropout",
    type=float,
    default=0.3,
    help="Dropout for last fc layer")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=10,
    help="Number of training epochs")
  parser.add_argument(
    "-o",
    "--optimizer",
    type=str,
    default="adam",
    help="Learning optimizer")
  parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate for default optimizer")
 parser.add_argument(
    "-nt",
    "--num_train",
    type=int,
    default=10000,
    help="Number of training examples per class")
  parser.add_argument(
    "-nv",
    "--num_valid",
    type=int,
    default=2000,
    help="Number of validation examples per class")
   parser.add_argument(
    "-t",
    "--train_data",
    type=str,
    default="/mnt/train-data/inaturalist_12K/train",
    help="Absolute path to training data")
  parser.add_argument(
    "-v",
    "--val_data",
    type=str,
    default="/mnt/train-data/inaturalist_12K/val",
    help="Absolute path to validation data")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "-g",
    "--gpus",
    type=int,
    default=0,
    help="Distribute run over this many GPUs"
  )

  args = parser.parse_args()
  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # create run name
  if not args.model_name:
    print("warning: no run name provided")
    args.model_name = "model"
  else:
    os.environ['WANDB_DESCRIPTION'] = args.model_name

  train_multi_gpu_model(args)
