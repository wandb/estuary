# multi_gpu_keras.py
#----------------------
# training a simple 7-layer CNN with data parallelism in Keras,
# using the built-in multi_gpu_model() function, instrumented
# with wandb

import argparse
import time
import os
import sys

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import multi_gpu_model

import wandb
from wandb.keras import WandbCallback

# training config
#--------------------------------
img_width, img_height = 299, 299

# utils
#--------------------------------
def load_class_from_module(module_name):
  components = module_name.split('.')
  mod = __import__(components[0])
  for comp in components[1:]:
    mod = getattr(mod, comp)
  return mod

def load_optimizer(optimizer, learning_rate):
  """ Dynamically load relevant optimizer """
  optimizer_path = "tensorflow.keras.optimizers." + optimizer
  optimizer_module = load_class_from_module(optimizer_path)
  return optimizer_module(lr=learning_rate)  

def log_params(args):
  """ Extract params of interest about the model. Log these and any 
  experiment-level settings to wandb. Note that the model definition
  will change after the model is distributed across multiple GPUs--log
  any parameters specific to the core model or any architectural
  details (e.g. the number of convolutional layers or fully-connected
  layers) before parallelizing the model with multi_gpu_model() """
  wandb.config.update({
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "n_train" : args.num_train,
    "n_valid" : args.num_valid,
    "optimizer" : args.optimizer.lower(),
    "dropout" : args.dropout,
    "lr" : args.learning_rate,
    "GPU" : args.gpus
  })

# model definition & training
#--------------------------------
def build_core_model(args):
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
  model.add(Dropout(args.dropout))
  model.add(Dense(args.num_classes, activation='softmax')) 
  print(model.summary())
  return model

def build_distributed_model(args):
  """ Load core model and optionally parallelize across the given number of gpus""" 
  if args.gpus == 1:
    # for 1 GPU, simply load the core model
    model = build_core_model(args)
    core_model = model
  else:
    # for >1 GPU, load the core model on CPU and then distribute via Keras
    with tf.device('/cpu:0'):
      core_model = build_core_model(args)
    model = multi_gpu_model(core_model, gpus=args.gpus)
  
  lr_optimizer = load_optimizer(args.optimizer, args.learning_rate)
  model.compile(loss='categorical_crossentropy',
                optimizer=lr_optimizer,
                metrics=['accuracy'])
  print(model.summary())
  # keep a reference to the original/core model to save it correctly at the end
  return model, core_model

def train_multi_gpu_model(args):
  """ Train a data-parallel model across multiple gpus """ 
  wandb.init(project="estuary")
  
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)
  
  parallel_model, core_model = build_distributed_model(args)
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

  parallel_model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.epochs,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  # to correctly save the weights/model after data parallel training,
  # we actually need to save the original/core model
  # save_model_filename = args.model_name + ".h5"
  # core_model.save_weights(save_model_filename)

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
  parser.add_argument(
    "-g",
    "--gpus",
    type=int,
    default=0,
    help="Distribute run over this many GPUs"
  )

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
    default="RMSprop",
    help="Learning optimizer (match Keras package name exactly")
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
    default="/mnt/disks/datasets/inaturalist_12K/train",
    help="Absolute path to training data")
  parser.add_argument(
    "-v",
    "--val_data",
    type=str,
    default="/mnt/disks/datasets/inaturalist_12K/val",
    help="Absolute path to validation data")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
 
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
