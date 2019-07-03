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

#from keras_callbacks import PerClassMetrics

# training config
#--------------------------------
# TODO: vary by base model 
img_width, img_height = 299, 299

# data location
# remote
data_remote = '/home/stacey/keras/nature_data'
# local
train_data_local = '/Users/stacey/Code/nature_data/val'
val_data_local = '/Users/stacey/Code/nature_data/train'

def build_core_model(dropout, num_classes):
  """ Build core 7-layer model """
  if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
  else:
    input_shape = (img_width, img_height, 3)

  model = Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  return model

def build_parallel_gpu_model(optimizer, dropout, num_classes, gpus=1, lr=None, config=None, args=None):
  """ Build a parallel model over several gpus """
  with tf.device('/cpu:0'):
    model = build_core_model(dropout, num_classes)

  # if learning rate is supplied, default to SGD with momentum=0.8
  if lr:
    lr_optimizer = optimizers.SGD(lr=lr, momentum=0.8)
  else:
    lr_optimizer = optimizer
  # log model here before we parallelize it
  log_model_params(model, config, args)

  parallel_model = multi_gpu_model(model, gpus=gpus)
  parallel_model.compile(loss='categorical_crossentropy',
                optimizer=lr_optimizer,
                metrics=['accuracy'])
  print(parallel_model.summary())
  return parallel_model

def build_single_gpu_model(optimizer, dropout, num_classes, lr=None):
  """ Build a regular model """
  model = build_core_model(dropout, num_classes)

  # if learning rate is supplied, default to SGD with momentum=0.8
  if lr:
    lr_optimizer = optimizers.SGD(lr=lr, momentum=0.8)
  else:
    lr_optimizer = optimizer
  model.compile(loss='categorical_crossentropy',
                optimizer=lr_optimizer,
                metrics=['accuracy'])
  print(model.summary())
  return model

def log_model_params(model, wandb_config, args):
  """ Extract params of interest about the model (e.g. number of different layer types).
      Log these and any experiment-level settings to wandb """
  num_conv_layers = 0
  num_fc_layers = 0
  for l in model.layers:
    layer_type = l.get_config()["name"].split("_")[0]
    if layer_type == "conv2d":
      num_conv_layers += 1
    elif layer_type == "dense":
      num_fc_layers += 1

  wandb_config.update({
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "n_conv_layers" : num_conv_layers,
    "n_fc_layers" : num_fc_layers,
    "img_dim" : img_width,
    "num_classes" : args.num_classes,
    "n_train" : args.num_train,
    "n_valid" : args.num_valid,
    "optimizer" : args.optimizer,
    "dropout" : args.dropout,
    "GPU" : args.distrib
  })

def run_distrib_exp(args):

  wandb.init(project="estuary")
   # data generator from Keras finetuning tutorial
  # TODO: consider modifying data augmentation strategy--more or less variety
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  if args.distrib == 1:
    # this is not a distributed model
    model = build_single_gpu_model(args.optimizer, args.dropout, args.num_classes)
    log_model_params(model, wandb.config, args)
  else:
    model = build_parallel_gpu_model(args.optimizer, args.dropout, args.num_classes, args.distrib, None, wandb.config, args)

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

# TODO: is it also just a new project...?
def curriculum_learn_exp(args):
  """ Run curriculum learning experiment, with a schedule for how many epochs to
  spend on which labels """
  wandb.init(project="keras_finetune")

  specific_train = "/mnt/data/inaturalist/curr_learn_25_s_620_100/train"
  specific_val = "/mnt/data/inaturalist/curr_learn_25_s_620_100/val"
  general_train = "/mnt/data/inaturalist/curr_learn_25_s_620_100_BY_CLASS/train"
  general_val = "/mnt/data/inaturalist/curr_learn_25_s_620_100_BY_CLASS/val"

  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  model = build_core_model(args.optimizer, args.dropout, 5, 0.025)
  log_model_params(model, wandb.config, args)

  switch_epoch = args.class_switch
  callbacks = [WandbCallback()]

  # train on first label set for switch_epochs
  train_generator = train_datagen.flow_from_directory(
    general_train,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  validation_generator = test_datagen.flow_from_directory(
    general_val,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=switch_epoch,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  # now we need to reload a model with 25 classes
  specific_model = build_model(args.optimizer, args.dropout, 25, 0.01)

  # reload everything up to dropout layer
  for i, layer in enumerate(specific_model.layers[:-3]):
    layer.set_weights(model.layers[i].get_weights())

  # train on second label set for epochs - switch_epoch
  spec_train_generator = train_datagen.flow_from_directory(
    specific_train,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  spec_validation_generator = test_datagen.flow_from_directory(
    specific_val,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  specific_model.fit_generator(
    spec_train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.epochs - switch_epoch,
    validation_data=spec_validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  save_model_filename = args.model_name + ".h5"
  specific_model.save_weights(save_model_filename)

def run_experiment(args):
  """Build data generators and model; run training"""
  wandb.init(project="keras_finetune")

  if args.local_mode:
    # reset paths to local
    args.train_data = train_data_local
    args.val_data = val_data_local

  # data generator from Keras finetuning tutorial
  # TODO: consider modifying data augmentation strategy--more or less variety
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  model = build_single_gpu_model(args.optimizer, args.dropout, args.num_classes)
  log_model_params(model, wandb.config, args)

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
  # compute per-class precision if set
  # currently this runs twice, using sklearn and DIY code, and includes the time
  # TODO: optimize our timing?
  if args.per_class:
    callbacks.append(PerClassMetrics(validation_generator, verbose=True))

  model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.epochs,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  save_model_filename = args.model_name + ".h5"
  model.save_weights(save_model_filename)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

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
    default=32,
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
    default=50,
    help="Number of training epochs")
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
    "-o",
    "--optimizer",
    type=str,
    default="adam",
    help="Learning optimizer")
  parser.add_argument(
    "-t",
    "--train_data",
    type=str,
    #default="/mnt/data/inaturalist/main_5000_800/train",
    default="/mnt/train-data/inaturalist_12K/train",
    help="Absolute path to training data")
  parser.add_argument(
    "-v",
    "--val_data",
    type=str,
    #default="/mnt/data/inaturalist/main_5000_800/val",
    default="/mnt/train-data/inaturalist_12K/val",
    help="Absolute path to validation data")
  parser.add_argument(
    "-l",
    "--local_mode",
    action="store_true",
    help="Run locally if flag is set")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "--per_class",
    action="store_true",
    help="If set, calculate per-class precision")
  parser.add_argument(
    "-cs",
    "--class_switch",
    type=int,
    default=0,
    help="Epoch on which to switch labels")
  parser.add_argument(
    "--distrib",
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

  if args.distrib:
    run_distrib_exp(args)
  elif args.class_switch:
    curriculum_learn_exp(args)
  else:
    run_experiment(args)


