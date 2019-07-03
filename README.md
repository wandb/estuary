# Distributed Training Experiments in Weights & Biases

This is a collection of distributed training experiments instrumented with Weights & Biases.

## Approaches

* basic data parallelism in Keras (multi_gpu_model)
* Tensorflow Distribtued Strategy

## Current scripts

* multi_gpu_keras.py: finetune a small CNN on 6-12K photos from iNaturalist
* finetune_experiments.py: finetune large pretrained models (Xception, ResNet, Inception) on iNaturalist

