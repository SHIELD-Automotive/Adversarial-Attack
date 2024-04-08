# Adversarial-Attack

This repository contains example codes demonstrating the training of Convolutional Neural Networks (CNNs) on the CIFAR-10 and MNIST datasets using Keras, along with implementations of adversarial attacks (Query and Transfer attacks) on these models.


**Contents:**

mnist_cnn.py - A script for training a CNN model on the MNIST dataset and applying query and transfer attacks.
cifar10_cnn.py - A script for training a CNN model on the CIFAR-10 dataset and applying query and transfer attacks.


**Each script includes:**

Code to load and preprocess the dataset.

A simple CNN model suitable for the dataset.

Functions to perform query and transfer attacks on the trained model.

Visualization of original and adversarial examples before and after attacks.


**Overview of the Adversarial Attacks:**

**Query Attack:** This attack introduces small perturbations to the input image, aiming to fool the model while keeping the changes imperceptible to the human eye.

**Transfer Attack:** Generates adversarial examples using one model (source) and tests their effectiveness on another model (target). This demonstrates the potential for adversarial examples to affect multiple models.

**Note**
The codes are intended for educational purposes to demonstrate adversarial attacks on machine learning models. They are not optimized for production use.
