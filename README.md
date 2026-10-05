# MNIST Handwritten Digit Recognizer

A handwritten digit recognition project built with Python, TensorFlow, Keras, NumPy, and Tkinter.

The project trains a Convolutional Neural Network (CNN) using the MNIST dataset and provides a graphical interface where the user can draw a digit from `0` to `9`.

The trained neural network processes the drawing and predicts which digit was written.

---

## Project Overview

The project has two main parts:

1. A training script that trains a neural network using the MNIST dataset.
2. A Tkinter graphical interface that allows the user to draw digits and test the trained model.

The main goal of the project is to understand the basic concepts behind image classification and neural networks.

---

## Features

- Trains a neural network using the MNIST dataset
- Uses a Convolutional Neural Network
- Recognizes handwritten digits from `0` to `9`
- Provides a graphical interface using Tkinter
- Allows drawing directly on a 28x28 grid
- Automatically preprocesses the drawn digit
- Centers and resizes the digit before prediction
- Saves the trained model as a `.keras` file
- Loads the trained model for inference
- Uses CPU or GPU depending on the TensorFlow installation

---

## Project Structure

```text
mnist-digit-recognizer/
├── app.py
├── train.py
├── model.keras
├── requirements.txt
├── README.md
└── .gitignore
