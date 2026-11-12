


# 🐾 Deep Learning Based Animal Recognition System

## Overview

The Deep Learning Based Animal Recognition System is an
AI-powered image classification application that identifies
objects and animal categories from uploaded images.

The application uses a pretrained ResNet-50 deep learning
architecture trained on the ImageNet dataset.

Users can upload an image and receive the top five
predicted classes with their confidence scores.

## Features

- Upload animal images.
- Automatic image preprocessing.
- Deep learning-based image classification.
- Top 5 prediction results.
- Confidence score visualization.
- Interactive Gradio interface.
- GPU inference using Hugging Face ZeroGPU.

## Technologies

- Python
- PyTorch
- ResNet-50
- Hugging Face Transformers
- Gradio
- Pillow

## Model Architecture

ResNet-50 is a 50-layer Convolutional Neural Network
(CNN) based on residual learning.

The model uses residual connections to improve the
training of deep neural networks.

## Dataset

ImageNet (pretrained model).

The pretrained model contains 1,000 image categories,
including different animal species.

## Installation

Install dependencies:

pip install -r requirements.txt

Run locally:

python app.py

## Deployment

Deploy this project using Hugging Face Spaces.

Select:
- SDK: Gradio
- Hardware: ZeroGPU
- License: MIT

## License

This project is licensed under the MIT License.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



---
title: Deep Learning Based Animal Recognition System

emoji: 🐾

colorFrom: green

colorTo: blue

sdk: gradio

sdk_version: 5.50.0

python_version: 3.10

app_file: app.py

license: mit

suggested_hardware: zero-a10g
---
