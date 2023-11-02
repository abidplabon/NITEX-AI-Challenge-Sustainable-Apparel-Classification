# NITEX AI Challenge: Sustainable Apparel Classification

## Overview
This project aims to classify sustainable apparel products using the Fashion MNIST dataset.I have chosen a VGG-like architecture for the Fashion MNIST dataset because it strikes a balance between model complexity and performance. It's particularly well-suited for image classification tasks, benefits from transfer learning, and is efficient for practical use. The model choice aligns with the dataset's characteristics and the challenge requirements



### Model Development
- Developed a deep learning model (tinyVGG) for apparel classification.
- tinyVGG model is designed to process grayscale images (1 input channel) and produce classification scores for a specific number of classes defined by output_shape. It consists of convolutional layers for feature extraction and a classifier for making class predictions. 
- Trained the model on the training data.


## Instructions
1. Set up a Python virtual environment.
    - create virtual environment naming 'venv' = python -m venv venv
    - activate venv = venv\Scripts\activate
2. Install the required dependencies from `requirements.txt`.
3. Run `evaluate_model.py` with the path to the test dataset folder as an argument.

## Author
Md.Abid Ahammed

