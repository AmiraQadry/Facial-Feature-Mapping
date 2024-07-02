# Facial Feature Mapping (CNN)
![](https://i.ytimg.com/vi/fsAPfjDS4cM/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLBdh6m6Qo9lUhdz1-O863GV7pzPmw)

This project focuses on developing a Convolutional Neural Network (CNN) model for accurate facial keypoint detection, localizing 15 key facial landmarks. 
The project is part of the Intern Infotech Virtual Learning Internship Program.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Data Augmentation](#data-augmentation)
- [Evaluation](#evaluation)
- [Results](#results)

## Project Overview

Facial keypoint detection is an essential task in computer vision, used in applications such as facial recognition, emotion detection, and augmented reality. 
This project uses CNNs to accurately detect 15 key facial landmarks.

## Dataset

The dataset consists of images of faces with 15 annotated keypoints. Each image is accompanied by a set of coordinates representing the keypoints.

## Usage

### Data Augmentation

Data augmentation is applied to enhance the dataset diversity:

```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)
```

### Evaluation

Evaluate the model on the validation set:

```python
loss, mae = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation MAE: {mae}')
```

## Model Architecture

The CNN architecture consists of several convolutional layers followed by dense layers to regress the keypoint coordinates. 

## Training

Training is conducted using the Adam optimizer and Mean Absolute Error (MAE) loss function. 
The model is trained for 50 epochs with data augmentation.

```python
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=50)
```

## Data Augmentation

Data augmentation is employed to increase the variability of the training data and improve the model's generalization. 
The augmentation techniques include rotation, width and height shifts, zoom, and horizontal flips.

## Evaluation

The model's performance is evaluated using the validation loss and MAE.

## Results

The model achieves a validation MAE of around 0.13-0.15.
