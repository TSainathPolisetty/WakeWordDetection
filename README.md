# Wake-Word Detection with TinyML
#### December 30, 2023

### Introduction
This project aims to develop a low-power, efficient audio analysis system for mental health monitoring, focusing on recognizing specific absolutist keywords like "never", "none", "all", "must", and "only".

### Experiment
- **Data Collection**: Gathered at least 50 recordings per keyword using the Open Speech Recording tool, resulting in over 250 samples. The dataset was enhanced with Pete Warden's Speech Commands dataset for diversity.
- **Data Processing**: Audio files converted to spectrograms for analysis, with padding for uniformity in dimensions.

### Algorithm
- **Model Architecture**: A Convolutional Neural Network (CNN) designed to process spectrograms.
  - Layers include Conv2D with MaxPooling, Flatten, and Dense layers, ending with a softmax output layer.
- **Training Process**: Utilized the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy metrics. The model underwent quantization for deployment on low-power devices.

### Deployment on Arduino
- **Steps**: Included model loading, interpreter setup, memory allocation, input processing, main loop execution, command recognition, and response handling.
- **Challenges**: Adjusted the micro speech example to fit the new model, focusing on compatibility and functionality.

## Demonstration - Click on the image to play video
[![Keyword Detection Demonstration](https://img.youtube.com/vi/woEzHXtRla0/maxresdefault.jpg)](https://youtu.be/woEzHXtRla0)

### Results & Analysis
- **Performance**: Achieved 96.22% training accuracy and similar test accuracy, with high class-wise accuracies for different keywords.
- **Real-time Prediction**: Displayed high accuracy in constrained environments but faced challenges with increased class complexity and background noise.

### Discussion and Summary
- **Challenges**: Managing a large number of classes and environmental noise in real-time predictions.
- **Future Resolutions**: Plans to enhance noise robustness, use advanced data augmentation, and explore more complex network architectures.
- **Real-Time Accuracy Expectations**: While real-time accuracy was lower than test accuracy, it offered insights into practical applicability and areas for enhancement.




