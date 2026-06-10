# MNIST Digit Classification using a Feedforward Neural Network

Implementation of a simple Feedforward Neural Network (FFNN) implementation in Keras/TensorFlow for classifying handwritten digits from the MNIST dataset.

## Project Overview

The objective of this project is to build and train a basic neural network to accurately classify images of handwritten digits (0-9). The MNIST dataset is used, which is a widely recognized benchmark for image classification tasks. The model's performance is evaluated using accuracy, loss, and a confusion matrix.

## Setup and Installation

To run this project, you'll need Python and the following libraries. It is recommended to use a virtual environment.

### Prerequisites

*   Python 3.x
*   pip (Python package installer)

### Installation Steps

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install tensorflow numpy scikit-learn matplotlib seaborn
    ```

## Usage

1.  **Data Loading and Preprocessing:** The MNIST dataset is loaded using `keras.datasets.mnist.load_data()`. The pixel values are normalized to a range of 0-1 by dividing by 255.

2.  **Model Architecture:** A Sequential Keras model is defined with:
    *   A `Flatten` layer to convert the 28x28 input images into a 1D array.
    *   A `Dense` hidden layer with 128 neurons and `relu` activation.
    *   A `Dense` output layer with 10 neurons (for 10 classes) and `softmax` activation.

3.  **Model Compilation:** The model is compiled using the `adam` optimizer, `sparse_categorical_crossentropy` loss function (suitable for integer labels), and `accuracy` as the evaluation metric.

4.  **Model Training:** The model is trained on the `X_train` and `y_train` data for 100 epochs, with validation performed on `X_test` and `y_test`.

5.  **Evaluation:** After training, the model is evaluated on the test set to determine its overall performance.

6.  **Visualization:**
    *   **Heatmap Visualization of a Digit:** An example MNIST image is visualized as a heatmap using `matplotlib.pyplot.imshow` with a 'hot' colormap to display pixel intensities.
    *   **Confusion Matrix:** A confusion matrix is generated using `sklearn.metrics.confusion_matrix` and visualized as a heatmap using `seaborn.heatmap`. This provides insight into the model's classification accuracy for each digit and identifies common misclassifications.

7.  **Model Saving:** The trained model is saved in HDF5 format (`mnist_ffnn_model.h5`) for later use without needing to retrain.

## Results

The model's performance metrics (loss and accuracy) on the test set are outputted during and after training. The confusion matrix visually summarizes correct and incorrect predictions for each digit. Typically, a well-trained model on MNIST achieves an accuracy upwards of 97-98%.
