# Basic-Galaxy-Morphology-CNN
# Galaxy Morphology Classification

This repository provides an implementation of a Convolutional Neural Network (CNN) for classifying galaxy morphologies using the Galaxy10 dataset. The project utilizes TensorFlow for building and training the model and follows standard preprocessing steps to ensure efficient data handling and model performance.

## Table of Contents
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)

---

## Dataset
The project uses the **Galaxy10 dataset** available [here](https://astronn.readthedocs.io/en/latest/galaxy10.html). This dataset contains:
- **21,785 galaxy images**
- **10 classes** of galaxy morphologies
- Each image is of size **69x69 pixels** with **3 color channels (RGB)**.

## Dependencies
The following Python libraries are required to run the code:
- `TensorFlow`
- `h5py`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install these dependencies using:
```bash
pip install tensorflow h5py numpy matplotlib scikit-learn
```

## Data Preprocessing
1. **Normalizing Images**: Pixel values are scaled to the range [0, 1] by dividing by 255.
2. **Splitting Data**: The dataset is divided into **training (80%)** and **validation (20%)** subsets using `train_test_split`.
3. **Data Augmentation**: The training data is augmented using random transformations like rotation, shifting, and flipping to improve model generalization.

## Model Architecture
The CNN model is built using TensorFlow's Keras API:
1. **Convolutional Layers**: Extract features from images using 3x3 filters.
2. **Max Pooling Layers**: Reduce feature map size while retaining essential information.
3. **Flatten and Dense Layers**: Map extracted features to the 10 output classes.

### Summary of the Model:
- Input: Images resized to **64x64x3**
- Architecture: Two convolutional layers, followed by max-pooling, flattening, and fully connected dense layers.
- Output: Probabilities for 10 classes using a softmax activation.

## Training
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical cross-entropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 32

Training is performed using the augmented data generator for the training and validation datasets.

## Results
A plot showing training and validation accuracy is generated to visualize the model's performance. The CNN demonstrates its ability to classify galaxies effectively, as evidenced by improving accuracy over epochs.

## Usage
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd Galaxy-Morphology-Classification
   ```
2. Ensure the Galaxy10 dataset is downloaded and available at `Galaxy10.h5`.
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Galaxy_Morphology_Classification.ipynb
   ```

---

## Acknowledgments
- The **[astroNN library](https://github.com/henrysky/astroNN)** for making datasets and tools accessible.
- The developers of the **Galaxy10 dataset** for contributing to open research in astrophysics.

---

Feel free to modify the sections based on specific adjustments or results from your implementation!
