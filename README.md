# **MNIST Digit Classifier with Custom Neural Network**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Model Type: DistilBERT](https://img.shields.io/badge/Model-DistilBERT-green)](https://huggingface.co/distilbert-base-uncased)

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Model Performance](#model-performance)
- [License](#license)

---

## **Overview**

The Product Entity Value Extractor is a machine learning model that extracts product-specific entity values such as weight, dimensions, and other attributes from images. The model processes product images and outputs predicted values with units, ensuring accurate conversion and mapping to predefined units.

## **Features**
Entity Extraction: Extracts attributes like weight, dimensions, and volume from product images.
Unit Conversion: Automatically converts and normalizes values to predefined units (e.g., cm, g, kg).
Image Processing: Uses computer vision techniques to preprocess images and enhance feature extraction.
High Precision: Model is evaluated using the F1 score and passes sanity checks for unit correctness.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/product-entity-extractor.git
cd product-entity-extractor

```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt


```
### **3. Set up Preprocessing and Models**
```bash
from sklearn.model_selection import train_test_split
from some_cv_library import ImageProcessor

# Add any additional setup for image preprocessing and model preparation


```



## **Usage**

### **1. Command Line Interface (CLI)**
```bash
python extract_entities.py --image_path "path/to/product_image.jpg" --output "output.csv"


```
## **2.Graphical User Interface (GUI)**


```bash
python mnist_classifier.py

```

## **Model Training**

## **2.Graphical User Interface (GUI)**

```bash
python train_entity_extractor.py --epochs 5 --batch_size 16 --lr 1e-4

```

## **Model Performance**
```bash

Metric               | Value
---------------------|--------
F1 Score             | 0.87
Accuracy (Sanity Check) | 95%


```



### **License**
```bash
This project is licensed under the MIT License. See the LICENSE file for more details.

```
