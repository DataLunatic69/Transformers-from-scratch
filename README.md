
# **Entity Extraction from Product Images Using Custom ML Model**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)


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

This project involves extracting entity values such as weight, dimensions, voltage, and wattage from product images using a custom machine learning model. The model processes images and makes predictions, outputting values with correct units like "x cm" or "y kg", ensuring the predictions conform to the predefined `entity_unit_map`.

## **Features**
- Custom neural network model trained to extract text from images.
- Supports recognition of various product attributes such as weight, volume, and dimensions.
- Handles unit conversion (e.g., "lbs" to "pounds", "cm" to "centimeters").
- Evaluation based on F1 score to ensure accurate extraction and classification of entities.


---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/entity-extraction-product-images.git
cd entity-extraction-product-images
```

### **2. Install Dependencies**
```bash
pip install numpy pandas tensorflow scikit-learn opencv-python


```
### **3. Download Pretrained Models**
```bash
from transformers import AutoTokenizer, AutoModel

model_ckpt = "your-model-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

```



## **Usage**

### **1. Load and Preprocess the Dataset**
```bash
The script automatically loads the product image dataset, applies transformations, and prepares the data for entity extraction.


```
## **2.Run the model**

### **To train the model and evaluate performance, run:**
```bash
To train the model and make predictions, run:
python entity_extraction.py


```
## **3.Evaluate Prediction**


```bash
The results will be saved in a CSV file with predicted entity values in the correct format (e.g., "3.5 cm").



```

## **Model Training**


```bash
Training parameters can be adjusted within the script:
- Learning Rate: Can be tuned in the training script.
- Epochs: Specify the number of training epochs.
- Dropout Rate: Controls dropout layers to prevent overfitting.
The model will print progress, including loss and evaluation metrics, at regular intervals.
.




```

## **Model Performance**
```bash

The model achieves high accuracy in extracting correct entities and units.

Metric           | Value
-----------------|--------
Test Accuracy    | 92%
F1 Score         | 0.91


```



### **License**
```bash
This project is licensed under the MIT License. See the LICENSE file for more details.


```
