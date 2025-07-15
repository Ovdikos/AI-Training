# AI-Training

This repository contains several Python projects implementing classic AI and machine learning algorithms **from scratch**. Each project includes a simple command-line interface and is easy to run with your own datasets.

---

## Projects Overview

### 1. KNN Classifier
Implementation of the **k-Nearest Neighbors (k-NN) classification algorithm** for machine learning tasks.  
The model can be trained on your own data, classify new samples, and evaluate accuracy on a test set.

**How to run:**
```bash
python K-NN/k-NN.py 3 K-NN/iris.data K-NN/iris.test.data
```

### 2. Language Detection Neural Network (Single Layer)
A simple single-layer neural network for automatic language detection.  
The model is trained on a set of texts in different languages and predicts the language of new phrases by analyzing letter frequencies.

**How to run:**
```bash
cd Single_Layer
python single_layer.py
```

### 3. Naive Bayes Classifier
A Naive Bayes classifier for determining mushroom edibility based on their features.  
The model analyzes data and automatically predicts whether a mushroom is edible or poisonous using the classic Naive Bayes approach.

**How to run:**
```bash
cd Naive_Bayes
python Naive_Bayes.py
```

### 4. Perceptron for Binary Classification
An implementation of the perceptron algorithm for binary classification tasks.  
The model learns to separate two classes based on numerical features and provides an interactive mode for real-time testing.

```bash
cd Perceptron
python Perceptron.py
```

Feel free to explore each project, experiment with your own data, and modify the code to deepen your understanding of fundamental machine learning algorithms.
