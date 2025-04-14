# 🚀 IMDB Movie Review Sentiment Analysis with RNN

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-RNN-brightgreen)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-yellow)

## 📋 Project Overview

This project implements a **Recurrent Neural Network (RNN)** for sentiment analysis on the IMDB movie review dataset. The model classifies movie reviews as either positive or negative, demonstrating the power of RNNs in understanding sequential text data and capturing contextual information.

### 🎯 Key Features

- **Simple RNN Architecture**: Utilizes a basic RNN architecture with embedding layer for efficient text processing
- **Interactive Web Interface**: Streamlit app for real-time sentiment prediction on user-provided movie reviews
- **High Accuracy**: Achieves ~80% validation accuracy on the IMDB dataset
- **Production-Ready**: Trained model saved in H5 format for easy deployment

## 🧠 Model Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 500, 128)          1280000

 simple_rnn (SimpleRNN)      (None, 128)               32896

 dense (Dense)               (None, 1)                 129

=================================================================
Total params: 1,313,025 (5.01 MB)
Trainable params: 1,313,025 (5.01 MB)
Non-trainable params: 0 (0.00 Byte)
```

The model consists of:
1. **Embedding Layer**: Converts words to dense vectors of fixed size (128 dimensions)
2. **SimpleRNN Layer**: Processes sequential data with ReLU activation
3. **Dense Layer**: Single output neuron with sigmoid activation for binary classification

## 🛠️ Technical Implementation

- **Data Preprocessing**:
  - Tokenization of text data
  - Padding sequences to uniform length (500 words)
  - Vocabulary size limited to 10,000 most frequent words

- **Training Strategy**:
  - Binary cross-entropy loss function
  - Adam optimizer
  - Early stopping to prevent overfitting
  - Batch size of 32 for efficient training

- **Deployment**:
  - Streamlit web application for user interaction
  - Pre-trained model loaded for inference
  - Text preprocessing pipeline for user input

## 📊 Performance

The model achieves:
- **Training Accuracy**: ~73%
- **Validation Accuracy**: ~65%

This demonstrates the model's ability to generalize well to unseen data while maintaining high performance on the training set.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Web App

```bash
streamlit run main.py
```

### Using the Model Programmatically

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the model (batch_64 model is used in the web app for optimal performance)
model = load_model('simple_rnn_imdb_batch_64.h5')

# Preprocess your text (see main.py for implementation details)
# ...

# Make prediction
prediction = model.predict(preprocessed_text)
sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
```

## 📚 Project Structure

- `rnn_model_training.ipynb`: Model training and evaluation notebook
- `embedding.ipynb`: Exploration of word embeddings
- `prediction.ipynb`: Testing the model on new data
- `main.py`: Streamlit web application
- `requirements.txt`: Required dependencies
- `simple_rnn_imdb_batch_64.h5`: Pre-trained model with batch size 64 - used in the web app

## 🔍 Why This Matters

Sentiment analysis has numerous real-world applications:
- **Customer Feedback Analysis**: Automatically categorize customer reviews
- **Social Media Monitoring**: Track brand sentiment across platforms
- **Market Research**: Analyze public opinion on products or services
- **Content Recommendation**: Enhance recommendation systems with sentiment data

## 🔮 Future Improvements

- Implement bidirectional RNNs for better context understanding
- Explore LSTM and GRU architectures for improved performance
- Add attention mechanisms to focus on important parts of reviews
- Incorporate transfer learning with pre-trained language models

---

## 👨‍💻 About the Developer

This project demonstrates proficiency in:
- Deep learning with TensorFlow
- Natural Language Processing (NLP)
- Recurrent Neural Networks
- Model deployment with Streamlit
- End-to-end ML project implementation

Feel free to reach out for collaboration or questions!
