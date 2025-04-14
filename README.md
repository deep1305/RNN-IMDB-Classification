# 🚀 IMDB Movie Review Sentiment Analysis with RNN

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-RNN%20%7C%20LSTM-brightgreen)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-yellow)

## 📋 Project Overview

This project implements both a **Simple Recurrent Neural Network (RNN)** and an improved **Bidirectional LSTM** for sentiment analysis on the IMDB movie review dataset. The models classify movie reviews as either positive or negative, demonstrating the power of recurrent architectures in understanding sequential text data and capturing contextual information.

### 🎯 Key Features

- **Multiple Model Architectures**: Progression from SimpleRNN to Bidirectional LSTM
- **Pre-trained Word Embeddings**: GloVe embeddings for improved semantic understanding
- **Interactive Web Interface**: Streamlit app for real-time sentiment prediction
- **High Accuracy**: Achieves up to 85% validation accuracy on the IMDB dataset
- **Production-Ready**: Trained models saved in H5 format for easy deployment

## 🧠 Model Evolution

### Initial SimpleRNN Architecture

```python
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
```

### Improved Bidirectional LSTM Architecture

```python
model = Sequential()
model.add(Embedding(max_features, embedding_dim,
                   weights=[embedding_matrix],  # Pre-trained GloVe embeddings
                   input_length=max_len,
                   trainable=False))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
```

## 📊 Performance Comparison

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------------|---------------|
| SimpleRNN (batch_size=32) | ~94% | ~80% | ~78% |
| SimpleRNN (batch_size=64) | ~73% | ~65% | ~63% |
| Bidirectional LSTM | ~92% | ~95% | ~82% | ~82% |

The improved Bidirectional LSTM model demonstrates significant performance gains over the SimpleRNN architecture, particularly in validation and test accuracy.

### 🚀 Why the Performance Boost?

1. **Memory Capacity**: LSTM cells can retain information over longer sequences compared to SimpleRNN, which suffers from vanishing gradient problems

2. **Bidirectional Processing**: Reading text in both directions captures context more effectively than unidirectional processing
   - Example: In "The movie was not bad at all", the meaning of "bad" is affected by words both before and after it

3. **Pre-trained Embeddings**: GloVe vectors contain semantic knowledge learned from billions of words
   - Words like "excellent", "amazing", and "fantastic" are already mapped to similar vectors
   - Helps with words that appear infrequently in the training data

4. **Regularization**: Dropout prevents the model from memorizing the training data, leading to better generalization

5. **Adaptive Learning**: Learning rate scheduling helps the model converge to better minima by reducing the step size when needed

6. **Deeper Architecture**: Multiple stacked layers allow the model to learn hierarchical features in the text

## 🛠️ Technical Implementation

### Data Preprocessing

- **Tokenization**: Converting text to sequences of integers
- **Padding**: Ensuring uniform sequence length (500 words)
- **Vocabulary**: Limited to 10,000 most frequent words
- **Text Cleaning**: Removing HTML tags, special characters, and extra spaces

### Key Improvements in the Advanced Model

1. **Bidirectional LSTM**: Processes sequences in both directions for better context understanding
2. **Dropout Layers**: Prevents overfitting by randomly deactivating neurons during training
3. **GloVe Embeddings**: Pre-trained word vectors that capture semantic relationships
4. **Learning Rate Scheduling**: Reduces learning rate when performance plateaus
5. **Advanced Text Preprocessing**: More thorough cleaning of input text

### Training Strategy

```python
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Learning rate reduction when performance plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.0001
)

# Model training with callbacks
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=15,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Web App

```bash
streamlit run main.py
```

### Using the Improved Model Programmatically

```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import re

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the improved model
model = load_model('improved_lstm_imdb.h5')

# Function to preprocess text
def preprocess_text(text):
    # Clean the text
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces

    # Tokenize and convert to sequence
    words = text.split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)

    return padded_review

# Example usage
sample_review = "This movie was fantastic! The acting was superb."
preprocessed_input = preprocess_text(sample_review)
prediction = model.predict(preprocessed_input)
sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {prediction[0][0]}')
```

## 📚 Project Structure

- `rnn_model_training.ipynb`: Initial SimpleRNN model training and evaluation
- `improved_rnn_model.ipynb`: Advanced Bidirectional LSTM model with GloVe embeddings
- `embedding.ipynb`: Exploration of word embeddings
- `prediction.ipynb`: Testing the models on new data
- `main.py`: Streamlit web application
- `requirements.txt`: Required dependencies
- `simple_rnn_imdb.h5`: SimpleRNN model with batch size 32 (~80% validation accuracy)
- `simple_rnn_imdb_batch_64.h5`: SimpleRNN model with batch size 64 (~65% validation accuracy)
- `improved_lstm_imdb.h5`: Bidirectional LSTM model with GloVe embeddings (~82% validation accuracy)

## 🔍 Why This Matters

Sentiment analysis has numerous real-world applications:
- **Customer Feedback Analysis**: Automatically categorize customer reviews
- **Social Media Monitoring**: Track brand sentiment across platforms
- **Market Research**: Analyze public opinion on products or services
- **Content Recommendation**: Enhance recommendation systems with sentiment data

## 🔮 Future Improvements

- Implement transformer-based architectures (BERT, RoBERTa)
- Add attention mechanisms to focus on important parts of reviews
- Incorporate transfer learning with larger pre-trained language models
- Develop multi-class sentiment analysis (beyond binary positive/negative)
- Create ensemble models combining different architectures

---

## 👨‍💻 About the Developer

Hi, I'm Deep, a passionate Machine Learning Engineer with a strong interest in Natural Language Processing and Deep Learning architectures. This project represents my journey in exploring and improving RNN-based models for sentiment analysis.

I'm particularly interested in how different neural network architectures can be optimized for specific NLP tasks, and I enjoy the process of iteratively improving models to achieve better performance.

### Connect With Me
- **GitHub**: [deep1305](https://github.com/deep1305)

I'm always open to collaboration, feedback, or discussions about machine learning and AI. Feel free to reach out if you have questions about this project or if you're interested in working together on future projects!
