# Stat486MiniProject

This project aims to classify audio files into 10 music genres using an variety of machine learning methods. 

[GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification): The "MNIST of audio data"

## Contents

- `README.md`: This file
- `compression.ipynb`: Notebook used for feature engineering compression ratio.
- `Ensemble.py`: Runs every traditional model and an ensemble of them and reports estimated accuracy.
- `Final_Report.ipynb`: Summary of our findings and conclusions.
- `eda.ipynb`: Exploratory data analysis on tabular data
- `scott_transformer_tf.py`: Transformer code using TensorFlow and Keras
- `scott_transformer_torch.py`: Transformer code using PyTorch and HuggingFace
- `CNN/CNN.ipynb`: Preliminary CNN code to explore various techniques; `main.ipynb` is the updated version of `CNN.ipynb`
- `CNN/main.ipynb`: Finalized neural network code operating on spectrograms
- `CNN/main.py`: Identical to `main.ipynb`, but optimized to run as a batch job

## Motivation
- Can we classify the genre of music tracks from audio files?
- Create traditional ensemble to predict genre from .wav file and spectrograms
- Create CNN to predict genre from .wav file and spectrograms
- Can we solve the same problem with a transformer?
- Explore genre patterns with compression techniques
- Evaluating via accuracy metric b/c classes perfectly balanced

## Results and challenges
- Achieved decent, but not excellent, classification results
- Had to learn nuances of processing audio files and spectrograms
- Difficult to perform EDA on image/audio data
- Had to pay careful attention to avoid data leakage in transfer learning; dataset is well known and many base models had already trained on it
- Transformers proved extremely difficult to implement effectively

## Traditional ensemble: Random Forest, KNN, SVM, XGBoost

- **Random Forest**: 56% out-of-sample accuracy. Had the hardest time classifying Reggae and Jazz.
- **K-Nearest Neighbors**: 45% out-of-sample accuracy. Struggled to classify reggae.
- **Support Vector Machines**: 56% out-of-sample accuracy. Had a hard time classifying Rock and Disco
- **XGBoost**: 55% out-of-sample accuracy. Had the hardest time classifying Disco and Rock.

## Deep Learning: CNN

- Stochastic gradient descent optimizer w/ 0.001 learning rate
- 500 epochs with early stopping and batch size of 32
- ReLU hidden layer activation, softmax output layer activation
- 68% out-of-sample accuracy
- [CNN Reference](https://www.kaggle.com/code/jamesslay/music-genre-classification-cnn-models/notebook)

## Exploratory: Vision Transformer

- Can we match the performance of our ensemble with a more advanced technique?
- TensorFlow + Keras
    - AdamW optimizer with learning rate 0.01 and 0.004 weight decay
    -  2 transformer layers w/ 4 heads
    - 64-dimension projection, patch size 6
    - Could not outperform random classifier
    - [TF Transformer Reference](https://keras.io/examples/vision/image_classification_with_vision_transformer/)
- PyTorch + HuggingFace
    - Transfer learning w/ [speech recognition HF transformer](https://huggingface.co/MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection) 
    - Adam optimizer w/ learning rate 0.01
    - Accuracy >20% – poor but 2x better than random 
    - [PyTorch/HF Transformer Reference](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb)
- Learned that transformers are ver7 challenging but very powerful! Would like to learn more in the future.

## Further Investigation

- Is there a way to analyze audio without first converting to spectrograms, i.e. analyzing the .wav files directly?
- Would an ensemble between our CNN and other supervised methods yield even better predictions?
- Can we learn more about transformers and use them to beat the performance of our ensemble or CNN?
