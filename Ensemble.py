# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD
import os
import glob
import numpy as np

# Read CSV Data
df = pd.read_csv('./data/features_30_sec.csv')
compression_df = pd.read_csv('./data/compression_feature.csv')
merged_df = pd.merge(df, compression_df, on='filename')

# Split the CSV data for each genre
unique_genres = merged_df['label'].unique()
train_data = []
validation_data = []

for genre in unique_genres:
    genre_data = merged_df[merged_df['label'] == genre]
    train_genre, val_genre = train_test_split(genre_data, test_size=20, shuffle=False)
    train_data.append(train_genre)
    validation_data.append(val_genre)

train_data = pd.concat(train_data)
validation_data = pd.concat(validation_data)

# Create train and test splits for X and y
X_train = train_data.drop(columns=['label', 'filename'])
y_train = train_data['label']
X_test = validation_data.drop(columns=['label', 'filename'])
y_test = validation_data['label']

print(y_train.unique())
print(y_test.unique())

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Read Spectrogram Data
def create_prediction_dataset(source, genres, image_count=20, batch_size=32):
    img_size = (288, 432)
    image_paths = []
    
    for genre in genres:
        path = os.path.join(source, genre)
        files = glob.glob(os.path.join(path, "*.png"))
        selected_files = files[-image_count:]
        image_paths.extend(selected_files)
    
    def process_image(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

source = "../data/images_original/"
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

prediction_dataset = create_prediction_dataset(source, genres, image_count=20, batch_size=32)

# RF
model_rf = RandomForestClassifier() # **rf_best_params
model_rf.fit(X_train, y_train_encoded)
y_probas_rf = model_rf.predict_proba(X_test)

# KNN
model_knn = KNeighborsClassifier() # **rf_best_params
model_knn.fit(X_train, y_train_encoded)
y_probas_knn = model_knn.predict_proba(X_test)

# SVM
model_svm = SVC(probability=True) # **rf_best_params
model_svm.fit(X_train, y_train_encoded)
y_probas_svm = model_svm.predict_proba(X_test)

# XGBoost
model_xgb = XGBClassifier() # **rf_best_params
model_xgb.fit(X_train, y_train_encoded)
y_probas_xgb = model_xgb.predict_proba(X_test)

# CNN
def load_model(weights_path, input_shape, num_classes):
    cnn = Sequential([
        BatchNormalization(input_shape=input_shape),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        Conv2D(256, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        Conv2D(512, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Flatten(),

        Dense(1024, activation="relu"),
        Dropout(0.5),
        Dense(512, activation="relu"),
        Dropout(0.5),
        BatchNormalization(),
        Dense(num_classes, activation="softmax")
    ])

    cnn.compile(optimizer=SGD(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cnn.load_weights(weights_path)
    return cnn

weights_path = ""
model = load_model(weights_path, (288, 432, 3), 10)
y_probas_cnn = model.predict(prediction_dataset)
predicted_class_indices = np.argmax(y_probas_cnn, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

# Combining Predictions
df_rf_probs = pd.DataFrame(y_probas_rf, columns=label_encoder.classes_)
df_knn_probs = pd.DataFrame(y_probas_knn, columns=label_encoder.classes_)
df_svm_probs = pd.DataFrame(y_probas_svm, columns=label_encoder.classes_)
df_xgb_probs = pd.DataFrame(y_probas_xgb, columns=label_encoder.classes_)
df_cnn_probs = pd.DataFrame(y_probas_cnn, columns=label_encoder.classes_)

# Ensemble the predictions by taking the max average probability
df_ensemble_probs = (df_rf_probs + 
                     df_knn_probs + 
                     df_svm_probs + 
                     df_xgb_probs + 
                     df_cnn_probs) / 5

predicted_classes_indices = df_ensemble_probs.idxmax(axis=1)

# Accuracy
print(accuracy_score(predicted_classes_indices, y_test))
