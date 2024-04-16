# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

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

# Combining Predictions
df_rf_probs = pd.DataFrame(y_probas_rf, columns=label_encoder.classes_)
df_knn_probs = pd.DataFrame(y_probas_knn, columns=label_encoder.classes_)
df_svm_probs = pd.DataFrame(y_probas_svm, columns=label_encoder.classes_)
df_xgb_probs = pd.DataFrame(y_probas_xgb, columns=label_encoder.classes_)

print(df_rf_probs.columns)
print(df_knn_probs.columns)
print(df_svm_probs.columns)
print(df_xgb_probs.columns)

# Ensemble the predictions by taking the max average probability
df_ensemble_probs = (df_rf_probs + 
                     df_knn_probs + 
                     df_svm_probs + 
                     df_xgb_probs
                     ) / 4

# Get the index of the column with the maximum probability for each row
predicted_classes_indices = df_ensemble_probs.values.argmax(axis=1)

# Convert indices to class labels
predicted_class_labels = label_encoder.inverse_transform(predicted_classes_indices)

# Calculating and printing the accuracy
accuracy = accuracy_score(predicted_class_labels, y_test)
print("Accuracy:", accuracy)
