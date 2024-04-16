# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

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
# rf_param_dist = {
#     'n_estimators': randint(50, 1000),
#     'max_depth': [None] + list(range(1, 21)),
#     'min_samples_split': randint(2, 11),
#     'min_samples_leaf': randint(1, 5)
# }
# rf_random_search = RandomizedSearchCV(RandomForestClassifier(), rf_param_dist, n_iter=20, cv=4)
# rf_random_search.fit(X_train, y_train)
# print("fit_rf")
# rf_best_params = rf_random_search.best_params_
# with open('best_params_rf.txt', 'w') as file:
#     for key, value in rf_best_params.items():
#         file.write(f"{key}: {value}\n")
rf_best_params = {'max_depth': 11, 
                'min_samples_leaf': 2, 
                'min_samples_split': 2, 
                'n_estimators': 720, 
                }
model_rf = RandomForestClassifier(**rf_best_params)
model_rf.fit(X_train, y_train)
y_probas_rf = model_rf.predict_proba(X_test)

# KNN
# knn_param_dist = {
#     'n_neighbors': randint(1, 15)
# }
# knn_random_search = RandomizedSearchCV(KNeighborsClassifier(), knn_param_dist, n_iter=20, cv=4)
# knn_random_search.fit(X_train_scaled, y_train)
# print("fit_knn")
# knn_best_params = knn_random_search.best_params_
# with open('best_params_knn.txt', 'w') as file:
#     for key, value in knn_best_params.items():
#         file.write(f"{key}: {value}\n")
knn_best_params = {'n_neighbors': 1}
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model_knn = KNeighborsClassifier(**knn_best_params)
model_knn.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
y_probas_knn = model_knn.predict_proba(X_test_scaled)

# SVM
# svm_param_dist = {
#     'C': uniform(0.1, 10),
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': ['scale', 'auto']
# }
# svm_random_search = RandomizedSearchCV(SVC(probability=True), svm_param_dist, n_iter=20, cv=4)
# svm_random_search.fit(X_train, y_train)
# print("fit_svm")
# svm_best_params = svm_random_search.best_params_
# with open('best_params_svm.txt', 'w') as file:
#     for key, value in svm_best_params.items():
#         file.write(f"{key}: {value}\n")
svm_best_params = {'C': 1,
                'kernel': 'rbf',
                'gamma': 'auto'}
model_svm = SVC(**svm_best_params, probability=True)
model_svm.fit(X_train_scaled, y_train)
y_probas_svm = model_svm.predict_proba(X_test_scaled)

# XGBoost
# xgb_param_dist = {
#     'n_estimators': randint(50, 200), 
#     'learning_rate': uniform(0.05, 0.15), 
#     'max_depth': randint(5, 21) 
# }
# xgb_random_search = RandomizedSearchCV(XGBClassifier(), xgb_param_dist, n_iter=20, cv=4)
# xgb_random_search.fit(X_train, y_train_encoded)
# print("fit_xgb")
# xgb_best_params = xgb_random_search.best_params_
# with open('best_params_xgb.txt', 'w') as file:
#     for key, value in xgb_best_params.items():
#         file.write(f"{key}: {value}\n")
xgb_best_params = {
    'n_estimators': 179, 
    'learning_rate': .1, 
    'max_depth': 7 
}
model_xgb = XGBClassifier(**xgb_best_params)
model_xgb.fit(X_train, y_train_encoded)
y_probas_xgb = model_xgb.predict_proba(X_test)

# Combining Predictions
df_rf_probs = pd.DataFrame(y_probas_rf, columns=label_encoder.classes_)
df_knn_probs = pd.DataFrame(y_probas_knn, columns=label_encoder.classes_)
df_svm_probs = pd.DataFrame(y_probas_svm, columns=label_encoder.classes_)
df_xgb_probs = pd.DataFrame(y_probas_xgb, columns=label_encoder.classes_)

# Ensemble the predictions by taking the max average probability
df_ensemble_probs = (df_rf_probs + 
                     df_knn_probs + 
                     df_svm_probs + 
                     df_xgb_probs
                     ) / 4

# Random Forest Accuracy
predicted_classes_indices = df_rf_probs.values.argmax(axis=1)
predicted_class_labels = label_encoder.inverse_transform(predicted_classes_indices)
accuracy = accuracy_score(predicted_class_labels, y_test)
print("Random Forest Accuracy:", accuracy)

# KNN Accuracy
predicted_classes_indices = df_knn_probs.values.argmax(axis=1)
predicted_class_labels = label_encoder.inverse_transform(predicted_classes_indices)
accuracy = accuracy_score(predicted_class_labels, y_test)
print("KNN Accuracy:", accuracy)

# SVM Accuracy
predicted_classes_indices = df_svm_probs.values.argmax(axis=1)
predicted_class_labels = label_encoder.inverse_transform(predicted_classes_indices)
accuracy = accuracy_score(predicted_class_labels, y_test)
print("SVM Accuracy:", accuracy)

# XGB Accuracy
predicted_classes_indices = df_xgb_probs.values.argmax(axis=1)
predicted_class_labels = label_encoder.inverse_transform(predicted_classes_indices)
accuracy = accuracy_score(predicted_class_labels, y_test)
print("XGBoost Accuracy:", accuracy)

# Ensemble Accuracy
predicted_classes_indices = df_ensemble_probs.values.argmax(axis=1)
predicted_class_labels = label_encoder.inverse_transform(predicted_classes_indices)
accuracy = accuracy_score(predicted_class_labels, y_test)
print("Ensemble Accuracy:", accuracy)
