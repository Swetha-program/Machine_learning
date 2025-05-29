import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from xgboost import XGBClassifier
from sklearn.utils import resample
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("data/health_data.csv")
df.drop(columns=['id'], inplace=True)

# 2. Encode categorical features
label_encoders = {}
for col in df.select_dtypes('object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Split early to avoid leakage
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the dataset into train and test before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Bmi
# For training data
X_train = X_train.assign(bmi=X_train['bmi'].fillna(X_train['bmi'].mean()))
# For test data (using training mean to avoid data leakage)
X_test = X_test.assign(bmi=X_test['bmi'].fillna(X_train['bmi'].mean()))


# 5. Feature selection on training set only
mis_scores = mutual_info_classif(X_train, y_train)
chi_scores, _ = chi2(X_train, y_train)
anova_scores = f_classif(X_train, y_train)[0]

# Combine into DataFrame
scores_df = pd.DataFrame({
    'Feature': X_train.columns,
    'MIS': mis_scores,
    'Chi2': chi_scores,
    'ANOVA': anova_scores
}).set_index('Feature')
# Drop weak or unwanted features
drop_features = ['bmi', 'smoking_status', 'heart_disease', 'hypertension']
X_train = X_train.drop(columns=drop_features)
X_test = X_test.drop(columns=drop_features)  
# After feature selection, show the remaining selected features
selected_features = X_train.columns.tolist()


# 6. Scale using training data only
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Convert back to DataFrame for visualization
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# SMOTE
# Step 1: Undersample majority class (ratio 1:10)
undersample = RandomUnderSampler(sampling_strategy=0.1, random_state=42)

# Step 2: Apply SMOTE to oversample minority class to balance it
smote = SMOTE(sampling_strategy=1, random_state=42)

# Create pipeline for undersampling then oversampling
balance_pipeline = Pipeline([
    ('under', undersample),
    ('smote', smote)
])

# Apply balancing to training data only
X_train_bal, y_train_bal = balance_pipeline.fit_resample(X_train_scaled, y_train)


# Train model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,         
    max_depth=4,              
    learning_rate=0.1,        
    subsample=0.8,            
    colsample_bytree=0.8,     
    reg_alpha=0.3,            
    reg_lambda=1,             
    min_child_weight=2,
    gamma=0.05                
)

xgb_model.fit(X_train_bal, y_train_bal)

# Save model, scaler, and encoders
os.makedirs("model", exist_ok=True)

# Save the model using XGBoost's native method
xgb_model.save_model("model/stroke_model.json")  # or .bin if you prefer

# You can still use pickle for these:
with open("model/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("model/label_encoders.pkl", "wb") as encoder_file:
    pickle.dump(label_encoders, encoder_file)


print(" Training complete! Model, scaler, and encoders saved.")


