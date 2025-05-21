import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import joblib
import matplotlib.pyplot as plt

# Loading and preprocessing data
def load_and_preprocess_data(file_path='rrrr.csv'):
    df = pd.read_csv(file_path)
    
    # Dropping irrelevant or problematic columns
    columns_to_drop = ['id_student', 'prediction', 'code_module', 'code_presentation']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Creating binary target: 1 for Withdrawn, 0 for others
    df['dropout'] = (df['final_result'] == 'Withdrawn').astype(int)
    df = df.drop(columns=['final_result'])
    
    # Handling missing values
    numerical_cols = ['studied_credits', 'sum_click', 'count_click', 'date_registration', 
                      'date_unregistration', 'final_score', 'module_presentation_length']
    categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    
    # Filling missing numerical values with median
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].replace('', np.nan)
            df[col] = df[col].fillna(df[col].median())
    
    # Filling missing categorical values with mode
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encoding categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Converting all columns to numeric, replacing non-numeric with NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filling any remaining NaN with column median
    df = df.fillna(df.median())
    
    return df, label_encoders

# Training the model and plotting ROC curve
def train_model(df):
    # Defining features and target
    X = df.drop(columns=['dropout'])
    y = df['dropout']
    
    # Scaling numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initializing Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Defining hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Performing GridSearchCV
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Getting best model
    best_model = grid_search.best_estimator_
    
    # Predicting on test set
    y_pred = best_model.predict(X_test)
    
    # Calculating metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Calculate ROC curve and AUC
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probability for the positive class (1: Withdrawn)
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Dropout Prediction')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    return best_model, scaler, metrics, grid_search.best_params_, auc

# Main execution
if __name__ == "__main__":
    # Loading and preprocessing data
    df, label_encoders = load_and_preprocess_data()
    
    # Training model and plotting ROC
    model, scaler, metrics, best_params, auc = train_model(df)
    
    # Saving model and scaler
    joblib.dump(model, 'dropout_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    # Printing results
    print("Best Parameters:", best_params)
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print(f"AUC: {auc:.4f}")