# ==============================================================================
# CODEALPHA DATA SCIENCE INTERNSHIP
# Repository Name: CodeAlpha_Iris_Classification
# Task Completed: TASK 1 - Iris Flower Classification
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Global configuration for professional layouts
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_preprocess_pipeline():
    print("--- Step 1: Loading Iris Dataset from Source ---")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    # Reading raw CSV dataset
    df = pd.read_csv(url, names=columns)
    
    print("\n[DATASET PREVIEW]")
    print(df.head())
    
    # Dynamic Exploratory Inspection
    print("\n[MISSING ENTRIES AUDIT]")
    print(df.isnull().sum())
    
    # Feature & Target Vector Formatting
    X = df.drop(columns=['species'])
    y = df['species']
    
    # Categorical Label Encoding
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    return X, y_encoded, encoder, df

def execute_model_training(X, y):
    print("\n--- Step 2: Training Predictive Machine Learning Engine ---")
    
    # Stratified Train-Test Segmentation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_split=0.2, random_state=42, stratify=y
    )
    
    # Ensemble Classifier Implementation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictive Testing Performance
    predictions = model.predict(X_test)
    
    return model, X_test, y_test, predictions

def render_visualization_analytics(df, y_test, predictions, encoder):
    print("\n--- Step 3: Generating Insight Plots & Evaluation Graphics ---")
    
    # Visualization 1: Feature Pairplot Distribution Analysis
    pair_plot = sns.pairplot(df, hue='species', diag_kind='kde', palette='husl')
    pair_plot.fig.suptitle('Iris Feature Distribution Matrix & Class Separation', y=1.02, fontsize=14)
    plt.savefig('iris_feature_matrix.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Visualization 2: Model Performance Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Classifier Predictive Performance Confusion Matrix', pad=15, fontsize=13)
    plt.xlabel('Predicted Species Labels')
    plt.ylabel('Actual Verified Labels')
    plt.tight_layout()
    plt.savefig('model_confusion_matrix.png', dpi=150)
    plt.close()
    
    print(">> Validation plots successfully saved to current workspace!")

def generate_performance_logs(y_test, predictions, encoder):
    print("\n--- Step 4: Final Evaluation Performance Summary ---")
    acc = accuracy_score(y_test, predictions)
    print(f"Overall Classification Accuracy Metric: {acc * 100:.2f}%\n")
    print("[DETAILED CLASSIFICATION ANALYSIS REPORT]")
    print(classification_report(y_test, predictions, target_names=encoder.classes_))

if __name__ == "__main__":
    X, y, encoder, original_df = load_and_preprocess_pipeline()
    model, X_test, y_test, preds = execute_model_training(X, y)
    render_visualization_analytics(original_df, y_test, preds, encoder)
    generate_performance_logs(y_test, preds, encoder)
