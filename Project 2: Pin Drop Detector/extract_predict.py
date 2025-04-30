import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import joblib
import os
import glob
import re
from material_model import extract_material_features
from height_model import extract_height_features    
from distance_model import extract_distance_features

# Configuration (assuming this exists or define it)
CONFIG = {'sample_rate': 10000}  # Adjust according to your setup


def load_model(model_dir, purpose):
    """
    Load a trained model and its feature columns for a specific purpose
    """
    model_path = os.path.join(model_dir, f"{purpose}_classifier_model.pkl")
    feature_path = os.path.join(model_dir, f"{purpose}_feature_columns.pkl")
    
    clf = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)
    print(f"Loaded {purpose} model from {model_path}")
    return clf, feature_cols

def predict(signal_data, clf, feature_cols, feature_extractor):
    """
    Generic prediction function using specified feature extractor
    """
    features = feature_extractor(signal_data)
    features_df = pd.DataFrame([features])
    X = features_df[feature_cols]
    return clf.predict(X)[0]

def runner(model_dir, signal_data):
    material_clf, material_feats = load_model(model_dir, "material")
    height_clf, height_feats = load_model(model_dir, "height")
    distance_clf, distance_feats = load_model(model_dir, "distance")

    # Get predictions from each model
    material_pred = predict(signal_data, material_clf, material_feats, extract_material_features)
    height_pred = predict(signal_data, height_clf, height_feats, extract_height_features)
    distance_pred = predict(signal_data, distance_clf, distance_feats, extract_distance_features)

    return f"Material: {material_pred}, Height: {height_pred}, Distance: {distance_pred}"
    # print(f"Material: {material_pred}, Height: {height_pred}, Distance: {distance_pred}")


if __name__ == "__main__":
    # Load all three models
    model_dir = "model_direct"  # Directory containing all models
    material_clf, material_feats = load_model(model_dir, "material")
    height_clf, height_feats = load_model(model_dir, "height")
    distance_clf, distance_feats = load_model(model_dir, "distance")

    # Load new signal data
    signal_file = 'datasets/data25_4_final_ver1/eraser_30h10d_11.csv'
    signal_data = pd.read_csv(signal_file, header=0).squeeze().astype(float).values

    # Get predictions from each model
    material_pred = predict(signal_data, material_clf, material_feats, extract_material_features)
    height_pred = predict(signal_data, height_clf, height_feats, extract_height_features)
    distance_pred = predict(signal_data, distance_clf, distance_feats, extract_distance_features)

    print(f"Material: {material_pred}, Height: {height_pred}, Distance: {distance_pred}")