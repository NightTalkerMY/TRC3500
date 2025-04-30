import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib
import os
import glob
import re
from scipy import signal, stats
# import pywt  # Requires pywt package installation

# Configuration constants
CONFIG = {
    'sample_rate': 10000,
    'feature_cols': [
        # Existing features
        'peak_amplitude',
        'energy',
        'bounce_duration',
        
        # New material features
        'rise_time',
        'peak_interval',
        'impact_aftermath_ratio',
        'decay_time_constant',
        'derivative_max',
    ],
    'model_path': 'model_direct/height_classifier_model.pkl',
    'feature_path': 'model_direct/height_feature_columns.pkl',
    'file_pattern': r'(?P<obj_type>.+?)_(?P<height>\d+)h(?P<distance>\d+)d_(?P<trial>\d+)\.csv'
}


def extract_height_features(signal_data):
    """Features for height classification (10cm vs 30cm)"""
    abs_signal = np.abs(signal_data)
    peaks, _ = signal.find_peaks(abs_signal, height=0.1*abs_signal.max(),
                               distance=int(0.05*CONFIG['sample_rate']))
    
    # Timing features
    rise_time = (np.argmax(abs_signal) / CONFIG['sample_rate']) if len(abs_signal) > 0 else 0
    peak_interval = (peaks[1] - peaks[0])/CONFIG['sample_rate'] if len(peaks)>1 else 0
        
    return {
        # Existing features
        'peak_amplitude': abs_signal.max(),
        'energy': np.sum(abs_signal**2),
        'bounce_duration': (peaks[-1] - peaks[0])/CONFIG['sample_rate'] if len(peaks)>1 else 0,
        
        # New height features
        'rise_time': rise_time,
        'peak_interval': peak_interval,
        'impact_aftermath_ratio': abs_signal.max() / (np.mean(abs_signal[np.argmax(abs_signal):]) + 1e-6),
        'decay_time_constant': (np.argmax(abs_signal <= 0.1*abs_signal.max()) - np.argmax(abs_signal)) 
                             / CONFIG['sample_rate'],
        'derivative_max': np.max(np.abs(np.diff(signal_data)))
    }

def parse_filename(filename):
    """Parse filename using regex with validation."""
    match = re.match(CONFIG['file_pattern'], filename)
    if not match:
        return None
    groups = match.groupdict()
    return {
        **groups,
        'height': int(groups['height']),
        'distance': int(groups['distance']),
        'trial': int(groups['trial']),
        'scenario': f"{groups['height']}" # labelling
    }

def load_process_data(directory):
    """Load and process all data files with integrated validation."""
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise ValueError("No CSV files found with expected pattern")
    
    data = []
    for f in files:
        try:
            meta = parse_filename(os.path.basename(f))
            if not meta:
                continue
                
            df = pd.read_csv(f, usecols=lambda c: any(k in c.lower() for k in ['adc', 'value']))
            signal_data = df.iloc[:, 0].values
            features = extract_height_features(signal_data)
            
            data.append(pd.Series({
                **features,
                **meta,
                'object_type': 1 if '50cents' in meta['obj_type'] else 2
            }))
        except Exception as e:
            print(f"Skipped {os.path.basename(f)}: {str(e)}")
    
    return pd.DataFrame(data)

def train_evaluate_model(df, retrain=True):
    if retrain:
        X = df[CONFIG['feature_cols']]
        y = df.scenario

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Train Random Forest on training data
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate on test data
        y_pred = clf.predict(X_test)
        print("\n#################################### Height classification ####################################")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        # Add precision, recall, and F1-score
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model and feature columns
        joblib.dump(clf, CONFIG['model_path'])
        joblib.dump(CONFIG['feature_cols'], CONFIG['feature_path'])
    else:
        clf = joblib.load(CONFIG['model_path'])

    return clf


def visualize_results(clf, df):
    """Generate required visualizations."""
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=CONFIG['feature_cols'], 
             class_names=clf.classes_, rounded=True)
    plt.savefig("decision_tree.png", bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    for label in df.scenario.unique():
        subset = df[df.scenario == label]
        plt.scatter(subset.settling_time, subset.peak_amplitude, label=label)
    plt.xlabel('Settling Time'), plt.ylabel('Peak Amplitude')
    plt.legend(), plt.tight_layout()
    plt.savefig("feature_visualization.png")

def classify_height_drops(directory, retrain=True):
    """Main classification pipeline."""
    df = load_process_data(directory)
    clf = train_evaluate_model(df, retrain)
    # visualize_results(clf, df)
    
    print("\nFeature Importance:")
    print(pd.DataFrame({
        'feature': CONFIG['feature_cols'],
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False))
    print("Classification height complete!")

    return clf, df

if __name__ == "__main__":
    classifier, data = classify_height_drops("Training Model/28_4_3", retrain=True)
    print("Classification complete!")