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
        'spectral_centroid',
        'high_freq_energy',
        'mean_frequency',
        'decay_rate',
        
        # New material features
        'zero_crossing_rate',
        'skewness',
        'kurtosis',
        'dominant_frequency',
        'harmonic_ratio',
        'derivative_std'
    ],
    'model_path': 'model_direct/material_classifier_model.pkl',
    'feature_path': 'model_direct/material_feature_columns.pkl',
    'file_pattern': r'(?P<obj_type>.+?)_(?P<height>\d+)h(?P<distance>\d+)d_(?P<trial>\d+)\.csv'
}


def extract_material_features(signal_data):
    """Features for material classification (coin vs eraser)"""
    abs_signal = np.abs(signal_data)
    peaks, _ = signal.find_peaks(signal_data, height=0.1*abs_signal.max(), 
                               distance=int(0.05*CONFIG['sample_rate']))
    
    # Time-domain features
    zero_crossings = np.where(np.diff(np.sign(signal_data)))[0].size
    derivative = np.diff(signal_data)
    
    # Frequency-domain features
    freqs, psd = signal.welch(signal_data, CONFIG['sample_rate'], nperseg=min(256, len(signal_data)))
    total_energy = np.sum(psd)
    dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
    
    # Harmonic detection
    harmonic_ratio = 0
    if len(psd) > 1:
        fundamental_idx = np.argmax(psd)
        harmonic_idx = np.argmax(psd[:len(psd)//2])  # Look in lower half for harmonic
        harmonic_ratio = psd[harmonic_idx] / (psd[fundamental_idx] + 1e-6)

    return {
        # Existing features
        'spectral_centroid': np.sum(freqs * psd) / (total_energy + 1e-6),
        'high_freq_energy': psd[freqs > 200].sum(),
        'mean_frequency': np.average(freqs, weights=psd),
        'decay_rate': (np.log(abs_signal[peaks[-1]]/abs_signal[peaks[0]])/(peaks[-1]-peaks[0])) 
                      if len(peaks)>1 else 0,
        
        # New material features
        'zero_crossing_rate': zero_crossings / len(signal_data),
        'skewness': stats.skew(signal_data),
        'kurtosis': stats.kurtosis(signal_data),
        'dominant_frequency': dominant_freq,
        'harmonic_ratio': harmonic_ratio,
        'derivative_std': np.std(derivative)
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
        'scenario': f"{groups['obj_type']}" # labelling
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
            features = extract_material_features(signal_data)
            
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
        print("\n#################################### Material classification ####################################")
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

def classify_material_drops(directory, retrain=True):
    """Main classification pipeline."""
    df = load_process_data(directory)
    clf = train_evaluate_model(df, retrain)
    # visualize_results(clf, df)
    
    print("\nFeature Importance:")
    print(pd.DataFrame({
        'feature': CONFIG['feature_cols'],
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False))
    print("Classification material complete!")

    return clf, df

if __name__ == "__main__":
    classifier, data = classify_material_drops("Training Model/28_4_3", retrain=True)
    print("Classification complete!")