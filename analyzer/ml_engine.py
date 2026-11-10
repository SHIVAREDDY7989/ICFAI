"""
Machine Learning Engine for Breast Cancer Classification Benchmark.
Handles data validation, model training, metric evaluation,
and high-resolution (300 DPI) multi-format chart generation.
"""

import os
import io
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from django.conf import settings

# Exact 10 feature columns + 1 target
FEATURE_COLUMNS = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave_points_mean',
    'symmetry_mean',
    'fractal_dimension_mean',
]

TARGET_COLUMN = 'diagnosis'
ALL_REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def get_sample_csv() -> str:
    """
    Dynamically generates a valid sample CSV matching the exact schema
    using scikit-learn's breast cancer dataset.
    Maps:
      0 (malignant in sklearn) -> 'Malignant' (Target 1)
      1 (benign in sklearn)    -> 'Benign'    (Target 0)
    """
    raw_data = load_breast_cancer()
    # Scikit-learn feature names for first 10 columns:
    # 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    # 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
    features_10 = raw_data.data[:, :10]
    df = pd.DataFrame(features_10, columns=FEATURE_COLUMNS)

    # Scikit-learn target: 0 = malignant, 1 = benign
    # Map according to project specification: Benign -> 0, Malignant -> 1
    labels = ['Malignant' if t == 0 else 'Benign' for t in raw_data.target]
    df[TARGET_COLUMN] = labels

    return df.to_csv(index=False)


def validate_and_preprocess_csv(file_obj) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates uploaded CSV file:
    - Normalizes column names (lowercase, stripped whitespace, underscores)
    - Checks required columns
    - Drops rows with NaNs
    - Validates numeric feature data types
    - Validates diagnosis values ('Benign'/'Malignant', 'B'/'M', 0/1)
    - Ensures at least 20 rows and both classes present
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Unable to parse CSV file: {str(e)}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Normalize column names: strip whitespace, lowercase, replace spaces with underscores
    col_map = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=col_map)

    # Check for missing required columns
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {', '.join(missing_features)}. "
            f"Please ensure all 10 clinical features are included."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COLUMN}'. "
            f"Accepted values are 'Benign'/'Malignant', 'B'/'M', or 0/1."
        )

    # Filter to required columns
    df = df[ALL_REQUIRED_COLUMNS].copy()

    # Drop NaNs
    initial_count = len(df)
    df = df.dropna()
    dropped_nans = initial_count - len(df)

    # Coerce features to numeric
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where features couldn't be converted to numeric
    df = df.dropna(subset=FEATURE_COLUMNS)

    # Validate target column and map to 0 (Benign) and 1 (Malignant)
    def normalize_target(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s in ['0', '0.0', 'benign', 'b']:
            return 0
        elif s in ['1', '1.0', 'malignant', 'm']:
            return 1
        return None

    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(normalize_target)
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Minimum rows check
    if len(df) < 20:
        raise ValueError(
            f"Insufficient data: Dataset contains only {len(df)} valid rows after cleaning. "
            f"A minimum of 20 complete rows is required for 80/20 stratified benchmarking."
        )

    # Check both classes present
    unique_classes = df[TARGET_COLUMN].unique()
    if len(unique_classes) < 2:
        found_class = "Benign" if 0 in unique_classes else "Malignant"
        raise ValueError(
            f"Dataset contains only one class ({found_class}). "
            f"Both Benign and Malignant samples are required for classification benchmarking."
        )

    benign_count = int((df[TARGET_COLUMN] == 0).sum())
    malignant_count = int((df[TARGET_COLUMN] == 1).sum())

    if benign_count < 2 or malignant_count < 2:
        raise ValueError(
            f"Each class must have at least 2 samples for stratified split. "
            f"Found Benign: {benign_count}, Malignant: {malignant_count}."
        )

    stats = {
        'total_rows': int(len(df)),
        'benign_count': benign_count,
        'malignant_count': malignant_count,
        'dropped_rows': dropped_nans,
    }

    return df, stats


def train_and_benchmark_models(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Splits data 80/20 stratified, scales with StandardScaler,
    trains the 6 classification algorithms, and computes metrics.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Exact 6 classifiers
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('SVM', SVC(kernel='rbf', random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('Naive Bayes', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ]

    results = []

    for name, clf in classifiers:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        acc = round(float(accuracy_score(y_test, y_pred) * 100), 2)
        prec = round(float(precision_score(y_test, y_pred, zero_division=0) * 100), 2)
        rec = round(float(recall_score(y_test, y_pred, zero_division=0) * 100), 2)
        f1 = round(float(f1_score(y_test, y_pred, zero_division=0) * 100), 2)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])

        results.append({
            'name': name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'cm': [[tn, fp], [fn, tp]],
        })

    split_info = {
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'test_benign': int((y_test == 0).sum()),
        'test_malignant': int((y_test == 1).sum()),
    }

    return results, split_info


def generate_and_save_charts(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Renders Chart 1 (Confusion Matrices 2x3 grid) and Chart 2 (Accuracy Comparison bar chart).
    Saves figures at 300 DPI to media/exports/ in .png, .jpg, and .jpeg using Pillow.
    Returns Base64 encoded strings and file URLs.
    """
    export_dir = Path(settings.MEDIA_ROOT) / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Chart 1: Confusion Matrices 2x3 grid
    # -------------------------------------------------------------
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
    fig_cm.suptitle(
        'Confusion Matrices of Classification Algorithms',
        fontsize=18,
        fontweight='bold',
        y=0.98,
        color='#1e293b'
    )

    axis_x_labels = ['Predicted Benign', 'Predicted Malignant']
    axis_y_labels = ['Actual Benign', 'Actual Malignant']

    for idx, res in enumerate(results):
        ax = axes_cm[idx // 3, idx % 3]
        cm_matrix = np.array(res['cm'])

        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'},
            xticklabels=axis_x_labels,
            yticklabels=axis_y_labels
        )

        ax.set_title(
            f"{res['name']}\nAccuracy: {res['accuracy']:.2f}% | FN: {res['fn']}",
            fontsize=12,
            fontweight='bold',
            pad=10,
            color='#0f172a'
        )
        ax.set_xlabel('Predicted Label', fontsize=10, labelpad=8, fontweight='medium')
        ax.set_ylabel('Actual Label', fontsize=10, labelpad=8, fontweight='medium')
        ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    cm_png_path = export_dir / 'confusion_matrices.png'
    cm_jpg_path = export_dir / 'confusion_matrices.jpg'
    cm_jpeg_path = export_dir / 'confusion_matrices.jpeg'

    fig_cm.savefig(cm_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cm)

    # Convert PNG to JPG & JPEG using Pillow (RGB conversion)
    with Image.open(cm_png_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(cm_jpg_path, 'JPEG', quality=95, dpi=(300, 300))
        rgb_img.save(cm_jpeg_path, 'JPEG', quality=95, dpi=(300, 300))

    # Base64 for instant HTML rendering
    with open(cm_png_path, 'rb') as f:
        cm_b64 = base64.b64encode(f.read()).decode('utf-8')

    # -------------------------------------------------------------
    # Chart 2: Accuracy Comparison bar chart
    # -------------------------------------------------------------
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))

    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    max_acc = max(accuracies)
    min_acc = min(accuracies)

    # Color logic: highest green (#2ca02c), lowest red (#d62728), rest blue (#1f77b4)
    # If all equal, color blue
    bar_colors = []
    for acc in accuracies:
        if acc == max_acc and max_acc != min_acc:
            bar_colors.append('#2ca02c')  # Green
        elif acc == min_acc and max_acc != min_acc:
            bar_colors.append('#d62728')  # Red
        else:
            bar_colors.append('#1f77b4')  # Blue

    bars = ax_acc.bar(names, accuracies, color=bar_colors, width=0.55, edgecolor='#0f172a', linewidth=1)

    ax_acc.set_title('Algorithm Accuracy Comparison', fontsize=16, fontweight='bold', pad=15, color='#1e293b')
    ax_acc.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax_acc.set_ylim(0, 108)
    ax_acc.grid(axis='y', linestyle='--', alpha=0.4)
    ax_acc.set_axisbelow(True)
    ax_acc.tick_params(axis='x', rotation=15, labelsize=10)

    # Display accuracy percentages on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax_acc.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#0f172a'
        )

    plt.tight_layout()

    acc_png_path = export_dir / 'accuracy_comparison.png'
    acc_jpg_path = export_dir / 'accuracy_comparison.jpg'
    acc_jpeg_path = export_dir / 'accuracy_comparison.jpeg'

    fig_acc.savefig(acc_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig_acc)

    with Image.open(acc_png_path) as img:
        rgb_img_acc = img.convert('RGB')
        rgb_img_acc.save(acc_jpg_path, 'JPEG', quality=95, dpi=(300, 300))
        rgb_img_acc.save(acc_jpeg_path, 'JPEG', quality=95, dpi=(300, 300))

    with open(acc_png_path, 'rb') as f:
        acc_b64 = base64.b64encode(f.read()).decode('utf-8')

    return {
        'confusion_matrices_b64': cm_b64,
        'accuracy_comparison_b64': acc_b64,
        'confusion_matrices_paths': {
            'png': str(cm_png_path),
            'jpg': str(cm_jpg_path),
            'jpeg': str(cm_jpeg_path),
        },
        'accuracy_comparison_paths': {
            'png': str(acc_png_path),
            'jpg': str(acc_jpg_path),
            'jpeg': str(acc_jpeg_path),
        }
    }


def execute_full_benchmark(file_obj) -> Dict[str, Any]:
    """
    End-to-end pipeline:
    1. Validates and preprocesses CSV
    2. Trains and benchmarks 6 algorithms
    3. Generates 300 DPI multi-format figures & base64
    4. Compiles summary statistics and highlights
    """
    df, stats = validate_and_preprocess_csv(file_obj)
    results, split_info = train_and_benchmark_models(df)
    charts = generate_and_save_charts(results)

    # Determine best model (highest accuracy, tie-breaker fewest FN)
    sorted_by_acc = sorted(results, key=lambda r: (r['accuracy'], -r['fn']), reverse=True)
    best_model = sorted_by_acc[0]

    # Determine worst model (lowest accuracy)
    worst_model = sorted_by_acc[-1]

    # Determine safest clinical model (fewest False Negatives, tie-breaker highest recall & accuracy)
    sorted_by_safety = sorted(results, key=lambda r: (r['fn'], -r['recall'], -r['accuracy']))
    safest_model = sorted_by_safety[0]

    return {
        'results': results,
        'dataset_stats': {
            **stats,
            **split_info,
        },
        'best_model': best_model,
        'worst_model': worst_model,
        'safest_model': safest_model,
        'charts': charts,
    }
