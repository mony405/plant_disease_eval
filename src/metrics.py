import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
INPUT_PATH = "outputs/predictions/results.json"
PLOT_DIR = "outputs/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

def extract_json_from_text(text):
    """
    Robustly extracts crop and status from model output.
    Handles raw JSON strings or JSON wrapped in markdown blocks.
    """
    try:
        # 1. Try direct loading
        data = json.loads(text)
        return data.get("crop"), data.get("status")
    except json.JSONDecodeError:
        # 2. Try regex to find content between curly braces if model added text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("crop"), data.get("status")
            except:
                pass
    return None, None

def calculate_metrics():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run inference.py first.")
        return

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    y_true_full = []
    y_pred_full = []
    
    parsing_errors = 0

    for entry in data:
        gt = entry["ground_truth"]
        raw_pred = entry["prediction_raw"]
        
        # Ground Truth label (e.g., "Strawberry_Leaf_scorch")
        # We assume the dataset label matches the 'crop_status' format
        true_label = gt["full_label"]
        
        pred_crop, pred_status = extract_json_from_text(raw_pred)
        
        if pred_crop and pred_status:
            # Reconstruct the combined label for the confusion matrix
            # Replacing spaces with underscores to match typical dataset formatting
            predicted_label = f"{pred_crop.replace(' ', '_')}_{pred_status.replace(' ', '_')}"
            
            y_true_full.append(true_label)
            y_pred_full.append(predicted_label)
        else:
            parsing_errors += 1

    print(f"--- Evaluation Summary ---")
    print(f"Total Samples: {len(data)}")
    print(f"Successful Parses: {len(y_pred_full)}")
    print(f"Parsing Errors: {parsing_errors}")
    print("-" * 30)

    if not y_pred_full:
        print("No valid predictions to evaluate.")
        return

    # 1. Print Text Report
    report = classification_report(y_true_full, y_pred_full)
    print("Classification Report:")
    print(report)

    # 2. Generate Confusion Matrix Plot
    labels = sorted(list(set(y_true_full) | set(y_pred_full)))
    cm = confusion_matrix(y_true_full, y_pred_full, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Plant Disease Recognition: Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")

if __name__ == "__main__":
    calculate_metrics()