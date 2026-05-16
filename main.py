import os
import sys

# Ensure the src directory is in the python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.prepare_data import create_local_split
from src.inference import run_evaluation
from src.metrics import calculate_metrics

def main():
    print("="*40)
    print("STARTING PLANT DISEASE EVALUATION PIPELINE")
    print("="*40)

    # Step 1: Prepare the Dataset
    if not os.path.exists("data/test_split.json"):
        print("\n[STAGE 1] Preparing test split...")
        test_data = create_local_split()
        
        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)
        
        # FIX: Save using native json library instead of .to_json()
        import json
        with open("data/test_split.json", "w") as f:
            json.dump(test_data, f, indent=4)
        print("Done.")
    else:
        print("\n[STAGE 1] Existing test split found. Skipping preparation.")

    # Step 2: Run Inference
    # This loads the model in 4-bit and runs on your GPU
    print("\n[STAGE 2] Running model inference...")
    try:
        run_evaluation()
    except Exception as e:
        print(f"Inference failed with error: {e}")
        return

    # Step 3: Calculate Metrics
    # This parses the results and generates the confusion matrix
    print("\n[STAGE 3] Calculating metrics and generating plots...")
    calculate_metrics()

    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("Check 'outputs/plots/' for the confusion matrix.")
    print("="*40)

if __name__ == "__main__":
    main()