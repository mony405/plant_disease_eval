import os
import random
import json
import shutil

def create_local_split(base_dir="data/crops", output_dir="data/test_split", num_per_class=10, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    eval_data = []
    
    # 1. Get all main crop directories (Apple, Cherry, Corn, etc.)
    crop_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    print("Sampling images locally from your dataset directory...")
    for crop in crop_folders:
        crop_path = os.path.join(base_dir, crop)
        
        # 2. Get all sub-class folders inside this crop (e.g., Apple_Black_rot)
        class_folders = [f for f in os.listdir(crop_path) if os.path.isdir(os.path.join(crop_path, f))]
        
        for class_folder in class_folders:
            class_path = os.path.join(crop_path, class_folder)
            
            # 3. Find all valid image files
            valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
            images = [f for f in os.listdir(class_path) if f.endswith(valid_extensions)]
            
            if not images:
                continue
            
            # 4. Perform true random sampling for this specific class
            sampled_images = random.sample(images, min(len(images), num_per_class))
            
            for idx, img_name in enumerate(sampled_images):
                src_path = os.path.join(class_path, img_name)
                
                # Define flat folder destination name
                dest_filename = f"{class_folder}_{idx}.jpg"
                dest_path = os.path.join(output_dir, dest_filename)
                
                # Copy file locally
                shutil.copy2(src_path, dest_path)
                
                # 5. Append metadata matching your training format exactly
                eval_data.append({
                    "original_index": f"{class_folder}_{img_name}",
                    "label": class_folder,
                    "crop": crop,
                    "status": class_folder, # Matches your target status format (e.g. Strawberry_Leaf_scorch)
                    "local_path": dest_path
                })
                
    return eval_data

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    test_list = create_local_split()
    
    with open("data/test_split.json", "w") as f:
        json.dump(test_list, f, indent=4)
        
    print(f"--- Complete! Processed {len(test_list)} total images. ---")
    print("Files copied to 'data/test_split/' and indexed in 'data/test_split.json'.")