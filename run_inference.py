#====================================
#--- Importing Required Libraries ---
#====================================

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pprint import pprint
from ultralytics import YOLO              # to load YOLO model (YOLOv8n (nano) specific for deployment in Jetson Nano)
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from collections import Counter
from llama_cpp import Llama               # to load the TinyLlama Model (Used Quantized GGUF Model)
from paths import *                       # to load the paths for the test images
from config import (
    GREEN_LOWER, GREEN_UPPER,
    HEALTHY_GREEN_RATIO, LEAF_PIXEL_THRESHOLD,
    SMALL_TREE_RATIO, MEDIUM_TREE_RATIO,
    ISOLATION_DISTANCE, VISIBLE_CROWN_THRESHOLD,
    CANOPY_SPARSE, CANOPY_PARTIAL
)


#====================================
#---- Tree Attribute Definitions ----
#====================================

def estimate_health(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_ratio = cv2.countNonZero(green_mask) / (image.shape[0] * image.shape[1])
    return "Healthy" if green_ratio > HEALTHY_GREEN_RATIO else "Unhealthy"

def has_leaves(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_pixels = cv2.countNonZero(green_mask)
    return green_pixels > LEAF_PIXEL_THRESHOLD

def categorize_height(box_height, image_height):
    ratio = box_height / image_height
    if ratio < SMALL_TREE_RATIO:
        return "Small"
    elif ratio < MEDIUM_TREE_RATIO:
        return "Medium"
    else:
        return "Large"

def is_isolated(box, all_boxes, threshold=ISOLATION_DISTANCE):
    x1, y1, w1, h1 = box
    center_x1 = x1
    for other in all_boxes:
        if np.array_equal(box, other):
            continue
        x2, y2, w2, h2 = other
        center_x2 = x2
        if abs(center_x1 - center_x2) < threshold:
            return False
    return True

def visible_crown(image, box):
    x, y, w, h = [int(i) for i in box]
    crown_region = image[y:y+int(h*0.3), x:x+w]
    hsv = cv2.cvtColor(crown_region, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    return cv2.countNonZero(green_mask) > VISIBLE_CROWN_THRESHOLD

def canopy_coverage(image, box):
    x, y, w, h = [int(i) for i in box]
    region = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_ratio = cv2.countNonZero(green_mask) / (w * h)

    if green_ratio < CANOPY_SPARSE:
        return "Sparse"
    elif green_ratio < CANOPY_PARTIAL:
        return "Partial"
    else:
        return "Full"

def classify_attributes(image, box, all_boxes, image_height):
    return {
        "HealthStatus": estimate_health(image),
        "HasLeaves": has_leaves(image),
        "TreeHeightCategory": categorize_height(box[3], image_height),
        "IsIsolated": is_isolated(box, all_boxes),
        "VisibleCrown": visible_crown(image, box),
        "CanopyCoverage": canopy_coverage(image, box)
    }


#====================================
#- YOLO Results and JSON Attributes -
#====================================

# --- Paths ---
model = YOLO(YOLO_MODEL_PATH)
input_image_dir = INPUT_IMAGE_PATH
output_image_dir = OUTPUT_IMAGE_PATH
output_json_dir = OUTPUT_JSON_PATH

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

# --- Lists ---
image_YOLO_set = []
image_file_set = []
test_image_set = []
image_set = []

# --- Load and run YOLO on images ---
for image_file in os.listdir(input_image_dir):
    if image_file.lower().endswith((".jpg", ".png")):
        image_path = os.path.join(input_image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"File does not exist: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        results = model(image_path, conf=0.15)
        image_file_set.append(image_file)
        image_set.append(image)
        test_image_set.append(image)
        image_YOLO_set.append(results[0])

# --- Draw bounding boxes and SAVE images ---
for idx, output in enumerate(image_YOLO_set):
    print(f"Image: {image_file_set[idx]}")

    img = test_image_set[idx].copy()
    boxes = output.boxes.xyxy.cpu().numpy()
    box_count = len(boxes)
    print(f"Number of detections: {box_count}")

    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    # Save image to output directory
    save_path = os.path.join(output_image_dir, image_file_set[idx])
    cv2.imwrite(save_path, img)

    print(f"Saved annotated image: {save_path}")
    print("------------------------------------------------------------")

# --- Extract bounding boxes, class IDs, and confidence scores ---
all_image_attributes = []

for index in range(len(image_YOLO_set)):
    boxes = image_YOLO_set[index].boxes.xywh.cpu().numpy()
    classes = image_YOLO_set[index].boxes.cls.cpu().numpy()
    confs = image_YOLO_set[index].boxes.conf.cpu().numpy()
    image_height = test_image_set[index].shape[0]

    all_attributes = []

    for box, cls, conf in zip(boxes, classes, confs):
        attr = classify_attributes(test_image_set[index], box, boxes, image_height)
        attr["bbox"] = box.tolist()
        attr["class"] = "Clustered Trees" if int(cls) == 0 else "Single Tree"
        attr["confidence_score"] = float(conf)
        all_attributes.append(attr)

    all_image_attributes.append(all_attributes)

# --- Save JSON outputs ---
for index in range(len(image_file_set)):
    tree_data = {
        "image": image_file_set[index],
        "tree_attributes": all_image_attributes[index]
    }

    base_name = os.path.splitext(image_file_set[index])[0]
    json_filename = f"{base_name}.json"
    output_path = os.path.join(output_json_dir, json_filename)

    with open(output_path, "w") as f:
        json.dump(tree_data, f, indent=2)

    print(f"Saved JSON: {output_path}")


#====================================
#---------- Saving Prompts ----------
#====================================

json_dir = OUTPUT_JSON_PATH
inference_output_dir = os.path.join(json_dir, "inference_prompts")
os.makedirs(inference_output_dir, exist_ok=True)

# Ensure sorted order
json_file_set = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

prompt_set = []

for json_file in json_file_set:
    json_file_path = os.path.join(json_dir, json_file)
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract Data from JSON (And Aggregate for All Trees in Image)
    image_name = data["image"]
    trees = data["tree_attributes"]

    total_trees = len(trees)
    total_confidence = 0.0
    total_area = 0.0
    clustered_trees = 0
    single_trees = 0
    uncertain_trees = 0

    health_counter = Counter()
    has_leaves_count = 0
    isolated_count = 0
    visible_crown_count = 0
    canopy_coverage_counter = Counter()
    height_category_counter = Counter()

    for t in trees:
        confidence = t["confidence_score"]
        total_confidence += confidence
        w, h = t["bbox"][2], t["bbox"][3]
        total_area += w * h

        if confidence < 0.15:
            uncertain_trees += 1

        if t.get("class", "").lower() == "clustered tree":
            clustered_trees += 1
        else:
            single_trees += 1

        health_status = t.get("HealthStatus", "Unknown")
        health_counter[health_status] += 1

        if str(t.get("HasLeaves", "False")).lower() == "true":
            has_leaves_count += 1
        if str(t.get("IsIsolated", "False")).lower() == "true":
            isolated_count += 1
        if str(t.get("VisibleCrown", "False")).lower() == "true":
            visible_crown_count += 1

        canopy_coverage_string = t.get("CanopyCoverage", "Unknown").lower()
        canopy_coverage_counter[canopy_coverage_string] += 1

        height_category = t.get("TreeHeightCategory", "Unknown")
        height_category_counter[height_category] += 1

    avg_confidence = total_confidence / total_trees if total_trees else 0
    avg_bbox_area = total_area / total_trees if total_trees else 0

    prompt = f"""### Instruction:
You are a forestry assistant analyzing this drone image with object detection and semantic attributes.

### Input:
Image: {image_name}
Total Trees: {total_trees}
- Single: {single_trees}
- Clustered: {clustered_trees}
- Uncertain Detections (<0.5 confidence): {uncertain_trees}

Detection Stats:
- Avg Confidence: {avg_confidence:.2f}
- Avg BBox Area: {avg_bbox_area:.2f}

Tree Attributes Summary:
- Health Statuses: {dict(health_counter)}
- Trees with Leaves: {has_leaves_count}
- Isolated Trees: {isolated_count}
- Trees with Visible Crown: {visible_crown_count}
- Canopy Coverage Types: {dict(canopy_coverage_counter)}
- Tree Height Categories: {dict(height_category_counter)}

### Questions:
1. Assess plantation health and density. Are trees healthy, dense, and well-distributed?
2. What actions should be taken to improve plantation health?
3. Are there signs of overcrowding, poor spacing, or deforested gaps?
4. If there are any clustered trees detected, is their overall canopy coverage healthy?

### Response:
"""

    prompt_set.append(prompt)

    # Save prompt to .txt file
    txt_filename = os.path.splitext(json_file)[0] + "_inference.txt"
    txt_path = os.path.join(inference_output_dir, txt_filename)
    with open(txt_path, "w") as txt_file:
        txt_file.write(prompt)

    print(f"Saved inference summary: {txt_path}")

print("\nAll inference prompts saved in:", inference_output_dir)


#====================================
#------- Running LLM Inference ------
#====================================
#====================================
#------- Running LLM Inference ------
#====================================

inference_results_dir = OUTPUT_INFERENCE_RESULTS_PATH
os.makedirs(inference_results_dir, exist_ok=True)

llm = Llama(model_path=LLAMA_MODEL_PATH)

for index, json_file in enumerate(json_file_set):
    prompt = prompt_set[index]
    
    # Generate output from Llama
    output = llm(prompt, max_tokens=768, temperature=0.5, top_p=0.9, repeat_penalty=1.1)
    result_text = output['choices'][0]['text'].strip()

    # Display output
    print(f"Output for {json_file}:")
    print(result_text)
    print("\n" + "-"*110 + "\n")

    # Save output to file
    base_name = os.path.splitext(json_file)[0]
    result_file_path = os.path.join(inference_results_dir, f"{base_name}_llm_output.txt")
    with open(result_file_path, "w") as f:
        f.write(result_text)

print("\nAll LLM inference results saved in:", inference_results_dir)
