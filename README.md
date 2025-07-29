# Tree Plantation Health Detection Model

## Code Documentation

### 1. Introduction

This document provides detailed documentation for the implementation of a Tree Plantation Health Detection Model intended for deployment on Jetson Nano-based drones and UAVs.

The purpose of this project is to detect and analyze the health condition of trees using a YOLO-based object detection model, annotated images, and custom logical attributes. This guide breaks down the code into sections and provides usage instructions, explanations, and prerequisites required to run the implementation.

The detection model has been optimized for Jetson Nano, which is commonly used in drones with limited RAM and memory. A highly efficient YOLOv8 (nano) model and a quantized version of TinyLlama have been used to ensure smooth inference without heavy computational load.

---

### 2. Prerequisites

Before running the code, ensure you have the following:

- Python 3.8 or higher  
- Jupyter Notebook  
- Required Python libraries (install using `pip`):
  - `ultralytics`
  - `opencv-python`
  - `matplotlib`
  - `os`
  - `json`
- A trained YOLOv8 model (e.g., `best.pt`) placed in the correct directory  
- Input image directory with test images  
- Output directory for saving annotated images and JSON files
- Quantized TinyLlama Model GGUF (You may download from the following link: [Click Here](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF))

---

### 3. Implementation Overview

This implementation offers a practical solution for automated tree health monitoring in afforestation zones using YOLO-based object detection. It's designed for deployment on drones to enable large-scale, real-time environmental analysis.

<details>
<summary>Key Steps</summary>

1. **Import Required Libraries**  
   Load necessary Python libraries including Ultralytics YOLO, OpenCV, JSON, and Matplotlib for visualization.

2. **Load YOLO Model**  
   Use a pre-trained YOLOv8 model (`best.pt`) to detect objects in plantation images.

3. **Read Test Images**  
   Load all test images from a specified directory into memory for processing.

4. **Run Detection**  
   Pass each image through the YOLO model to obtain bounding boxes and class predictions.

5. **Extract and Visualize Results**  
   Use OpenCV to draw bounding boxes and extract metadata from predictions.

6. **Save Metadata in JSON Format**  
   Store model predictions, including standard COCO-style data and custom attributes (e.g., `HealthStatus`, `HasLeaves`, `TreeHeightCategory`, etc.), in structured JSON files.

7. **Save Annotated Images**  
   Save output images with drawn bounding boxes for verification and record-keeping.

</details>

---

### 4. Use Case and Benefits

This model provides a scalable framework for integrating computer vision into smart forestry and environmental monitoring. When mounted on drones, it enables:

- Tracking the progress of afforestation efforts  
- Identifying unhealthy or isolated trees  
- Planning data-driven forestry interventions  
- Reducing manual fieldwork  
- Accelerating decision-making  
- Ensuring transparent and accountable forest governance

---
