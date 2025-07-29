# Tree Plantation Health Detection Model

# Code Documentation

## 1. Introduction

#### This document provides detailed documentation for the implementation of aTree

#### Plantation Health Detection Modelfor deployment in Jetson Nano indrones and UAVs.

#### The purpose of this project is todetect and analyze the health condition of treesusing a

#### YOLO-based object detection model, custom attributes, and annotated images. This guide

#### breaks down the code into sections and provides explanations, usage guidance,and

#### prerequisites required to run the implementation.

#### Thedetection model has been created for Jetson Nano used in drones and UAVs which have

#### limited RAM and memory.Hence, highlyoptimized YOLOv8 (nano)model andquantized

#### version of TinyLlamahas been used for inference so that the model can run smoothly on

#### drones without any physical computation constraints.

## 2. Prerequisites

#### Before running the code, ensure you have the following installed:

#### · Python 3.8 or higher

#### · Jupyter Notebook

#### · Required Python libraries(install using pip):

#### ultralytics

#### opencv-python

#### matplotlib

#### os

#### json

#### · A trained YOLOv8 model (e.g., best.pt) placed in the appropriate path

#### · Input image directory with test images

#### · Output directory for saving bounding box images and JSONs

## 3. Implementation

#### This implementation presents a practical solution forautomated tree health monitoring

#### in afforestation zones usingYOLO-based object detection, ideally deployable ondrones

#### for large-scale environmental tracking. By combining deep learning with logical

#### annotation, this model allows for near real-time analysis of tree conditions, crucial for

#### government forestry departments,reforestation initiatives, andeco-restoration

#### projects.

#### The steps implemented in this project can be summarized as:

#### · Importing Required Libraries: Loaded all necessary Python libraries, including

#### Ultralytics' YOLO, OpenCV, JSON, and matplotlib for visualization.

#### · Loading theYOLO Model: A pre-trained YOLOv8 model (best.pt) was loaded to

#### perform detection on plantation images.

#### · Reading Test Images: Images from a specified directory were loaded into a list for

#### processing.

#### · Running Detection: Each test image was passed through the YOLO model to get

#### predictions including bounding boxes and class information.

#### · Extracting and Visualizing Results: Bounding boxes were drawn on the images

#### using OpenCV, and relevant prediction metadata was extracted.

#### · Storing Metadata in JSON: The model's predictions, including both standard COCO-

#### style and custom logical attributes (like HealthStatus, HasLeaves,

#### TreeHeightCategory, etc.), were saved into structured JSON files.

#### · Saving Annotated Images: The result images with bounding boxes were saved to

#### disk for visual inspection and record-keeping.

#### This model provides ascalable frameworkfor integratingcomputer vision into smart

#### environmental monitoring systems. When mounted on drones, it can assist government

#### bodies in:

#### · Tracking afforestation progress,

#### · Detecting unhealthy or isolated trees,

#### · Planning interventions with data-driven insights.

#### By automating observation and logging, this solution helpsreduce manual fieldwork,

#### improvedecision-making speed, and maintaintransparent, evidence-based forest

#### governance.
