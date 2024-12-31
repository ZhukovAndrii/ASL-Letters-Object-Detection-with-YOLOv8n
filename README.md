# ASL-Letters-Object-Detection-with-YOLOv8n

## Video Demo:
https://youtu.be/n5A5rx2VAz4

## Description:
This project demonstrates a real-time object detection system for recognizing and classifying **American Sign Language (ASL) letters** using a fine-tuned **YOLOv8n (Nano)** model. The system runs on Mac, using the camera for live detections.

The project includes all stages of development, from fine-tuning the YOLOv8 model on custom data using Google Colab to deploying it for real-time inference on macOS. This README details the methodology, challenges faced, and how to set up and run the project.

---

## Features:
- Fine-tuned YOLOv8n model for ASL letter detection (A-Z).
- Real-time detection using OpenCV for live video feeds.
- Lightweight design suitable for real-time performance on not very powerful devices.
- Configurable confidence thresholds and resolutions for optimal performance.

---

## Files:
### `best_model.py`
- Script used to fine-tune the YOLOv8n model in Google Colab.
- Adjusts training parameters such as `epochs`, `batch size`, and `image size`.
- Uses the ASL dataset for training and validation.

### `best2.pt`
- The fine-tuned YOLOv8n model weights generated after training.
- Used for inference in the real-time detection script.

### `realtime_detection.py`
- Main Python script for real-time detection:
  - Captures frames from the Mac's camera.
  - Runs the YOLOv8 model for inference.
  - Annotates frames with bounding boxes and class labels for detected ASL letters.

### `requirements.txt`
- Specifies the dependencies required for the project:
  - `ultralytics` for YOLOv8.
  - `opencv-python` for video stream handling.

### `README.md`
- Documentation file describing the project, its functionality, challenges, and future perspectives.

---

## How It Works:
### **Model Fine-Tuning (Google Colab)**
The model fine-tuning process is implemented in Google Colab. Below is a simplified explanation of how the training script (`train.py`) works:
1. **Import YOLOv8 and Dependencies**:
   - The `ultralytics` library is used to load the pre-trained YOLOv8n model.
   - The model is set up for training with the custom dataset.

2. **Dataset Loading**:
   - The ASL Letters dataset is downloaded from Roboflow.
   - The dataset's structure is configured in a YAML file.

3. **Hyperparameter Configuration**:
   - Training parameters such as `epochs`, `batch size`, and `confidence threshold` are set.
   - Augmentation techniques like rotation and scaling are applied to improve generalization.

4. **Model Training**:

#### Data Preparation:
  - Images and corresponding labels are loaded, resized to the specified `imgsz` and augmented.
  - They are grouped into batches.

#### Model Initialization
- The YOLOv8 model is initialized using the pre-trained weights (`yolov8n.pt`).
- Layers of the model can be fine-tuned or frozen.

#### Forward Pass
- Each batch of images is passed through the model outputting:
  - Bounding box predictions (x, y, width, height).
  - Class probabilities for each bounding box.
  - Confidence scores.

#### Loss Computation
- The loss function combines multiple components:
  - **Box Loss**: how close predicted bounding boxes are to ground-truth boxes.
  - **Class Loss**: how close predicted class is to ground-truth class.
  - **Objectness Loss**: how well regions with objects are identified.

#### Backpropagation
- Gradients are calculated based on the loss function.
- These gradients are used to adjust the model’s weights using the Adam optimizer.

#### Validation
- After each epoch, the model checks how well it generalize on a validation set:
  - Predictions are compared with ground-truth labels.
  - Metrics such as **Precision**, **Recall**, and **mAP (Mean Average Precision)** are calculated.

5. **Save Trained Weights**:
   - After training, the best fine-tuned model (`best2.pt`) is saved.

### **Real-Time Detection (Mac)**
1. The fine-tuned model (`best2.pt`) is loaded into `realtime_detection.py`.
2. OpenCV captures frames from the Mac's camera.
3. YOLOv8 model processes every frame.
4. Detected objects are annotated with bounding boxes and class labels.
5. The annotated video stream is displayed in a live window.

---

## Installation and Usage:
### Prerequisites:
- Python 3.7 or higher installed on macOS.
- A camera connected to your Mac.

### Installation:
1. Clone the repository:

   -  git clone https://github.com/your-username/your-repo.git
   -  cd your-repo
   -  pip install -r requirements.txt

   **Ensure best.pt is in the same directory as realtime_detection.py**

   -  python3 realtime_detection.py

   **Press 'q' to exit**

## Challenges faced:

### Misclassification:
 - The model often misclassified human faces as the letter J.
 - Filtering low-confidence detections might help.
### Dataset Imbalance:
 - Some ASL letters had fewer training samples, leading to imbalanced performance.
 - Data augmentation was used to solve this problem.
### Real-Time Performance:
 - Balancing accuracy and speed on a live camera feed was challenging.
 - The YOLOv8n model was chosen for its lightweight architecture.
 - Creating ncnn model with onnx is another option for not nano models.

## Limitations:

### Lighting Conditions:
The system requires well-lit environments for better performance.
### Model Scope:
The model is trained only for ASL letters detection and cannot generalize beyond this scope.
### Hardware Dependency:
Performance is dependent on the Mac's camera and computational power. For example on Raspberry Pi it might struggle.

## Future Perspectives:

### Dataset Expansion:
Incorporate more diverse and extensive ASL datasets for improved accuracy.
### Face Detection Preprocessing:
Add a face-detection step to eliminate false positives from human faces.
### Web-Based Deployment:
Build a web app for easier accessibility and use across devices.
### Integration with Accessibility Tools:
Extend functionality to translate ASL letters into words or sentences.

## Summary:

This project integrates computer vision and deep learning to create accessibility tool. I fine-tuned YOLOv8 model on ASL Letters dataset for object detection in real-time. In future this project may be extended to be more developed accessibility solution.
