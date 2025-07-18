# 🐠 Active Vision-Based Real-Time Aquaculture Net Pen Inspection Using ROV

This repository contains the source code, models, and scripts used in the paper:

> **Waseem Akram et al.**, *"Active vision-based real-time aquaculture net pens inspection using ROV"*, 2025.

---

##  Overview
![Project Overview](media/project.png)

This project presents a novel **gradient-aware active vision system** for aquaculture net inspection using a **commercial ROV**. It integrates:

- A **learning-based pose controller** to maintain optimal ROV positioning using image gradients.
- A **CNN-based distance classification model** to support robust pose estimation.
- A **YOLO-based defect detector** for identifying net holes and plastic debris.

---

## Key Features

- **Gradient-aware pose control** using CNNs for desired inspection distance.
- **Dual-sided gradient regulation** for precise yaw control.
- **YOLOv5-based net defect detection** (holes, plastic debris).
- **Tested in both pool and real fish farm environments**.

<p align="center">
  <img src="media/methodology.png" alt="Methodology Diagram" width="600"/>
</p>

<p align="center"><em>
Proposed block diagram of the system. The ROV receives camera input and executes inspection tasks, while the topside server handles net defect detection and control signal generation. Deep learning is employed for defect classification, and pre-defined setpoints for depth and yaw are sent to the ROV for motion control.
</em></p>

---

## 📁 Repository Structure

```
rov_gradiant_control/
├── distance_classification/     # CNN-based distance classifier (MobileNetV2, etc.)
│   ├── dataset/                 # Net images for training (annotated)
│   ├── MobileNetV2/                 # Saved PyTorch models
│   └── mobileNetV2.pth
├── defect_detection/       # YOLOv5 model and dataset
│   ├── dataset/                   # Labeled images and annotations (holes, plastic)
|   ├── Test data/                   # Sample videos for testing the trained model
│   ├── detection model/yolov5/                 # YOLOv5 trained weights
│   └── best.pth
├── disance_control/               # Control module for ROV pose 
│   ├── distance_controller.py

```

---

##  Getting Started

### Requirements

- Python 3.8+
- PyTorch
- OpenCV
- ROS2 (Foxy or Humble)
- YOLOv5 repo (https://github.com/ultralytics/yolov5)
- Blueye Python SDK (if deploying on Blueye ROV)

### Installation

```bash
# Clone the repo
git clone https://github.com/drwa92/rov_gradiant_control.git
cd rov_gradiant_control

```

---



##  Distance Classification Module

This module utilizes the publicly available [MobileNetV2](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html) architecture to classify the ROV's inspection distance relative to the aquaculture net.

### Provided Assets

- **Trained Model:** [`mobilenetv2.pth`](distance_classification/MobileNetV2/mobilenetv2_best.pth)
- **Custom Dataset:** Located in [`distance_classification/dataset`](distance_classification/dataset), annotated with three distance classes:
  - `Far`
  - `Close`
  - `Good`


### Inference Example

To run predictions using the trained model:

```bash
python distance_classification.py
```
---

## Net Defect Detection Module

This module uses a custom-trained **YOLOv5** model to detect **net defects**, specifically *holes* and *plastic debris*. The code supports input from videos, webcam, or a live ROV stream.

### Folder Structure

- `Test data/`: Contains sample videos to test the trained model.
- `dataset/`: Includes the custom dataset used for training, with annotations for *holes* and *plastic*.



###  Setup

1. Clone the YOLOv5 repository:

```bash
cd defect_detection
git clone https://github.com/ultralytics/yolov5
```

### Running Inference

To run detection on a test video or a live ROV stream, edit the `capture_source` in `yolo_inference.py`:

Then launch the script:

```bash
python yolo_inference.py
```

This will open a window displaying detections with bounding boxes and FPS.

---

##  Real-Time ROV Control Module

Ensure your Blueye ROV is connected and the camera stream is available.

```bash
Python distance_controller.py
```

This script reads camera input, estimates distance/yaw via CNN+gradient, and sends movement commands to the ROV.

---



## Sample Results

<p align="center">
  <img src="media/qual_close.jpg" alt="YOLOv5 Close Distance Qualitative Evaluation" width="800"/>
</p>

<p align="center"><em>
Qualitative evaluation of the YOLOv5 model for aquaculture net pens defect detection on close-distance dataset. The first two columns show the input and the corresponding detection results for a close-distance dataset. The third column shows the corresponding gradient information of the input image.
</em></p>
---

##  Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{akram2025rov,
  title={Active vision-based real-time aquaculture net pens inspection using ROV},
  author={Akram, Waseem and Din, Muhayy Ud and Heshmat, Mohamed and Casavola, Alessandro and Seneviratne, Lakmal and Hussain, Irfan},
  journal={},
  year={2025}
}
```

---

## 🤝 Acknowledgements

- This work is supported by **Khalifa University** under Award Nos. RC1-2018-KUCARS-8474000136, CIRA-2021-085, and MBZIRC-8434000194.  
Special thanks to **LABUST, University of Zagreb** for supporting sea trials and data collection.
