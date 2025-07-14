# 🐠 Active Vision-Based Real-Time Aquaculture Net Pen Inspection Using ROV

This repository contains the source code, models, and scripts used in the paper:

> **Waseem Akram et al.**, *"Active vision-based real-time aquaculture net pens inspection using ROV"*, Scientific Reports, 2025.

---

## 📌 Overview

This project presents a novel **gradient-aware active vision system** for aquaculture net inspection using a **commercial ROV**. It integrates:

- A **learning-based pose controller** to maintain optimal ROV positioning using image gradients.
- A **CNN-based distance classification model** to support robust pose estimation.
- A **YOLO-based defect detector** for identifying net holes and plastic debris.
- A real-time control loop implemented in **ROS2**.

---

## 🎯 Key Features

- 🔍 **Gradient-aware pose control** using CNNs for desired inspection distance.
- 🎯 **Dual-sided gradient regulation** for precise yaw control.
- 🧠 **YOLOv5-based net defect detection** (holes, plastic debris).
- 📡 **Real-time ROS2 interface** to control and monitor a Blueye Pro ROV.
- 🧪 **Tested in both pool and real fish farm environments**.

---

## 📁 Repository Structure

```
rov_gradiant_control/
├── distance_classification/     # CNN-based distance classifier (MobileNetV2, etc.)
│   ├── dataset/                 # Net images for training (annotated)
│   ├── models/                 # Saved PyTorch models
│   └── train_distance_model.py
├── net_defect_detection/       # YOLOv5 model training and inference
│   ├── data/                   # Labeled images and annotations (holes, plastic)
│   ├── yolov5/                 # YOLOv5 training framework
│   └── detect.py
├── ros2_control/               # ROS2 interface and control loop
│   ├── depth_controller.py
│   ├── yaw_controller.py
│   └── pose_estimation_node.py
├── visualization/              # Tools for plotting gradient and yaw error
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch
- OpenCV
- ROS2 (Foxy or Humble)
- YOLOv5 dependencies (`requirements.txt` inside `yolov5/`)
- Blueye Python SDK (if deploying on Blueye ROV)

### Installation

```bash
# Clone the repo
git clone https://github.com/drwa92/rov_gradiant_control.git
cd rov_gradiant_control

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Training Models

### 📏 Distance Classification

```bash
cd distance_classification
python train_distance_model.py --model MobileNetV2 --epochs 50
```

### 🕳️ Net Defect Detection (YOLOv5)

```bash
cd net_defect_detection/yolov5
python train.py --img 416 --batch 16 --epochs 50 --data net.yaml --weights yolov5s.pt
```

---

## 📡 Real-Time ROS2 Interface

Ensure your Blueye ROV is connected and the camera stream is available.

```bash
ros2 run ros2_control pose_estimation_node.py
```

This node reads camera input, estimates distance/yaw via CNN+gradient, and sends PID control commands to the ROV.

---

## 📊 Evaluation

Use the `visualization/` scripts to generate error plots for:

- Distance convergence (gradient-based)
- Yaw alignment performance
- Detection accuracy (mAP, precision, recall)

---

## 📸 Sample Results

| Distance Adjustment | Defect Detection |
|---------------------|------------------|
| ![](media/distance_control.gif) | ![](media/defect_detection.png) |

---

## 📄 Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{akram2025rov,
  title={Active vision-based real-time aquaculture net pens inspection using ROV},
  author={Akram, Waseem and Din, Muhayy Ud and Heshmat, Mohamed and Casavola, Alessandro and Seneviratne, Lakmal and Hussain, Irfan},
  journal={Scientific Reports},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This work is supported by **Khalifa University** under Award Nos. RC1-2018-KUCARS-8474000136, CIRA-2021-085, and MBZIRC-8434000194.  
Special thanks to **LABUST, University of Zagreb** for supporting sea trials and data collection.
