# Road-Damage-Detection-using-GAN
A hybrid deep learning system for pothole detection that combines GAN-based synthetic data generation, context-aware augmentation, and real-time object detection using YOLOv8s.

This project focuses on solving the problem of limited training data by generating realistic pothole images and improving dataset diversity.
Overview

Pothole detection is crucial for smart city infrastructure and road safety. Traditional approaches suffer due to lack of diverse datasets. This project introduces a data-centric pipeline that enhances performance by improving training data rather than only model complexity.
Project Pipeline-
Real Dataset
     ↓
Pothole Extraction
     ↓
Train GAN (DCGAN)
     ↓
Generate Synthetic Potholes
     ↓
Collect Clean Roads
     ↓
Context-Aware Augmentation
     ↓
Merge Dataset
     ↓
Train YOLOv8s
     ↓
Detection
Core Techniques:
GAN-Based Data Generation-
Uses DCGAN to create new pothole samples
Improves dataset diversity
Context-Aware Augmentation-
Irregular mask blending
Brightness matching
Edge smoothing
Realistic placement on roads
YOLOv8s Detection-
Lightweight and efficient
Supports real-time inference
Transfer learning enabled
Results:
Improved detection accuracy
Better generalization to unseen road conditions
Reduced overfitting compared to real-only dataset
