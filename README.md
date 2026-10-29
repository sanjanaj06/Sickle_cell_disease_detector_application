---
title: Sickle Cell Disease Detection & Risk Assessment
emoji: 🔬
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Sickle Cell AI Diagnostic & Clinical Risk Assessment System

An end-to-end medical AI diagnostic platform combining Deep Learning, Explainable AI, and Computer Vision for peripheral blood smear analysis.

## Features
- **Deep Learning Classification:** MobileNetV1 CNN model for binary disease classification.
- **Explainability (XAI):** Real-time Grad-CAM attention heatmaps highlighting decision-driving regions.
- **Cell Morphometry:** OpenCV Watershed segmentation for automated RBC counting, sickle cell flagging, and cell clustering density.
- **Clinical Risk Engine:** Image-Based Severity Index (IBSI), age-stratified Hemoglobin ranges, and Vaso-Occlusion Risk Score (VORS).
- **Exportable PDF Reports:** One-click clinical report export for medical records.
