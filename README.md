# 01. Computer Vision foundations

Stage 1 of an applied research series on vision-enabled AI agents.

This repository explores the foundational building blocks of computer vision, from image preprocessing and data augmentation to inference and fine-tuning with modern pretrained models (CNNs, Vision Transformers, YOLO, etc.).

The aim is to establish a solid technical base for future stages, where these capabilities will be integrated into autonomous, vision-aware AI agents capable of perception, reasoning, and multimodal interaction.

## Objectives

1. Understand core computer vision workflows (preprocessing, inference, evaluation).
2. Work hands-on with PyTorch or TensorFlow and OpenCV.
3. Experiment with pretrained architectures (ResNet, ViT, YOLOv11/v12) and promptable segmentation with SAM2/SAM3.
4. Fine-tune a pretrained model on a custom dataset: transfer learning, choosing a loss function, handling class imbalance, and tracking experiments (Weights & Biases or MLflow).
5. Build a real annotation and data pipeline: labeling tools (CVAT, Label Studio), active learning to prioritize what gets labeled next, and synthetic data generation for rare classes.
6. Evaluate rigorously: mAP, IoU, precision/recall curves, confusion matrices, and calibration, not just accuracy.
7. Build a modular inference pipeline that can later be extended by agentic reasoning systems.

## Tech stack

`Python` / `PyTorch` / `OpenCV` / `Hugging Face Transformers` / `Jupyter Notebooks` / `CVAT or Label Studio` / `Weights & Biases or MLflow`

## Milestones

1. **Image classification playground**: a CLI app that classifies or segments images locally.
2. **Fine-tune and evaluate**: fine-tune a model on a custom or deliberately imbalanced dataset, track the experiment, and publish a metrics report (mAP/IoU/precision-recall/calibration) in this README.

### Next Stage: Vision Embeddings Agent (`02-vision-embeddings-agent`)

The focus shifts from perception to representation: learning how to turn visual content into semantic embeddings that can be searched, compared, and reasoned over by AI systems.
