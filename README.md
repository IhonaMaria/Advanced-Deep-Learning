# Advanced-Deep-Learning
This repository includes the main practical sessions developed during my Master's deep learning course. 
The code also contains reasoning and some theoretical concepts.
All the code has been done in Python using Google Collab in collaboration with my collegue Lloyd Linton Jones.
Exercises have been proposed by MHEDAS professors.

The contents covered in this repository are the following:

## YOLO object detection
YOLO (You Only Look Once) is a fast and accurate object detection algorithm. Instead of scanning images piece by piece, it looks at the entire image at once and predicts bounding boxes and class probabilities in a single step. YOLOv5, the version used here, improves on earlier versions with better speed and performance, making it great for tasks like real-time detection in medical imaging.
This session focused on using YOLOv5 to detect blood cells in microscope images from the BCCD dataset. The dataset contains high-quality images of different blood cell types, making it perfect for this task. Steps included preparing the data, training the YOLOv5 model on GPU, and visualizing the results with clean, well-documented code. The goal was to learn object detection techniques while ensuring reusable and easy-to-understand implementations.

## Generative Adversarial Networks
Generative Adversarial Networks (GANs) are powerful tools for generating synthetic data by training two neural networks—a generator and a discriminator—in opposition to one another. In medical imaging, GANs are used to create synthetic datasets that can help train AI models for tasks like lesion classification, segmentation, and detection. This session uses MediGAN, a Python library offering pretrained generative models for medical image synthesis, making it easier to generate and work with synthetic data.
This session explored MediGAN, demonstrating how to use it for medical image synthesis. We learned to access and implement pretrained models to generate synthetic datasets, which can enhance AI model training and adaptation in clinical tasks. 

## Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used to provide visual explanations for decisions made by Convolutional Neural Networks (CNNs). By highlighting regions in an image that contribute most to a model's prediction, Grad-CAM enhances model interpretability, making it especially useful in applications like medical imaging where trust and transparency are critical. This session combines Grad-CAM with dimensionality reduction techniques like PCA, T-SNE, and UMAP to visualize and analyze intermediate layer outputs of CNNs.
In this lab, we explored Grad-CAM for generating visual explanations of CNN classification results. The practical tasks included feature extraction from CNN layers, visualizing these features with dimensionality reduction techniques, and applying Grad-CAM to localize important regions in images. The session emphasized reusable coding practices, detailed visualizations with proper labels and legends, and the use of GPU to accelerate computations.

## Vision Transformers
Vision Transformers (ViTs) are an innovative approach to computer vision, applying the transformer architecture—originally designed for NLP—to image analysis. Instead of convolutions, ViTs split images into patches and process them as sequences, enabling effective global context understanding. The practical sessions focused on implementing and training Vision Transformers. In the first part, we explored the basics of ViTs using pretrained models, including preparing datasets with train-test-validation splits, applying data augmentation techniques, and designing a classification pipeline. The second part emphasized fine-tuning pretrained ViTs for specific tasks, optimizing hyperparameters (image size, batch size), and visualizing training results.

## UNET segmentation
U-Net is a popular deep learning architecture designed for image segmentation tasks. It uses an encoder-decoder structure to capture context and precise localization, making it ideal for medical imaging. This session focused on segmenting CT scans using a U-Net-like architecture, enabling us to learn how to extract meaningful regions such as lung areas and infections from medical images.
We worked on segmenting CT scans of COVID-19 patients, using a dataset containing expert-labeled segmentations of lungs and infections. The lab involved preprocessing data, training the U-Net model on GPU for faster computation, and evaluating performance with appropriate metrics. Key tasks included writing modular and reusable code, documenting steps thoroughly, and visualizing results with well-labeled plots. The session highlighted the importance of annotations in medical imaging and provided insights into evaluating segmentation performance.

