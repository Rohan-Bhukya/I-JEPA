# Self-Supervised Visual Understanding from  Masked Images( I-JEPA)

## Overview 
This project implements a self-supervised representation learning system based on the *Image-based Joint-Embedding Predictive Architecture (I-JEPA)*. The system learns rich visual representations by predicting the representations of masked target regions of an image from a single context block, operating entirely in latent/embedding space rather than pixel space.



## Table of Contents
- Objective
- Approach
- Dataset and preprocessing
- Architecture overview
- Results
- Applications
- Learning outcomes
- References

## Results

We performed Linear Probe and kNN over STL10 dataset on frozeen encoder, (5000 labeled training images , 8000 test images)

### STL-10 Evaluation Results

| **Model** | **Probe Train (%)** | **Probe Test (%)** | **k-NN Test (%)** |
|---|---:|---:|---:|
| random-init | 47.9 | 41.1 | 26.7 |
| jepa-ep10.pth.tar | 78.8 | 60.4 | 51.3 |
| jepa-ep100.pth.tar | 95.4 | 73.9 | 70.0 |

![Description(tsne_comparison-1.png))]

## Applications

| **Domain** | **Application** |
|---|---|
| Medical Imaging | Improving downstream clinical tasks such as lesion detection, anatomical segmentation, and data augmentation |
| Autonomous Systems | Learning visual representations for scene understanding and navigation |
| Robotics | Learning representations from noisy sensory signals to predict object states, such as slipping and grasping |
| Cardiac Imaging | EchoJEPA: Learning cardiac anatomy and dynamics from echocardiography |
| Remote Sensing | Analyzing satellite and aerial imagery |
| Image Retrieval | Learning representations for image and region retrieval |
## Learning Outcomes 

- Developed a Functional I-JEPA model that learns visual representations through self-supervised joint-embedding prediction.
- Implementated context encoder, EMA-updated target encoder, and predictor network.
- Implementated multi-block masking strategy and latent-space prediction objective.
- Pretraining pipeline producing transferable representations without labelled data
-Developed understanding about SSL, JEPA, ViT, EMA teacher-student training, masking strategies, latent-space prediction code implementation and evaluation.


## References 

1. [Official Meta AI blog](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
2. [I-JEPA Reaseach Paper](https://arxiv.org/pdf/2301.08243)
3. [PyTorch Official Documentation](https://pytorch.org/)
4. [Official implementation](https://github.com/facebookresearch/ijepa)


## Collaborators




