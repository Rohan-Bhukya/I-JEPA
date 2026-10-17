# Self-Supervised Visual Understanding from  Masked Images (I-JEPA)

## Overview 
This project implements a self-supervised representation learning system based on the *Image-based Joint-Embedding Predictive Architecture (I-JEPA)*. The system learns rich visual representations by predicting masked target regions from a single context block, operating entirely in latent/embedding space rather than pixel space.


## Table of Contents
- Objective
- Approach
- Dataset and preprocessing
- Architecture overview
- Results
- Applications
- Learning outcomes
- References

## Objective
- Functional I-JEPA model that learns visual representations through self-supervised joint-embedding prediction.
- Implementation of the context encoder, EMA-updated target encoder, and predictor network.
- Implementation of the multi-block masking strategy and latent-space prediction objective.


## Dataset and Preprocessing

### Dataset:
<<<<<<< HEAD
1.  **Source:** STL-10 (Self-Taught Learning 10), a benchmark dataset from Stanford, inspired by CIFAR-10 and designed specifically for unsupervised and self-supervised representation learning.
=======
1.  **Source:** STL-10,  designed specifically for unsupervised and self-supervised representation learning.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b
- **Structure:** The dataset contains 113,000 color images at 96×96 resolution, split as follows:
    - Unlabeled set: 100,000 images used for self-supervised pretraining (I-JEPA training, no labels needed).
    - Training set: 5,000 labeled images (500 per class, 10 classes) used for linear probing and fine-tuning.
    - Test set: 8,000 labeled images (800 per class) used for final evaluation.
    - Classes: 10 object categories: airplane, bird, car, cat, deer, dog, horse, monkey, ship, and truck.
<<<<<<< HEAD
- Notes: The unlabeled set is drawn from a broader distribution than the labeled classes (it includes additional animals and vehicles), which makes it a good test of how well the learned representations generalize. Images are sourced from ImageNet.

2. ImageNet-1K (Large-Scale Dataset)

- **Source:** ImageNet Large Scale Visual Recognition Challenge (ILSVRC-2012), the standard benchmark used in the original I-JEPA paper.
- **Structure:** The dataset contains about 1.4 million color images of varying resolution (typically resized to 224×224 for ViT input), split as follows:
  - Training set: about 1.28 million images (roughly 1,300 per class), used for self-supervised pretraining (labels ignored).
  - Validation set: 50,000 labeled images (50 per class), used for linear probing and final evaluation.
  - Classes: 1,000 object categories covering animals, vehicles, household objects, food, and more.
- Notes: With 224×224 inputs and 16×16 patches, each image becomes a 14×14 grid of 196 patches. Because of its scale and diversity, ImageNet-1K is the standard test of whether the learned representations transfer to a wide range of categories.


=======
    - 
2. ImageNet-100
- Structure: ImageNet-100 is a 100-class subset of ImageNet, containing approximately 130,000 images, split as follows:
- Training set: Approximately 120,000 labeled images across 100 classes, used for training or self-supervised pretraining (labels are ignored during I-JEPA pretraining).
- Validation set: Approximately 5,000 images used for model evaluation and hyperparameter selection.
- Test set: The exact test split depends on the dataset version; many ImageNet-100 implementations use the validation set for final evaluation.
- Classes: 100 object categories selected from the original ImageNet-1K dataset.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b

### Preprocessing Pipeline:

- Data Loading & Parsing
    - Loaded STL-10 splits (unlabeled, train, test) using torchvision.datasets.STL10.
    - Used the unlabeled split (100,000 images) for pretraining, with no labels required.
    - Used the labeled train/test splits (5,000 / 8,000 images) only for downstream evaluation.
- Image Preprocessing
<<<<<<< HEAD
    - Applied light augmentation: random resized cropping and horizontal flipping. I-JEPA does not rely on heavy hand-crafted augmentations such as color jitter or rotation, since the masking strategy provides the learning signal.
=======
    - Applied light augmentation: random resized cropping. I-JEPA does not rely on heavy hand-crafted augmentations such as color jitter or rotation, since the masking strategy provides the learning signal.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b
    - Converted images to tensors and normalized them to the range [-1, 1].
-  Patchification
    - Split each image into non-overlapping patches (e.g., 8×8 for 96×96 images, giving a 12×12 grid of 144 patches or 16x16 for 224x224 images, 196 patches).
    - Linearly projected patches into embeddings and added positional encodings.
    - Stored the patch grid dimensions for use by the masking module.
- Multi-Block Mask Generation
    - Sampled one large context block and several (typically 4) target blocks per image, using predefined scale and aspect-ratio ranges.
    - Removed any overlap between the context block and target blocks, so the target regions are never visible to the context encoder.
    - Produced index tensors for context patches and target patches, saved per sample.
<<<<<<< HEAD
- Data Loader
    - Returned (image, context mask indices, target mask indices) per sample for pretraining.
    - Implemented a MaskCollator to handle varying numbers of context and target patches across samples by cropping to the minimum count in the batch, so tensors can be stacked.
    - Used a standard (image, label) loader for linear probing and evaluation, with no masking applied.
=======
- Data Loader and Collator
    - Returned (image, context mask indices, target mask indices) per sample for pretraining.
    - Implemented a MaskCollator to handle varying numbers of context and target patches across samples by cropping to the minimum count in the batch, so tensors can be stacked.
- Evaluation Data
    - Used the labeled STL-10 training and test splits (5,000 and 8,000 images) for downstream evaluation.
    - Also supported ImageNet-100, a 100-class subset of ImageNet, for evaluating the learned visual representations on a larger and more diverse dataset.
    - Used the pretrained encoder to extract image representations and evaluated them using a linear classifier trained on labeled data.
    - Kept labels out of self-supervised pretraining; labels were used only for downstream evaluation.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b

## Architecture Overview

I-JEPA is a single-stage self-supervised model made of three networks trained jointly. The pretrained encoder is then evaluated separately on downstream tasks.

| Component | Type | Function |
|-----------|------|----------|
| Context Encoder | ViT (trainable) | Encodes the visible context patches into representations |
| Target Encoder | ViT (EMA of context encoder, no gradients) | Encodes the full image to produce prediction targets |
| Predictor | Narrow ViT (trainable) | Predicts target-block representations from context representations and positional mask tokens |

### Combined I-JEPA Pipeline

```
Image → Patchify → Context Block (visible patches) → Context Encoder → Context Representations
Image → Patchify → Full Image → Target Encoder (EMA) → Target Representations
Context Representations + Positional Mask Tokens → Predictor → Predicted Target Representations
```

---

### Pretraining Model

### Components

**Context Encoder (ViT)**
- Vision Transformer (e.g., ViT-Small or ViT-Base) trained from scratch on the STL-10 unlabeled split.
- Splits the 96×96 image into 8×8 patches and adds positional embeddings.
<<<<<<< HEAD
- Processes **only the visible context patches** through transformer blocks.
=======
- Processes only the visible context patches through transformer blocks.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b
- Outputs a context representation for each visible patch.

**Target Encoder (ViT)**
- Identical architecture to the context encoder.
<<<<<<< HEAD
- Weights are an **Exponential Moving Average (EMA)** of the context encoder, with no gradient updates.
- Processes the **full image** and outputs patch-level representations.
=======
- Weights are an Exponential Moving Average (EMA) of the context encoder, with no gradient updates.
- Processes the full image and outputs patch-level representations.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b
- Representations at the target block locations serve as the prediction targets.

**Predictor (Narrow ViT)**
- Lightweight transformer with a smaller embedding dimension and fewer layers than the encoders.
<<<<<<< HEAD
- Takes context representations plus learnable **mask tokens** carrying the positional embeddings of the target patches.
=======
- Takes context representations plus learnable mask tokens carrying the positional embeddings of the target patches.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b
- Outputs predicted representations for each target patch.

### Intuitive Explanation

<<<<<<< HEAD
The model learns to "fill in the blanks" in **representation space** rather than pixel space. Given a partial view of an image (the context), it predicts what the abstract features of the hidden regions (the targets) should look like. For example, if the context shows the body and legs of a horse, the predictor should produce an embedding for the hidden head region that is semantically consistent with "horse head", without needing to reconstruct exact pixels, textures, or colors. This pushes the encoder toward high-level semantics instead of low-level detail.
=======
The model learns to "fill in the blanks" in representation space rather than pixel space. Given a partial view of an image (the context), it predicts what the abstract features of the hidden regions (the targets) should look like. For example, if the context shows the body and legs of a horse, the predictor should produce an embedding for the hidden head region that is semantically consistent with "horse head", without needing to reconstruct exact pixels, textures, or colors. This pushes the encoder toward high-level semantics instead of low-level detail.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b

### Mathematical Formulation

**Context Encoder:**

<<<<<<< HEAD
$$s_x = f_\theta(x_{\text{context}})$$

**Target Encoder (EMA):**

$$s_y = \text{sg}\big(\bar{f}_{\bar\theta}(y)\big), \qquad \bar\theta \leftarrow m\,\bar\theta + (1-m)\,\theta$$

where $\text{sg}(\cdot)$ is the stop-gradient operator and $m$ is the EMA momentum (typically close to 1, e.g., 0.996 → 1.0).

**Predictor:**

$$\hat{s}_y(i) = g_\phi\big(s_x,\ \{m_j\}_{j \in B_i}\big)$$

where $B_i$ is the $i$-th target block and $m_j$ are mask tokens with positional embeddings for patch $j$.

**Loss Function (Smooth L1 Latent Prediction Loss):**

$$\mathcal{L}(\theta,\phi) = \frac{1}{M}\sum_{i=1}^{M}\ \sum_{j \in B_i} \text{smooth}_{L1}\big(\hat{s}_y(j) - s_y(j)\big)$$

where the smooth L1 function is applied element-wise over the embedding dimensions:

$$
\text{smooth}_{L1}(z) =
\begin{cases}
\dfrac{1}{2\beta} z^2, & \text{if } |z| < \beta \\[6pt]
|z| - \dfrac{\beta}{2}, & \text{otherwise}
\end{cases}
$$

and summed over the $D$ embedding dimensions of each patch. Here $M$ is the number of target blocks (typically 4) and $\beta$ is the transition threshold (default $\beta = 1$).

Smooth L1 behaves like L2 for small errors (smooth gradients near zero) and like L1 for large errors (less sensitive to outliers), which makes training more stable.

=======
```math
s_x = f_\theta(x_{\text{context}})
```

**Target Encoder (EMA):**

```math
$$s_y = \mathrm{sg}(f_{\bar{\theta}}(y))$$
```

The target encoder parameters are updated using Exponential Moving Average (EMA):

```math
\bar{\theta} \leftarrow m\bar{\theta} + (1-m)\theta
```

where `sg(·)` is the stop-gradient operator and \(m\) is the EMA momentum, typically close to 1 (e.g., increasing from 0.996 to 1.0).

**Predictor:**

```math
\hat{s}_y^{(i)} = g_\phi(s_x, m_j \mid j \in B_i)
```

The predictor uses the context representations and mask tokens with positional embeddings to predict the representations of the target regions.

Here, \(B_i\) is the \(i\)-th target block, and \(m_j\) represents a mask token associated with target patch \(j\).

**Loss Function (L2 Latent Prediction Loss):**

```math
\mathcal{L}(\theta,\phi)
=
\frac{1}{M}
\sum_{i=1}^{M}
\frac{1}{|B_i|D}
\sum_{j \in B_i}
\left\|
\hat{s}_y^{(j)} - s_y^{(j)}
\right\|_2^2
```

where:

- \(M\) is the number of target blocks (typically 4).
- \(B_i\) is the set of patches in target block \(i\).
- \(D\) is the embedding dimension.
- \$$\hat{s}_y^{(j)}$$ is the predicted representation of target patch \(j\).
- \$$s_y^{(j)}$$ is the corresponding target encoder representation.

L2 loss measures the squared difference between predicted and target embeddings. Larger errors receive a stronger penalty because the error is squared.

The objective is to make the predicted embeddings of masked regions match the target encoder's embeddings. This encourages the context encoder to learn useful visual representations by capturing relationships between visible and masked regions.

The asymmetric architecture, predictor, stop-gradient operation, and EMA target encoder help stabilize representation learning without requiring negative samples.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b
This ensures that the predicted embeddings of masked regions match the target encoder's embeddings of those regions, forcing the context encoder to capture meaningful semantic structure. Collapse is avoided through the asymmetric design (predictor + EMA + stop-gradient) rather than negative samples.

---

## Downstream Evaluation Model

**Linear Probe**
<<<<<<< HEAD
- The pretrained target (or context) encoder is **frozen**.
=======
- The pretrained target (or context) encoder is frozen.
>>>>>>> b059187dea2b326b4521e8d9afb86327af1e0a8b
- Patch representations are average-pooled into one global image vector.
- A single linear layer maps the vector to the 10 STL-10 (or 1000 ImageNet) classes.
- Trained with cross-entropy on the labeled training images and evaluated on the test images.

**Optional Fine-Tuning**
- The encoder is unfrozen and trained end-to-end with a small learning rate along with the classification head.

---


## Results


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
- Implemented a context encoder, EMA-updated target encoder, and predictor network.
- Implemented a multi-block masking strategy and latent-space prediction objective.
- Pretraining pipeline producing transferable representations without labelled data
-Developed understanding of SSL, JEPA, ViT, EMA teacher-student training, masking strategies, latent-space prediction code implementation, and evaluation.


## References 

1. [Official Meta AI blog](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
2. [I-JEPA Research Paper](https://arxiv.org/pdf/2301.08243)
3. [PyTorch Official Documentation](https://pytorch.org/)
4. [Official implementation](https://github.com/facebookresearch/ijepa)



