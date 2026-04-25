# CLIP for Medical Image Understanding — Fine-Tuning on ROCO

Exploring OpenAI's [CLIP](https://github.com/openai/CLIP) model for medical image–text matching, with fine-tuning on the [ROCO](https://github.com/razorx89/roco-dataset) radiology dataset.

*Notebook developed in 2024 in collaboration with Xiaoyang Wei, PhD student at LIPADE, Université Paris Cité.*

## Objective

Evaluate how well CLIP understands medical images out-of-the-box, then fine-tune it on radiology data (ROCO) and measure the improvement in image–text alignment.

## What This Notebook Covers

### 1. CLIP on natural images
- Load `ViT-B/16` and compute cosine similarity between 8 natural images (from scikit-image) and their text descriptions
- Visualize the similarity matrix to confirm CLIP works well on everyday images

### 2. CLIP on medical images (before fine-tuning)
- Test the same model on 8 ROCO radiology images (CT scans, MRI, X-rays, angiograms)
- Show that CLIP struggles with medical domain — low similarity scores on the diagonal

### 3. Zero-shot classification
- Zero-shot image classification on CIFAR-100 and CIFAR-10 using cosine similarity as logits
- Evaluate top-1 accuracy without any task-specific training

### 4. Fine-tuning on ROCO
- Custom `CLRDataset` class for loading ROCO images and captions
- Contrastive learning objective (symmetric cross-entropy on image–text pairs)
- Training with Adam optimizer + cosine annealing, batch size 16, 10 epochs, lr = 3e-6

### 5. CLIP on medical images (after fine-tuning)
- Re-compute the similarity matrix on the same ROCO samples
- Show improved diagonal alignment after domain adaptation

### 6. Model comparison
- Compare all available CLIP architectures (RN50, RN101, ViT-B/32, ViT-B/16, ViT-L/14, etc.)
- Report parameter count, input resolution, context length for each

## Datasets

| Dataset | Type | Size | Usage |
|---------|------|------|-------|
| [ROCO](https://github.com/razorx89/roco-dataset) | Radiology images + captions | ~80K images | Fine-tuning + evaluation |
| CIFAR-100 | Natural images (100 classes) | 60K images | Zero-shot classification |
| CIFAR-10 | Natural images (10 classes) | 60K images | Zero-shot classification |
| scikit-image samples | Natural images | 8 images | Baseline similarity demo |

## Tech Stack

Python, PyTorch, OpenAI CLIP, torchvision, scikit-image, Matplotlib, Google Colab (GPU)

## Key Results

- CLIP (ViT-B/16) achieves strong image–text matching on natural images but shows poor alignment on medical radiology images
- After fine-tuning on ROCO, the similarity scores on the diagonal improve significantly, demonstrating successful domain adaptation
- Zero-shot classification on CIFAR-10 confirms CLIP's strong generalization on natural image benchmarks
