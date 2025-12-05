# Image Value Judge

A simple open-source project that analyzes a single image and outputs:
1. **Object Classification** – Identify what object appears in the image (HuggingFace ViT model).
2. **Damage Detection** – Measure the visual damage level using OpenCV.
3. **Value Judgement** – Provide a text-based interpretation based on the object and its damage level.

---

##  Features

- **HuggingFace Image Classification**
  - Extract the Top-1 label and confidence.
- **Damage Score Calculation**
  - Uses Laplacian variance to estimate surface cracks, scratches, or structural damage.
- **Value Judgement System**
  - Simple rule-based interpretation combining object type + damage level.

---

## Installation

Make sure Python 3.9+ is installed.

Install required libraries:

```bash
pip install opencv-python pillow transformers torch
