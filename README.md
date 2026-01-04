# CT-Segmentation

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository transitions from the physical foundations of CT reconstruction to clinical image analysis. 

## Project Overview
While traditional reconstruction focuses on image quality, this project focuses on automated information extraction. Usage of the Medical Segmentation Decathlon (MSD) dataset ([link to paper](https://arxiv.org/pdf/1902.09063)).

## Data Analysis
| Visualization | Description | Plot |
| :-- | :-- | :--- |
| **Sample** | **Left** sample w/o pancreas segmentation mask <br>**Right:** sample with pancreas segmentation mask  | *![sample](docs/assets/pancreas_sample.png)* |
| **Windowing** | **Left** Intensity distribution incl. pancreas window <br>**Right:** Intensity Filtered slice| *![windowing](docs/assets/windowing.png)* |
| **Boundary Intensity Overlap Analysis** | **brown** pancreas intensity <br>**grey:** Intensity of adjacent objects/ intensity of boundaries | *![boundary](docs/assets/boundary_intensity.png)* |

## Segmentation "Journey"
| Visualization | Description | Plot | Interpretation |
| :-- | :-- | :--- | :--- |
| **[Baseline](notebooks/02a_seg_baseline.ipynb)** | **Left** Original CT slice <br>**Mid:** Naive Thresholding <br>**Right** Ground truth  | *![sample](docs/assets/baseline_model.png)* | too naive -> PCA |
| **[PCA](notebooks/02b_seg_pca.ipynb)** | **PCA** PC1-PC3 <br> | *![sample](docs/assets/pca_model.png)* | high explainability in PC1-PC3 BUT significant overlap -> advanced methods|
| **[3D U-Net](notebooks/02a_seg_baseline.ipynb)** | **Left** Original CT slice <br>**Mid:** Naive Thresholding <br>**Right** Ground truth  | *![sample](docs/assets/baseline_model.png)* | too naive -> PCA |

## Repository Structure
* `src/`: Code for data pipelines, model architectures, and training.
* `notebooks/`: Data Analysis and qualitative model evaluation.
* `tests/`: Tests for tensor shapes and data consistency.
* `data/`: Storage for data.

## Installation & Usage

### Prerequisites
This project uses **uv** for dependency management. Install it via:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

## Setup
```bash
git clone https://github.com/yannikFruehwirth/ct-segmentation-monai
cd ct-segmentation-monai
uv sync
```

## Data Preparation
```bash
uv run src/data/download_data.py
```
