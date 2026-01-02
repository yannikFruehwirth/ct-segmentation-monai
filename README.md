# CT-Segmentation

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository transitions from the physical foundations of CT reconstruction to clinical image analysis. 

## Project Overview
While traditional reconstruction focuses on image quality, this project focuses on automated information extraction. Usage of the Medical Segmentation Decathlon (MSD) dataset ([link to paper](https://arxiv.org/pdf/1902.09063)).


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
