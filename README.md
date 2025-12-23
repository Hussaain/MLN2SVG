# MLN2SVG: Domain-aware spatially variable gene detection using contrastive variational autoencoder and multi-level neighbor

MLN2SVG is a deep learning framework designed for the integrative analysis of spatial transcriptomics (ST) data. It leverages multi-level neighborhood modeling and graph-based learning to identify spatial domains and spatially variable genes (SVGs) with high precision and robustness across multiple ST platforms.

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Hussaain/MLN2SVG.git
cd MLN2SVG

2️⃣ Create and activate a conda environment
bash
conda create -n mln2svg python=3.8
conda activate mln2svg

📦 Dependencies
Package	Version
Python	3.8.0
NumPy	1.22.4
pandas	2.0.3
AnnData	0.9.2
Scanpy	1.9.8
Squidpy	1.2.3
SciPy	1.10.1
scikit-learn	1.3.2
matplotlib	3.7.5
numba	0.55.2
natsort	8.4.0
wordcloud	1.9.4
rpy2	3.5.17
PyTorch	2.4.0 + CUDA 11.8
Tested on Linux with NVIDIA H800 GPU


📊 Datasets
All datasets used in this study are publicly available:

Dataset	Technology	Source
Human DLPFC	Visium	spatialLIBD
Human Breast Cancer	Visium	10x Genomics
Mouse Brain	Visium	10x Genomics
Mouse Olfactory Bulb	Stereo-seq	STOmics
Mouse Embryogenesis	SeqFISH	CRUK Spatial Atlas
Mouse Visual Cortex	STARmap


Detailed dataset descriptions are provided in Supplementary Table 1 of the manuscript.

🖥️ Computing Environment
All experiments were conducted on a Linux workstation with:

OS: Ubuntu 22.04

GPU: NVIDIA H800 (84.9 GB)

CUDA: 11.8

Python: 3.8.0
