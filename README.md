# Acoustic Anomaly Detection for Industrial Pumps
This repository contains a Jupyter notebook that implements and compares several machine learning and deep learning models for detecting anomalies in industrial pump sounds using the MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset. The goal is to distinguish normal pump operation from abnormal (faulty) sounds.

## Overview
Acoustic anomaly detection plays a crucial role in predictive maintenance. This project explores various unsupervised and supervised approaches to identify anomalies in pump sounds. The workflow includes:

- Data loading and preprocessing

- Feature extraction (MFCCs and Mel‑spectrograms)

- Exploratory data analysis (EDA) of features

- Modeling with:

  - PCA (baseline reconstruction error)

  - Gaussian Mixture Model (GMM)

  - LSTM Autoencoder

  - CNN Autoencoder

- Evaluation using ROC curves, AUC, confusion matrices, and threshold optimization

- Error analysis to identify the most challenging acoustic features

## Dataset
The MIMII Pump Sound Dataset contains recordings of normal and abnormal pump operations.

- Normal samples: 381 files

- Abnormal samples: 138 files

Each audio file is 10 seconds long, sampled at 16 kHz. Features are extracted using:

- MFCCs (20 coefficients) for sequence models

- Mel‑spectrograms (64 mel bins) for CNN models

## Requirements
Install the required packages using:

```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn tensorflow kagglehub tqdm joblib
```
Main dependencies:

- TensorFlow / Keras (for deep learning models)

- Scikit‑learn (PCA, GMM, evaluation metrics)

- Librosa (audio feature extraction)

- Matplotlib / Seaborn (visualization)

- Kagglehub (to download the dataset automatically)

## Usage
The project is provided as a Jupyter notebook that can be run end‑to‑end. You can open it directly in Google Colab using the badge at the top of the notebook.

### Running locally
1. Clone this repository.

2. Install the required packages.

3. Launch Jupyter Notebook or JupyterLab and open Acoustic_Anomaly_Detection.ipynb.

4. Execute the cells sequentially – the notebook will automatically download the dataset using `kagglehub`.

## Models & Results
Four anomaly detection approaches were evaluated. Anomaly scores are derived from:

- *PCA*: reconstruction error after dimensionality reduction (95% variance retained → 261 components).

- *GMM*: negative log‑likelihood of a 4‑component Gaussian mixture fitted to the MFCC mean vectors.

- *LSTM Autoencoder*: sequence‑to‑sequence model trained to reconstruct MFCCs.

- *CNN Autoencoder*: convolutional autoencoder on Mel‑spectrograms.

### ROC Curves
The notebook generates ROC curves comparing all models. The Area Under the Curve (AUC) scores are:

|Model   | AUC      |
|--------|----------|
|PCA     |	0.795   |
|GMM     |	0.940   |
|LSTM_AE |	0.880   |
|CNN_AE  |	0.751   |

## Error Analysis
To understand which acoustic features are hardest to reconstruct, the per‑MFCC reconstruction error is examined. The notebook highlights the three coefficients with the highest average error, which may be more sensitive to anomalies and could be candidates for further feature engineering.

## Key Findings from Exploratory Data Analysis
MFCC coefficient distributions differ noticeably between normal and abnormal sounds.

Correlation structures among MFCCs also show distinct patterns, indicating that anomalies alter the spectral relationships.

Spectrogram frequency bins exhibit amplitude and correlation changes that are leveraged by the CNN autoencoder.

## References
MIMII Dataset: MIMII – Malfunctioning Industrial Machine Investigation and Inspection

Librosa: https://librosa.org/

Scikit‑learn: https://scikit-learn.org/

TensorFlow: https://www.tensorflow.org/

