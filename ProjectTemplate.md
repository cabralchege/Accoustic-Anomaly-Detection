# Project Roadmap: Sikiza-Maji (Acoustic Anomaly Detection for Water Pumps)

## 1. Executive Summary
**Objective:** Develop a Machine Learning system to detect mechanical failures (cavitation, clogging, bearing wear) in water pumps using acoustic data.
**Approach:** Unsupervised Anomaly Detection. We will train models on "Healthy" pump sounds and flag any sound that deviates from this norm as "Anomalous."
**Dataset:** MIMII (Malfunctioning Industrial Machine Investigation and Inspection) - Pump Subset.

---

## Phase 1: Data Acquisition & Understanding

### 1.1 The Data Source
* **Action:** Download the MIMII dataset (specifically the `6dB` pump subset).
* **Structure:** The dataset contains `.wav` audio files separated into `normal` (thousands of files) and `abnormal` (hundreds of files).

> **DS Decision Note: Why MIMII?**
> In a real-world scenario, collecting labeled failure data is expensive and dangerous (you have to break a machine to record it). MIMII provides a "Gold Standard" proxy with specific fault labels (clogging, leakage) that allows us to benchmark our model's performance before deploying it on real Kenyan boreholes.

### 1.2 The Challenge
* **Class Imbalance:** We have far more healthy data than broken data.
* **Domain Shift:** Background noise differs between recording sessions.

---

## Phase 2: Feature Engineering (From Sound to Data)

We cannot feed raw audio waves into a standard model efficiently. We must convert physics (sound) into features (math).

### 2.1 Audio Preprocessing
* **Sampling Rate:** Downsample audio to 16kHz.
    * **Reason:** Human speech goes up to 8kHz (Nyquist theorem requires 16kHz sampling). Pump mechanical faults (humming, grinding) usually happen below 10kHz. Using 44.1kHz (music quality) adds computational cost without adding relevant information.
* **Duration:** Fix all clips to exactly 10 seconds.
    * **Reason:** ML models require fixed-size input tensors.

### 2.2 Feature Extraction: Log-Mel Spectrograms
* **Action:** Convert audio time-series to Mel-Spectrograms using `Librosa`.
* **Parameters:** `n_mels=64`, `hop_length=512`.

> **DS Decision Note: Why Spectrograms over Raw Waveforms?**
> Raw audio is high-dimensional and noisy. A spectrogram captures both **Time** (X-axis) and **Frequency** (Y-axis), revealing patterns like "periodic clicking" or "sustained high-pitch whining" that raw numbers hide.
>
> **DS Decision Note: Why "Mel" Scale?**
> The Mel scale compresses frequencies to match human hearing (more resolution at low frequencies). While pumps aren't humans, mechanical vibrations (RPM) tend to be dominant in lower frequencies, making the Mel scale efficient for capturing the "hum" of the motor.

### 2.3 Logarithmic Scaling
* **Action:** Apply `Log (dB)` scaling to the spectrogram.
* **Reason:** Sound volume follows a power law. A screech might be 1000x louder than a hum in energy terms, but only 2x "louder" in perception. Log scaling compresses these huge variances so the model doesn't ignore quiet but important sounds.

---

## Phase 3: Data Preprocessing (Crucial for School)

### 3.1 Normalization/Scaling
* **Selection:** **StandardScaler** (Zero Mean, Unit Variance).
* **Action:** Fit the scaler *only* on the Training Data (Healthy), then transform the Test Data.

> **DS Decision Note: Why StandardScaler vs. MinMaxScaler?**
> * **MinMaxScaler** squashes data between 0 and 1. If we have one loud "pop" (outlier) in our training data, it will squash all normal sounds to 0.001, destroying the signal.
> * **StandardScaler** (Z-score) centers the data around 0. Neural Networks converge faster when inputs are centered and have a variance of 1. It is far more robust to outliers in audio data.

---

## Phase 4: Baseline Modeling (The "Simple" Approach)

Before using Deep Learning, we must prove we need it. We start with a statistical baseline.

### 4.1 Model: PCA + Mahalanobis Distance
* **Concept:** Use Principal Component Analysis (PCA) to reduce the spectrogram features into lower dimensions (e.g., top 10 components).
* **Anomaly Detection:** Calculate the distance of a new sound from the center of the "Healthy" cluster.
* **Why:** PCA is computationally cheap and mathematically transparent. If PCA achieves 90% accuracy, using a deep neural network is over-engineering.

### 4.2 Metrics for Evaluation
* **Selection:** **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**.
* **Why:** We cannot use "Accuracy". If 99% of pumps are healthy, a model that says "All Healthy" has 99% accuracy but is useless. AUC measures how well we separate the two classes regardless of the threshold.

---

## Phase 5: Deep Learning (The "Complex" Approach)

Now we implement the **Convolutional Autoencoder (CAE)**.

### 5.1 Architecture Choice
* **Input:** Log-Mel Spectrogram (treated as a generic 1-channel image).
* **Encoder:** Convolutional Layers (`Conv2d`) $\rightarrow$ Downsampling.
* **Bottleneck:** A dense layer compressing the data (forcing the model to learn the "essence" of a healthy pump).
* **Decoder:** Transposed Convolutional Layers (`ConvTranspose2d`) $\rightarrow$ Upsampling.

> **DS Decision Note: Why Convolutional (CNN) vs. Dense (MLP) layers?**
> A spectrogram has **spatial correlation**. A "scratch" sound appears as a vertical line; a "hum" is a horizontal line. Dense layers ignore this structure. CNNs are designed to detect edges, lines, and textures (visual patterns) in the sound.

### 5.2 The Loss Function
* **Selection:** **Mean Squared Error (MSE)** between Input and Output.
* **Logic:**
    * Training: The model minimizes MSE on *Healthy* data.
    * Inference: We feed a *Broken* sound. The model tries to reconstruct it as if it were healthy. It fails. The MSE spikes.
    * **Anomaly Score = Reconstruction Error.**

### 5.3 Regularization
* **Technique:** **Early Stopping**.
* **Why:** We stop training when the Validation Loss stops decreasing. This prevents "Overfitting," where the model memorizes the training noise instead of learning the general patterns of a pump.

---

## Phase 6: Experimentation & Results

### 6.1 Hyperparameter Tuning
* **Bottleneck Size:** Try compressing to 32, 64, and 128 dimensions.
    * *Hypothesis:* If the bottleneck is too large, the model just copies the input (Identity function) and detects nothing. If too small, it loses vital information.
* **Kernel Size:** Try $3\times3$ vs $5\times5$.
    * *Hypothesis:* $3\times3$ captures fine details (high-frequency clicks). $5\times5$ captures broader trends (background hum).

### 6.2 Error Analysis (The "A+" Section)
* **Visualizing the Failure:** Plot the Input Spectrogram side-by-side with the Reconstructed Spectrogram for a "Clogged" pump.
* **Goal:** Visually show the "Ghosting" effect—where the model smoothed over the crackling sound of the clog because it didn't know how to draw it.

---

## Phase 7: Conclusion & Business Application

* **Deployment Strategy:** Export the model to **ONNX** or **TensorFlow Lite**.
* **Hardware:** Raspberry Pi Zero W + Piezoelectric Sensor (Cost: < KES 5,000).
* **Impact:** Explain how this specific model solves the "Sikiza-Maji" problem by detecting cavitation before the impeller is destroyed.

---

## Technical Stack Summary
1.  **Language:** Python 3.8+
2.  **Audio Processing:** `Librosa`
3.  **Data Manipulation:** `NumPy`, `Pandas`
4.  **Deep Learning:** `PyTorch` (preferred for research/school due to explicit debugging) or `Keras` (for speed).
5.  **Visualization:** `Matplotlib`, `Seaborn`.