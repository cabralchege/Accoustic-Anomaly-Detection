# Acoustic Anomaly Detection for Railway Infrastructure
Predicting rolling stock and track failures before they happen using Deep Learning and Audio Signal Processing.

## Executive Summary
Railway infrastructure in rugged terrain (like the SGR passing through Tsavo) faces constant stress from heat cycles and load. Traditional visual inspections are manual, slow, and reactive.

SGR-Guard is an automated predictive maintenance system. It uses Deep Learning (Convolutional Autoencoders) to "listen" to the acoustic signature of wheel-rail interactions. By learning the baseline audio of a "healthy" track, the system can flag unseen anomalies (cracks, wheel flats, or corrugation) by detecting high reconstruction errors in audio spectrograms.

## The Problem
Safety Risk: Microscopic rail cracks (Rolling Contact Fatigue) are invisible to the naked eye until catastrophic failure.

Operational Cost: Unscheduled downtime for SGR cargo trains costs millions in logistics delays.

Data Scarcity: We have thousands of hours of "normal" train sounds, but very few recordings of "crashes/failures," making standard classification models impossible to train.

## The Solution: Unsupervised Anomaly Detection
Instead of a classifier (Binary: Broken vs. Healthy), this project uses a Reconstruction-based approach:

Input: Raw audio logs converted to Mel-Spectrograms (visual representations of sound frequencies).

Model: A Convolutional Autoencoder is trained only on healthy data. It learns to compress and perfectly reconstruct the "sound of a smooth ride."

Inference: When the model encounters a "broken" sound (e.g., a clicking rail), it fails to reconstruct the anomaly.

Trigger: High Reconstruction Error (MSE) flags the segment as critical.

## Architecture Pipeline


Deep Learning: PyTorch (Convolutional Autoencoder)

Audio Processing: Librosa (FFT, Mel-Spectrograms)

Data handling: NumPy, Pandas

Visualization: Matplotlib, Seaborn

Deployment (UI): Streamlit (for the dashboard demo)

## Dataset Strategy (Proxy Data)
Since proprietary SGR acoustic data is sensitive, this project utilizes high-fidelity industrial proxy datasets that mimic the physics of rotating machinery and friction:

NASA Bearing Dataset (CWRU): Standard benchmark for bearing faults (simulating wheelset issues).

MIMII Dataset: Industrial machine sounds (fans/pumps) for background noise robustness.

## Scalability & Industry Application
This architecture is hardware-agnostic and scales to other high-value Kenyan sectors:

Energy (Wind Power):

Application: Monitoring gearbox health in remote turbines (e.g., Lake Turkana Wind Power).

Benefit: Reducing the need for dangerous manual climbs for inspection.

Manufacturing:

Application: Conveyor belt motor diagnostics in bottling plants.

Benefit: Predicting motor burnout to prevent production line stoppages.

Aviation:

Application: analyzing jet engine start-up sounds for irregularities.
