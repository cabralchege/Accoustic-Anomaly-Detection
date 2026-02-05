import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. APP CONFIGURATION & LOADING
# ==========================================
st.set_page_config(page_title="MIMII Pump Diagnostic", page_icon="🏭", layout="centered")

# Cache the model so it loads only once (faster)
@st.cache_resource
def load_system():
    try:
        # Load the saved model and scaler
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "models", "mimii_lstm_autoencoder.h5")
        SCALER_PATH = os.path.join(BASE_DIR, "models", "mfcc_scaler.pkl")

        model = tf.keras.models.load_model(MODEL_PATH, compile= False)
        scaler = joblib.load(SCALER_PATH)
        # model = tf.keras.models.load_model("models/mimii_lstm_autoencoder.h5")
        # scaler = joblib.load("models/mfcc_scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_system()

# Sidebar for controls
st.sidebar.title("🔧 Settings")
if model is not None:
    st.sidebar.success("System Online: Model Loaded")
else:
    st.sidebar.error("System Offline: Files Missing!")
    st.sidebar.warning("Make sure 'mimii_lstm_autoencoder.h5' and 'mfcc_scaler.pkl' are in the folder.")

# Threshold Control (User can adjust sensitivity)
# Set the default value to the 'best_threshold' you found in Step 5
THRESHOLD = st.sidebar.number_input(
    "Anomaly Threshold", 
    value=0.0450,  # <--- REPLACE THIS with your specific notebook value
    format="%.5f", 
    step=0.001
)

# ==========================================
# 2. UI LAYOUT
# ==========================================
st.title("🏭 Industrial Pump Diagnostic Tool")
st.markdown("### AI-Powered Anomaly Detection")
st.write("Upload a 10-second pump recording (`.wav`) to analyze its mechanical health.")

uploaded_file = st.file_uploader("Drop Audio File Here", type=["wav"])

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================
if uploaded_file is not None:
    # A. Display Audio Player
    st.audio(uploaded_file, format='audio/wav')
    
    # Save temp file for processing
    with open("temp_input.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🔍 Run Diagnostic"):
        with st.spinner("Analyzing acoustic signature..."):
            try:
                # --- PREPROCESSING (Matching Training Steps) ---
                # 1. Load Audio
                y, sr = librosa.load("temp_input.wav", sr=16000, duration=10)
                
                # 2. Extract MFCCs
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
                mfcc = mfcc.T # (Time, Features)
                
                # 3. Alignment (Pad/Crop to 313)
                target_length = 313
                if mfcc.shape[0] < target_length:
                    # Pad with zeros if too short
                    mfcc = np.pad(mfcc, ((0, target_length - mfcc.shape[0]), (0, 0)), mode='constant')
                else:
                    # Crop if too long
                    mfcc = mfcc[:target_length, :]
                
                # 4. Scale (Standardize)
                mfcc_scaled = scaler.transform(mfcc)
                input_data = np.expand_dims(mfcc_scaled, axis=0) # Shape: (1, 313, 20)

                # # ... inside the diagnostic button logic ...
                
                # # Scale
                # mfcc_scaled = scaler.transform(mfcc)
                
                # # === DEBUG BLOCK START ===
                # st.write("### 🔍 Debugging Data Stats")
                # st.write(f"**Min Value:** {np.min(mfcc_scaled):.4f}")
                # st.write(f"**Max Value:** {np.max(mfcc_scaled):.4f}")
                # st.write(f"**Mean Value:** {np.mean(mfcc_scaled):.4f}")
                # # === DEBUG BLOCK END ===

                # input_data = np.expand_dims(mfcc_scaled, axis=0)

                # --- PREDICTION ---
                # 5. Get Reconstruction
                reconstruction = model.predict(input_data, verbose=0)
                
                # 6. Calculate MSE (Anomaly Score)
                mse = np.mean(np.square(input_data - reconstruction))
                
                # --- RESULTS ---
                st.markdown("---")
                
                # Columns for clean layout
                col1, col2, col3 = st.columns(3)
                col1.metric("Anomaly Score", f"{mse:.5f}")
                col2.metric("Threshold Limit", f"{THRESHOLD:.5f}")
                
                # Determine Status
                ratio = mse / THRESHOLD
                if mse > THRESHOLD:
                    status_color = "red"
                    status_text = "CRITICAL FAIL"
                    col3.metric("Risk Factor", f"{ratio:.1f}x", delta_color="inverse")
                    st.error(f"🚨 **ABNORMAL DETECTED**")
                    st.write("The machine sound deviates significantly from the normal baseline.")
                else:
                    status_color = "green"
                    status_text = "NORMAL PASS"
                    col3.metric("Risk Factor", f"{ratio:.1f}x")
                    st.success(f"✅ **NORMAL OPERATION**")
                    st.write("Machine is operating within healthy parameters.")

                # --- VISUALIZATION ---
                # Plot the error over time to show WHERE the glitch happened
                st.markdown("#### 📈 Real-time Error Analysis")
                error_over_time = np.mean(np.square(input_data[0] - reconstruction[0]), axis=1)
                
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(error_over_time, color='#333333', alpha=0.7, label='Reconstruction Error')
                ax.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, label='Threshold Limit')
                
                # Fill area under curve if error > threshold
                ax.fill_between(range(len(error_over_time)), error_over_time, THRESHOLD, 
                                where=(error_over_time > THRESHOLD), 
                                color='red', alpha=0.3, interpolate=True)
                
                ax.set_ylabel("MSE Loss")
                ax.set_xlabel("Time (Frames)")
                ax.legend()
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists("temp_input.wav"):
                    os.remove("temp_input.wav")