import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Define dataset path
dataset_path = "NNFL"
output_path = "spectrogram_images"  # Output directory

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Parameters for spectrogram extraction
sr = 22050  # Sampling rate
n_fft = 2048  # FFT window size
hop_length = 512  # Number of samples between successive frames
n_mels = 128  # Number of Mel bands

# Process each language folder
for lang in ["En", "Es", "Fr"]:
    lang_path = os.path.join(dataset_path, lang)
    output_lang_path = os.path.join(output_path, lang)
    os.makedirs(output_lang_path, exist_ok=True)

    # Process each WAV file
    for filename in os.listdir(lang_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(lang_path, filename)

            # Load audio
            y, _ = librosa.load(file_path, sr=sr)

            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Plot spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma')
            plt.axis('off')  # Hide axes

            # Save image
            image_path = os.path.join(output_lang_path, filename.replace(".wav", ".jpg"))
            plt.savefig(image_path, format='jpg', bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Processed and saved: {image_path}")

print("Spectrogram extraction and image saving complete!")
