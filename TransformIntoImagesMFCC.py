import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for saving images in .jpg format

# Define dataset path
dataset_path = "NNFL"
output_path = "mfcc_images"  # Output directory

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Parameters for MFCC extraction
sr = 22050  # Sampling rate
n_mfcc = 13  # Number of MFCC coefficients

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
            y, sr = librosa.load(file_path, sr=sr)

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # Plot MFCCs and save as an image
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, x_axis='time', cmap='coolwarm')

            # Save the plot as a .jpg image
            image_path = os.path.join(output_lang_path, filename.replace(".wav", ".jpg"))
            plt.axis('off')  # Turn off the axis for the image
            plt.savefig(image_path, format='jpg', bbox_inches='tight', pad_inches=0)
            plt.close()  # Close the figure to free memory

            print(f"Processed and saved: {image_path}")

print("MFCC extraction and image saving complete!")
