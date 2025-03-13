import json
import os
import math
import librosa

DATASET_PATH = "Languages"
JSON_PATH = "Languages.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 5  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a JSON file along with genre labels.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to JSON file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :param num_segments (int): Number of segments we want to divide sample tracks into
    """

    # Dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": ["English", "French"],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure we're processing a genre sub-folder level
        if dirpath != dataset_path:

            # Save genre label (i.e., sub-folder name) in the mapping
            semantic_label = os.path.basename(dirpath)  # Use os.path.basename for cross-platform compatibility
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # Process all audio files in the genre sub-directory
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                try:
                    # Load audio file
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Process all segments of the audio file
                    for d in range(num_segments):

                        # Calculate start and finish sample for the current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # Extract MFCC
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate,
                                                    n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T  # Transpose for correct format

                        # Store only MFCC feature with the expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)  # Labels should match genre index
                            print("{}, segment:{}".format(file_path, d + 1))

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Save MFCCs to a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
