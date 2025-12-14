Language Classification Simulation (English / Spanish / French)

This repository contains a small AI research experiment on language identification using image-based representations of audio. The core idea is that different languages naturally exhibit distinct intonation patterns and frequency characteristics, and those differences can be captured as visual features. The experiment compares several convolutional and hybrid architectures on Mel Frequency Cepstral Coefficient (MFCC) images and spectrogram images, focusing more on model behavior than on heavy preprocessing or aggressive optimization.

Overview and Motivation

The concept originated from the observation that many audio classification projects target music genres, while fewer focus on language identification. To explore this gap, the experiment evaluates English (En), Spanish (Es), and French (Fr) with simple, transparent steps. The approach intentionally minimizes preprocessing so the results reflect how models learn from relatively raw signal representations rather than from carefully curated inputs.

Data and Inputs

The project expects folder-per-class datasets for both MFCC images and spectrogram images. The default structure used in the notebook places images in MFCC_images/En, MFCC_images/Es, MFCC_images/Fr and spectrogram_images/En, spectrogram_images/Es, spectrogram_images/Fr. Images are loaded via torchvision’s ImageFolder interface and resized to 128×128. Normalization uses a simple mean and standard deviation of 0.5 per channel. The notebook includes a quick visual sanity check to display representative MFCC and spectrogram samples from each language category.

Experiment Setup

The notebook LanguageClassificationSimulation.ipynb initializes environment diagnostics (Python, PyTorch, CUDA availability) and builds DataLoader objects for training, validation, and testing. The dataset is randomly split into training, validation, and test partitions using ratios of 0.70, 0.15, and 0.15 respectively. Batch size is set to 16, and basic parameters like image size and normalization are defined centrally for reproducibility. When CUDA is available, mixed-precision training (AMP) and cuDNN benchmarking are enabled to improve throughput.

Models Evaluated

The experiment compares several architectures: a compact custom convolutional network, a CNN-LSTM hybrid that slices the MFCC image into horizontal bands to model sequential features, and three standard ImageNet-pretrained backbones adapted for three-class output—ResNet18, AlexNet, and VGG16. Each model is trained for a small number of epochs with early stopping based on validation loss. Optimizers and loss functions are kept simple (Adam with a standard cross-entropy objective) to highlight relative model behavior.

Training and Evaluation

Training proceeds in short runs where each epoch reports training loss and accuracy alongside validation metrics. Early stopping prevents overfitting and saves time once the validation loss plateaus. After training concludes, the best validation checkpoint is restored, and a test pass computes final loss and accuracy. Confusion matrices and classification reports are printed when predictions are available, and a summary view ranks models by test accuracy and loss. A separate visualization cell produces bar charts of test accuracy and loss and can render a confusion matrix for the best-performing model.

Key Findings

Across repeated runs, ResNet18 consistently demonstrates stronger learning and greater stability than the other models, topping the comparison on MFCC inputs. The approach is intentionally lightweight, and with more rigorous preprocessing (duration trimming, normalization, augmentation), total accuracy can be pushed toward ninety percent. This project is primarily about comparison, experimentation, and learning rather than chasing a single optimized score.

How to Use

Place your MFCC and spectrogram images in the folder-per-class structure described above. Open the notebook LanguageClassificationSimulation.ipynb, run the diagnostics cell to confirm the environment, then execute the data loading, model definition, and training cells in order. To compare models on spectrogram images instead of MFCC images, switch the data source assignment in the training cell from the MFCC dataset to the spectrogram dataset. The notebook will report per-model test metrics and produce simple visual summaries.

Project Context

This experiment was well received in an academic setting for its creative framing and clarity. It originated during earlier studies and was revisited with a focus on transparency over extensive preprocessing. The idea emphasizes that language identification can benefit from the same visual-feature perspective commonly applied to other audio domains like music classification.

Further Work

Potential improvements include comprehensive audio preprocessing (duration normalization, silence trimming), targeted data augmentation for robustness, and hyperparameter tuning for each architecture. Exploring transformer-based vision backbones or audio-specific architectures could also provide additional insight into representation learning for language identification.

Reference Link

Code and experiment notes are also shared here: https://lnkd.in/gVyBmnAJ
