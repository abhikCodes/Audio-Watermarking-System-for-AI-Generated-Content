
## Prerequisites

*   Docker and Docker Compose must be installed on your system.

## Getting Started

### Running the Application

1.  Clone the repository:
    ```bash
    git clone <your-repository-link>
    cd <repository-name>
    ```
2.  Ensure Docker is running.
3.  Execute the following command from the root of the project directory:
    ```bash
    docker compose up
    ```
4.  This will build the Docker images for the frontend and backend and start the services.
    *   The frontend will typically be accessible at `http://localhost:3000`.
    *   The backend API will typically be accessible at `http://localhost:8000`.
    *(The first time you run this, it might take a while to download dependencies and build the images.)*

## How to Use

Once the application is running:

### Moth - Watermarking Audio

1.  Navigate to `http://localhost:3000` in your web browser.
2.  The default view should be the "Moth Encoder."
3.  Drag and drop a `.wav` audio file into the designated area or click to browse.
4.  Click the "**Apply AI Watermark**" button.
5.  The system will process the audio and apply the watermark.
6.  You can then:
    *   Listen to the original and watermarked audio using the embedded players.
    *   Download the original audio.
    *   Download the watermarked audio.

### Bat - Detecting Watermark

1.  From the application interface (top right), switch to the "**Bat Detector**" view.
2.  Drag and drop an audio file (preferably one you watermarked using the Moth encoder, or an original file for comparison) into the designated area.
3.  Click the "**Analyze Audio**" button.
4.  The Bat detector will analyze the audio and display the results:
    *   **AI-Generated Audio Detected!** (if watermark is found) with a confidence score.
    *   **Natural Audio Detected** (if no watermark is found) with a confidence score.
5.  You can listen to the uploaded audio and download it.

## Dataset

The primary dataset used for training the models is the **Speaker Recognition Audio Dataset** available on Kaggle.
*   **Link:** [https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset)

Additional synthetic AI-generated audio (e.g., from ElevenLabs or open-source models) and publicly available audio datasets (e.g., LibriSpeech, Free Music Archive) were considered for robustness testing.

## Pre-trained Model

The pre-trained models (Moth encoder and Bat decoder) used by the application are available for download.
*   **Model Drive Link:** [https://drive.google.com/drive/folders/14u7cjH1iIUYMrRxNgO4-4FBQ-zzCxrkA?usp=drive_link](https://drive.google.com/drive/folders/14u7cjH1iIUYMrRxNgO4-4FBQ-zzCxrkA?usp=drive_link)

These models are loaded by the backend service when the Docker container starts.

## Training the Model

Detailed information regarding the training process, including the specific architectures, data preprocessing, and training loops for both the Moth (encoder) and Bat (decoder) models, can be found in:
`speech_audio/models/training/train.py`

The training script allows for experimentation with different loss functions and hyperparameters. Logs from various training runs are available in the `speech_audio/logs/` directory.

## Results

The effectiveness of different loss functions was evaluated based on the Perceptual Evaluation of Speech Quality (PESQ) scores. Higher PESQ scores (closer to 4.5-4.64 for original vs. original) indicate better audio quality/less perceptible distortion.

| Loss Function             | Audio Type             | PESQ Score |
| :------------------------ | :--------------------- | :--------- |
| N/A                       | Original vs Original   | 4.64       |
| MSE                       | Original vs Watermarked | 2.28       |
| Spectrogram               | Original vs Watermarked | 3.69       |
| **Mel-Spectrogram Based** | **Original vs Watermarked** | **4.25**   |
| Psychoacoustic Loss       | Original vs Watermarked | 4.37       |

Through experiments, the **Log-Mel-Spectrogram Loss** was found to provide a good balance, resulting in an embedded watermark that is nearly imperceptible to the human ear while remaining detectable.

## Authors

*   **Yash Singh Pathania** ([yash.pathania@ucdconnect.ie](mailto:yash.pathania@ucdconnect.ie))
*   **Abhik Sarkar** ([abhik.sarkar@ucdconnect.ie](mailto:abhik.sarkar@ucdconnect.ie))

Project for [Module Name/Course - e.g., Speech & Audio Processing], University College Dublin.
Date: March 31, 2025 

## Future Directions

*   **Enhanced Robustness:** Further improve watermark resilience against more complex audio transformations and adversarial attacks.
*   **Conditional Watermark Removal:** Implement the proposed secure mechanism for authorized watermark removal.
*   **Copyright Protection:** Extend the technology for copyright management.
*   **Platform Integration:** Explore integration with content platforms (e.g., YouTube) for automated labeling.
*   **Cross-Media Applications:** Adapt the principles for video and image watermarking.
*   **Interoperability Standards:** Work towards standardized watermark detection across different AI platforms.

---

Happy reading, and have a sunny day ahead!
