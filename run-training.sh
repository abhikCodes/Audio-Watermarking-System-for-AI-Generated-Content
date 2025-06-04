#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Train the audio steganography model with specified loss function"
    echo ""
    echo "Options:"
    echo "  -d, --data-dir DIR     Directory containing training audio files (required)"
    echo "  -l, --loss TYPE        Loss function type (required)"
    echo "                         Options: spectrogram, log_mel, psychoacoustic"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --data-dir ./training_data --loss spectrogram"
}

# Function to check if directory exists and contains WAV files
check_data_directory() {
    local dir="$1"
    
    # Check if directory exists
    if [ ! -d "$dir" ]; then
        echo "Error: Directory '$dir' does not exist"
        exit 1
    fi
    
    # Check if directory contains WAV files
    local wav_count=$(find "$dir" -type f -name "*.wav" | wc -l)
    if [ "$wav_count" -eq 0 ]; then
        echo "Error: No WAV files found in directory '$dir'"
        echo "Please ensure your training data directory contains .wav files"
        exit 1
    fi
    
    echo "Found $wav_count WAV files in directory '$dir'"
}

# Function to check available disk space (in MB)
check_disk_space() {
    local required_mb=$1
    local available_mb
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        available_mb=$(df -m . | tail -1 | awk '{print $4}')
    else
        # Linux
        available_mb=$(df -m . | tail -1 | awk '{print $4}')
    fi
    
    echo "Available disk space: $available_mb MB"
    
    if [ "$available_mb" -lt "$required_mb" ]; then
        echo "Warning: Low disk space. You have $available_mb MB available, but training requires at least $required_mb MB."
        echo "Cleaning up old model files to free up space..."
        
        # Clean up old checkpoints if they exist
        if [ -d "checkpoints" ]; then
            echo "Removing old checkpoints..."
            rm -rf checkpoints
        fi
        
        # Get available space again
        if [[ "$OSTYPE" == "darwin"* ]]; then
            available_mb=$(df -m . | tail -1 | awk '{print $4}')
        else
            available_mb=$(df -m . | tail -1 | awk '{print $4}')
        fi
        
        echo "Available disk space after cleanup: $available_mb MB"
        
        if [ "$available_mb" -lt "$required_mb" ]; then
            echo "Error: Still not enough disk space after cleanup."
            echo "Please free up at least $required_mb MB of disk space and try again."
            exit 1
        fi
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -l|--loss)
            LOSS_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$DATA_DIR" ] || [ -z "$LOSS_TYPE" ]; then
    echo "Error: Both --data-dir and --loss are required"
    usage
    exit 1
fi

# Validate loss type
if [[ ! "$LOSS_TYPE" =~ ^(spectrogram|log_mel|psychoacoustic)$ ]]; then
    echo "Error: Invalid loss type. Must be one of: spectrogram, log_mel, psychoacoustic"
    exit 1
fi

# Check data directory
check_data_directory "$DATA_DIR"

# Check for disk space (require at least 500MB)
check_disk_space 500

# Create only the minimum necessary directories
mkdir -p "final_models/$LOSS_TYPE/moth" "final_models/$LOSS_TYPE/bat"

# Create .env file with loss function setting
# Use 'printf' instead of echo to avoid newline issues
printf "LOSS_FUNCTION=%s" "$LOSS_TYPE" > .env

# Run the training script
echo "Starting training with loss function: $LOSS_TYPE"
echo "Training data directory: $DATA_DIR"
echo "Final models will be saved in:"
echo "- final_models/$LOSS_TYPE/moth/moth_model.pth"
echo "- final_models/$LOSS_TYPE/bat/bat_model.pth"
echo "----------------------------------------"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python training/train.py --train_dir "$DATA_DIR"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Models saved in: final_models/$LOSS_TYPE/"
else
    echo "Training failed!"
    exit 1
fi 