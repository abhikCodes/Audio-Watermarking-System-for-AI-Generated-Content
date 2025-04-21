## Commands

1. Download dataset from https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset/data
2. Create your virtual environment ```python -m venv venv```
3. Activate your virtual environment ```source venv/bin/activate```
4. Install packages ```pip install -r requirements.txt```
5. Train Models ```python training/train.py --train_dir <DATA DIR>```
6. Test Audio ```python `training/train.py --test_dir <TEST DATA DIR>```: The watermarked audio will be saved in the output folder.

To test change parameters in config.py in configurations folder