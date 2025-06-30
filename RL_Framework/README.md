# Donkey Car Reinforcement Learning Setup & Training Guide

### Method 1: Clone with submodules (Recommended)
```bash
git clone --recursive https://github.com/your-repo/donkey-rl-project.git
cd donkey-rl-project
```

### Method 3: Complete manual setup (if above fails)
```bash
# Remove empty directories first
rm -rf rl-baselines3-zoo gym-donkeycar-1 aae-train-donkeycar

# Clone each component manually
git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
git clone https://github.com/araffin/gym-donkeycar-1.git  
git clone https://github.com/araffin/aae-train-donkeycar.git

# Switch to correct branches
cd gym-donkeycar-1 && git checkout feat/race_june && cd ..
cd aae-train-donkeycar && git checkout feat/race_june && cd ..
cd rl-baselines3-zoo && git checkout v1.8.0 && cd ..
```

## 📋 **Prerequisites**

### Required Software
- **Anaconda/Miniconda**
- **Git**
- **Python 3.8.x** (included in conda environment)

### Conda Environment
You need the exact environment file (`donkey_rl_env.yml`) with these key packages:
- **Python**: 3.8.20
- **stable-baselines3**: 1.8.0 
- **sb3-contrib**: 1.8.0
- **gym**: 0.22.0
- **torch**: 2.4.1

## 🚀 **Step-by-Step Installation**

### Step 1: Create Conda Environment
```bash
# Create environment from yml file
conda env create -f donkey_rl_env.yml
conda activate donkey_rl

# Verify core packages
python -c "import gym; print('Gym:', gym.__version__)"
python -c "import stable_baselines3; print('SB3:', stable_baselines3.__version__)"
```

### Step 2: Install RL-Baselines3-Zoo
```bash
cd rl-baselines3-zoo

# Check if you're on the right version (should be v1.8.0 for SB3 1.8.0 compatibility)
git branch
git tag | grep v1.8

# Install without dependency conflicts (you already have correct versions)
pip install -e . --no-deps

# Verify installation
python -c "import rl_zoo3; print('✓ RL Zoo installed')"
```

### Step 3: Install Gym-DonkeyCar Environment  
```bash
cd ../gym-donkeycar-1

# Verify branch (should be feat/race_june for TQC model compatibility)
git branch

# Install the environment
pip install -e .

# Test environment registration
python -c "import gym; import gym_donkeycar; env = gym.make('donkey-mountain-track-v0'); print('✓ Environment created'); env.close()"
```

### Step 4: Install AAE Training Utilities
```bash
cd ../aae-train-donkeycar

# Verify branch
git branch

# Install autoencoder utilities
pip install -e .

# Test import
python -c "import ae.wrapper; print('✓ Autoencoder wrapper available')"
```

### Step 5: Download Pre-trained Autoencoder
```bash
cd ../rl-baselines3-zoo

# Create logs directory
mkdir -p logs

# Download the pre-trained autoencoder (required for TQC model)
cd logs
curl -L -O https://github.com/araffin/aae-train-donkeycar/releases/download/live-twitch-2/ae-32_mountain.pkl
cd ..

# Set environment variable (Windows)
set AE_PATH=%cd%\logs\ae-32_mountain.pkl

# Set environment variable (Linux/Mac)  
export AE_PATH="$(pwd)/logs/ae-32_mountain.pkl"

# Verify file exists
echo %AE_PATH%  # Windows
echo $AE_PATH   # Linux/Mac
```

### Step 6: Verify Complete Installation
```bash
cd rl-baselines3-zoo

# Test all components work together
python -c "
import gym
import gym_donkeycar
import stable_baselines3
import rl_zoo3
import ae.wrapper
print('✓ All imports successful')

# Test environment creation
env = gym.make('donkey-mountain-track-v0')
print('✓ Donkey environment created')
env.close()
"
```

## 🎯 **Training & Running Models**

### Download Pre-trained Model (Recommended for Testing)
```bash
cd rl-baselines3-zoo

# Download reference TQC model
python -m rl_zoo3.load_from_hub --algo tqc --env donkey-mountain-track-v0 -orga araffin -f logs/

# Run the pre-trained model
python enjoy.py --algo tqc --env donkey-mountain-track-v0 -f logs/tqc/donkey-mountain-track-v0_1/ --load-best
```

### Train Your Own Model
```bash
cd rl-baselines3-zoo

# Quick training test (1000 steps)
python train.py --algo tqc --env donkey-mountain-track-v0 -f logs/ --seed 42 -n 1000

# Full training (2M steps - takes several hours)
python train.py --algo tqc --env donkey-mountain-track-v0 -f logs/ --seed 42 -n 2000000

# Monitor training progress
tensorboard --logdir logs/
```

### Run Your Trained Model
```bash
# Run the best model from training
python enjoy.py --algo tqc --env donkey-mountain-track-v0 -f logs/tqc/ --load-best

# Run a specific model checkpoint
python enjoy.py --algo tqc --env donkey-mountain-track-v0 --load-checkpoint logs/tqc/donkey-mountain-track-v0_1/best_model.zip
```

## 📁 **Project Structure**
```
donkey-rl-project/
├── donkey_rl_env.yml           # Conda environment file
├── rl-baselines3-zoo/          # Main training framework
│   ├── logs/                   # Training logs and models
│   │   ├── ae-32_mountain.pkl  # Pre-trained autoencoder
│   │   └── tqc/                # TQC algorithm results
│   ├── train.py                # Training script
│   ├── enjoy.py                # Model evaluation script
│   └── hyperparams/            # Algorithm hyperparameters
├── gym-donkeycar-1/            # Donkey Car gym environment
│   └── gym_donkeycar/          # Environment code
├── aae-train-donkeycar/        # Autoencoder training utilities
│   └── ae/                     # Autoencoder code
└── README.md                   # This file
```

## 🎮 **Usage Examples**

### Quick Test Run
```bash
# 1. Activate environment
conda activate donkey_rl

# 2. Download and run pre-trained model
cd rl-baselines3-zoo
python -m rl_zoo3.load_from_hub --algo tqc --env donkey-mountain-track-v0 -orga araffin -f logs/
python enjoy.py --algo tqc --env donkey-mountain-track-v0 -f logs/tqc/donkey-mountain-track-v0_1/ --load-best
```

### Train New Model
```bash
# 1. Start training
python train.py --algo tqc --env donkey-mountain-track-v0 -f logs/ -n 500000

# 2. Monitor progress
tensorboard --logdir logs/

# 3. Test trained model
python enjoy.py --algo tqc --env donkey-mountain-track-v0 -f logs/tqc/ --load-best
```

### Hyperparameter Tuning
```bash
# Optimize hyperparameters using Optuna
python train.py --algo tqc --env donkey-mountain-track-v0 -f logs/ -optimize --n-trials 100 --n-jobs 4
```



## 🔗 **References**

- **Hugging Face Model**: https://huggingface.co/araffin/tqc-donkey-mountain-track-v0
- **RL Baselines3 Zoo**: https://github.com/DLR-RM/rl-baselines3-zoo
- **Gym DonkeyCar**: https://github.com/tawnkramer/gym-donkeycar
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/

