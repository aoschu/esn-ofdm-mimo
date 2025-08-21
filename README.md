# ESN_with_OFDM_MIMO

**Echo State Networks (ESN) for symbol detection in OFDM/MIMO systems.**

## System Model 1 (Sionna + TensorFlow) — install & run
# Create env and install (recommended)
conda create -n esn_sm1 python=3.10 -y
conda activate esn_sm1
pip install -r requirements-sm1.txt

# If only CPU available:
pip install tensorflow-cpu==2.11.0
pip install sionna==0.19.2

# Quick check:
python -c "import sionna; import tensorflow as tf; print('sionna', sionna.__version__, 'tf', tf.__version__)"

# Run notebook:
jupyter lab system_model_1/system_model_01.ipynb

## System Model 2 (PyTorch / baselines) — install & run
conda create -n esn_sm2 python=3.10 -y
conda activate esn_sm2
pip install -r requirements-sm2.txt

# Quick check:
python -c "import torch; print('torch', torch.__version__)"

# Example run:
python system_model_2/Demo_MIMO_2x2_all_DL_model_comparion.py
