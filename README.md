# Thesis Code Repository
forked from https://github.com/facebookresearch/bounding_data_reconstruction

## Setup
* conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
* conda install conda-forge::hydra-core
* pip install dp-accounting
* conda install anaconda::scikit-learn
* conda install conda-forge::matplotlib
* pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
* pip install lpips

## Usage
python train_classifier.py --config-name=e2.yaml