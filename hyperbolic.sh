wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
sudo apt-get update -y && sudo apt-get install libgl1 -y
sudo apt-get install cmake -y

uv venv
source .venv/bin/activate
uv sync

sudo apt install git-lfs -y
sudo apt install htop -y
sudo apt install tmux -y

git config --global user.email "testdummyvt@gmail.com"
git config --global user.name "testdummyvt"


# python tools/yoloformat_download_dataset.py --local_dataset_dir ~/data/cocohuman
# python yolo11_train.py --model yolo11x --dataset /home/ubuntu/data/cocohuman/data.yaml --project /home/ubuntu/checkpoints/chpyolo11x_train --workers 48 --batch 192 --device '0,1,2,3,4,5,6,7'