wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
sudo apt-get update -y && sudo apt-get install libgl1 -y
sudo apt-get install cmake -y

uv venv
source .venv/bin/activate
uv sync


# python tools/yoloformat_download_dataset.py --local_dataset_dir ~/data/cocohuman
# python yolo11_train.py --model yolo11x --dataset /home/ubuntu/data/cocohuman/data.yaml --project /home/ubuntu/exp/yolo11x --workers 16 --batch 16