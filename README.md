# humanpartsdet

## Dataset Download

You can download the COCO Human Parts dataset using the `download_dataset.py` script. This script downloads the dataset and updates the `data.yaml` file with the correct local paths.

### Usage

To download the dataset, run the following command:

```bash
python tools/download_dataset.py --local_dataset_dir <path_to_your_dataset_directory>
```

Replace `<path_to_your_dataset_directory>` with the path where you want to store the dataset.


### Training

The `train.py` script is used to train a YOLO model on the COCO Human Parts dataset or any custom dataset specified in a YAML file. It leverages the `ultralytics` library for training.

#### Usage

To train the model, execute the following command:

```bash
python train.py --model <path_to_model.pt> --dataset <path_to_data.yaml> --project <path_to_project_dir> --epochs <num_epochs> --imgsz <image_size> --device <device> --workers <num_workers> --batch <batch_size> --plots --resume
```

**Arguments:**

*   `--model`: (Required) Path to the YOLO model file (e.g., `yolov11n.pt`).
*   `--dataset`: (Required) Path to the dataset YAML file (e.g., `data.yaml`).  This file should define the path to the training and validation data, the number of classes, and the class names. See [YOLOv11 documentation](https://docs.ultralytics.com/datasets/detect/) for the expected format.
*   `--project`: (Required) Directory to save the training results (logs, weights, etc.).
*   `--epochs`: (Optional, default: 100) Number of training epochs.
*   `--imgsz`: (Optional, default: 640) Input image size.
*   `--device`: (Optional, default: `cuda`) Device to use for training (`cpu` or `cuda`).
*   `--workers`: (Optional, default: 8) Number of workers for data loading.
*   `--batch`: (Optional, default: 16) Batch size.
*   `--plots`: (Optional) Enable or disable plotting of training metrics. Use `--plots` to enable and omit to disable.
*   `--resume`: (Optional) Resume training from the last checkpoint if available. Use `--resume` to enable and omit to disable.

**Example:**

```bash
python train.py --model yolov11n.pt --dataset data.yaml --project runs/humanparts --epochs 50 --imgsz 640 --device cuda --workers 4 --batch 32 --plots
```

This command trains the `yolov8n.pt` model using the dataset defined in `data.yaml`, saves the results in the `runs/humanparts` directory, trains for 50 epochs, uses an image size of 640, utilizes the CUDA device, sets the number of data loading workers to 4, uses a batch size of 32, and enables plots.