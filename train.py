import argparse
from ultralytics import YOLO

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Trains a YOLO model on a specified dataset.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the YOLO model file to be trained. Example: '/mnt/c/local/models/yolo11x.pt'"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The path to the dataset YAML configuration file. Example: '/mnt/c/local/datasets/cocohumanparts/data.yaml'"
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="The project directory where logs and results will be saved. Example: '/mnt/c/local/logs/cocohumanparts/yolo11x'"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of training epochs. Default: 100"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size. Default: 640"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to run training on. Options: 'cpu', 'cuda'. Default: 'cuda'"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of workers for data loading. Default: 24"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=72,
        help="Batch size for training. Default: 72"
    )

    parser.add_argument(
        "--plots",
        type=bool,
        default=True,
        nargs='?',
        const=True,
        help="Plot training loss and mAP metrics. Default: True"
    )

    return parser.parse_args()

def main(model_path, dataset_yaml, project_dir, epochs, imgsz, device, workers, batch, plots):
    """
    Main function to train a YOLO model.

    Args:
        model_path (str): The path to the YOLO model file to be trained.
        dataset_yaml (str): The path to the dataset YAML configuration file.
        project_dir (str): The project directory where logs and results will be saved.
        epochs (int): The number of training epochs.
        imgsz (int): Input image size.
        device (str): The device to run training on. Options: 'cpu', 'cuda'.
        workers (int): Number of workers for data loading.
        batch (int): Batch size for training.
        plots (bool): Plot training loss and mAP metrics.
    """
    # Load the YOLO model from the specified path.
    model = YOLO(model_path)
    
    # Train the YOLO model on the provided dataset.
    results = model.train(
        data=dataset_yaml,  # Path to the dataset YAML configuration file.
        epochs=epochs,        # Number of training epochs.
        imgsz=imgsz,          # Input image size.
        device=device,        # Device to run training on ('cpu' or 'cuda').
        project=project_dir,  # Project directory to save logs and results.
        workers=workers,      # Number of workers for data loading.
        batch=batch,          # Batch size for training.
        plots=plots           # Plot training loss and mAP metrics.
    )

    return results

if __name__ == '__main__':
    # Parse the command-line arguments.
    args = parse_arguments()
    
    # Train the model using the provided arguments.
    main(
        model_path=args.model,
        dataset_yaml=args.dataset,
        project_dir=args.project,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        batch=args.batch,
        plots=args.plots
    )