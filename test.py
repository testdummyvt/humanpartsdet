import yaml

if __name__ == "__main__":
    # Load the YAML file
    with open('/home/ubuntu/checkpoints/yolo11x/chpyolo11x_train/train/args.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Modify the variable
    data['model'] = '/home/ubuntu/checkpoints/yolo11x/chpyolo11x_train/train/weights/last.pt'
    data['data'] = '/home/ubuntu/data/cocohuman/data.yaml'
    data['project'] = "/home/ubuntu/checkpoints/yolo11x/chpyolo11x_train"
    data['save_dir'] = "/home/ubuntu/checkpoints/yolo11x/chpyolo11x_train/train"


    # Save the updated YAML file
    with open('/home/ubuntu/checkpoints/yolo11x/chpyolo11x_train/train/args.yaml', 'w') as file:
        yaml.safe_dump(data, file)

    print("Variable updated successfully!")
