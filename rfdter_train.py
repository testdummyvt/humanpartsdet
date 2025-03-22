from rfdetr import RFDETRBase

import torch
torch.cuda.empty_cache()

if __name__ == "__main__":

    dataset = "/home/ubuntu/data/cocoformat"
    output_dir = "/home/ubuntu/checkpoints/chpyolo11x_train"
    model = RFDETRBase()

    history = []
    # From roboflow colab notebook
    def callback2(data):
        history.append(data)
    
    model.callbacks["on_fit_epoch_end"].append(callback2)
    model.train(dataset_dir=dataset, output_dir = output_dir, epochs=10, batch_size=10, grad_accum_steps=4, lr=1e-4)

    import pandas as pd
    df = pd.DataFrame(history)

    df.to_csv(f"{output_dir}/history.csv")