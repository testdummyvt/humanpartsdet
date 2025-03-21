from rfdetr import RFDETRBase


if __name__ == "__main__":

    dataset = "/home/ubuntu/data/cocohuman"
    model = RFDETRBase()

    history = []
    # From roboflow colab notebook
    def callback2(data):
        history.append(data)
    
    model.callbacks["on_fit_epoch_end"].append(callback2)
    model.train(dataset_dir=dataset, epochs=10, batch_size=16, grad_accum_steps=4, lr=1e-4)