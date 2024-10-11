from ultralytics import YOLO
import os
# Load a model
def main():
    model = YOLO("yolov8m.pt")

    dataset = os.path.join('datasets', 'dataset.yaml')
    # Train the model
    train_results = model.train(
        data=dataset,  # path to dataset YAML
        epochs=200,  # number of training epochs
        imgsz=512,  # training image size
        device=0
    )


if __name__ == '__main__':
    main()

