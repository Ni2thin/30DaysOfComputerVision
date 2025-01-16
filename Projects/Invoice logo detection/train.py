from ultralytics import YOLO

#download the model. this will train yolo from scratch.
model = YOLO('yolov8.yaml')

#use the model
results = model.train(data='./yolo/config.yaml', epochs = 100)


