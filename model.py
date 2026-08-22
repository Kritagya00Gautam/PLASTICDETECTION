from ultralytics import YOLO
model = YOLO("runs/plastic_detector/weights/best.pt")  
results = model.predict(source="0",conf=0.25, show=True)