from ultralytics import YOLO
import numpy as np

# Load the trained model
model = YOLO('../runs/classify/train/weights/last.pt')

# Make prediction
results = model('yolov8_custom_training/train/normal/IM-0128-0001.jpeg')

# Get class names
names_dict = results[0].names
print("Class names:", names_dict)


probs_data = results[0].probs.data.tolist()
print("Probabilities (using .data):", probs_data)

# Option 2: Using .numpy() method
probs_numpy = results[0].probs.numpy()
print("Probabilities (using .numpy()):", probs_numpy)

# Get the predicted class
predicted_class_idx = results[0].probs.top1
predicted_class_name = names_dict[predicted_class_idx]
predicted_confidence = results[0].probs.top1conf

print(f"Predicted class: {predicted_class_name}")
print(f"Confidence: {predicted_confidence:.4f}")

# Alternative way using numpy argmax
predicted_class_idx_alt = np.argmax(probs_data)
predicted_class_name_alt = names_dict[predicted_class_idx_alt]
print(f"Predicted class (using argmax): {predicted_class_name_alt}") 