import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load model
MODEL_PATH = "Car_Color_Detection.keras"  # Update with your model's path
model = load_model(MODEL_PATH)

# Color mapping
COLOR_LABELS = [
    'beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 
    'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow'
]

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = Image.open(image_path).resize((128, 128))  # Adjust size to (128, 128)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_color():
    """Predict the car color from the selected image."""
    global selected_file
    if not selected_file:
        result_label.config(text="Please select an image first.")
        return
    preprocessed_image = preprocess_image(selected_file)
    prediction = model.predict(preprocessed_image)
    color_index = np.argmax(prediction)
    result_label.config(text=f"Predicted Color: {COLOR_LABELS[color_index]}")

def open_file():
    """Open file dialog to select an image."""
    global selected_file
    selected_file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if selected_file:
        # Display the image
        img = Image.open(selected_file)
        img.thumbnail((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="")

# Initialize GUI
selected_file = None
root = tk.Tk()
root.title("Car Color Detection")

# Widgets
Label(root, text="Car Color Detection", font=("Arial", 20)).pack(pady=10)
Button(root, text="Select Image", command=open_file, font=("Arial", 14)).pack(pady=10)
Button(root, text="Detect", command=predict_color, font=("Arial", 14)).pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

# Run the application
root.mainloop()
