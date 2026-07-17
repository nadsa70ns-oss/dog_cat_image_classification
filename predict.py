from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tf_keras as tk

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tk.models.load_model("/keras_model.h5", compile=False)

# Load the labels
class_names = open("/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load the image
image = Image.open("/image.png").convert("RGB")

# Resize the image
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Convert image to numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
