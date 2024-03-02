import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
# Load the CSV file
csv_path = 'D:\Final_year_project - Copy\dataset\kannada.csv' 
df = pd.read_csv(csv_path)
num_classes = df['class'].nunique()

# Define a function to load and preprocess images
def load_and_preprocess_images(image_paths, img_size=(32, 32)):
    images = []
    for path in image_paths:
        # Load image in grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
        # Threshold to convert the image to black and white
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        # Resize the image to the specified size
        img = cv2.resize(img, img_size)
        img = img / 255.0 # Normalize pixel values
        images.append(img)
    return np.array(images)

# Load and preprocess images
# Update the base path to the 'Img' folder
base_path = 'D:\Final_year_project - Copy\dataset'

# Concatenate base path with relative paths in the 'img' column
df['full_path'] = base_path + df['img']

# Load and preprocess images
image_paths = df['full_path'].values
X = load_and_preprocess_images(image_paths, img_size=(64, 64)) # Increase size to (64, 64) for better resolution

# Convert labels to categorical format
y = to_categorical(df['class'].values - 1) # Subtract 1 to make classes start from 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train.reshape(-1, 64, 64, 1), y_train, epochs=10, validation_data=(X_test.reshape(-1, 64, 64, 1), y_test))
# Save the model
model.save('D:\Final_year_project - Copy\Model\kannada_model', save_format='tf')
