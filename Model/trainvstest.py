import pandas as pd
import numpy as np
import cv2
from tensorflow import keras
from keras.utils import to_categorical
import random

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, img_size=(64, 64), convert_to_grayscale=True):
    images = []
    for path in image_paths:
        # Load image
        img = cv2.imread(path)

        # Check if the image is not empty
        if img is not None:
            # Convert to grayscale if needed
            if convert_to_grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the image to the specified size
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize pixel values
            images.append(img)
        else:
            print(f"Error loading image: {path}")

    return np.array(images)

# Load the CSV file for testing data
testing_data_path = '/content/drive/MyDrive/Colab Notebooks/data/kannada.csv'
df_test = pd.read_csv(testing_data_path)

# Assuming you have base_path_test from your testing data
base_path_test = '/content/drive/MyDrive/Colab Notebooks/data/'
df_test['full_path'] = base_path_test + df_test['img']

# Randomly select 30 images from the testing dataset
num_images_to_test = min(30, len(df_test))
random_test_indices = random.sample(range(len(df_test)), num_images_to_test)
df_test_sample = df_test.iloc[random_test_indices]

# Load and preprocess selected testing data
test_images_paths = df_test_sample['full_path'].values
test_images = load_and_preprocess_images(test_images_paths, convert_to_grayscale=True)

# Convert labels to categorical format
test_labels = to_categorical(df_test_sample['class'].values - 1)

# Load the .h5 model
model_path = '/content/drive/MyDrive/Colab Notebooks/models/kannada_training.h5'
loaded_model = keras.models.load_model(model_path)

# Make predictions on the testing data
predictions = loaded_model.predict(test_images.reshape(-1, 64, 64, 1))

# Get the predicted class index for each example
predicted_classes = np.argmax(predictions, axis=1) + 1  # Adding 1 to convert back to original class values

# Create a DataFrame to store the mapping between testing data, predicted data, and actual labels
result_df = pd.DataFrame({
    'Actual Class': df_test_sample['class'],
    'Predicted Class': predicted_classes
})

# Calculate testing accuracy
testing_accuracy = np.sum(result_df['Actual Class'] == result_df['Predicted Class']) / len(result_df) * 100

# Display the table
print("Mapping of Testing Data and Predicted Data:")
print(result_df.to_string(index=False))

print(f"\nTesting Accuracy: {testing_accuracy:.2f}%")

# Save the output to a CSV file
output_csv_path = '/content/drive/MyDrive/Colab Notebooks/output/predicted_vs_actual_sample.csv'
result_df.to_csv(output_csv_path, index=False)