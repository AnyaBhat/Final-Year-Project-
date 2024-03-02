import cv2
import pytesseract
import matplotlib.pyplot as plt

# Path to Tesseract executable (Update this path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Function to perform OCR and segment Kannada letters
def segment_kannada_letters(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour and extract the letter
    segmented_letters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letter = img[y:y + h, x:x + w]
        segmented_letters.append(letter)

    return segmented_letters

# Example usage
image_path = 'path/to/your/kannada_sentence_image.png'
segmented_letters = segment_kannada_letters(image_path)

# Display segmented letters
plt.figure(figsize=(10, 5))
for i, letter in enumerate(segmented_letters):
    plt.subplot(1, len(segmented_letters), i + 1)
    plt.imshow(cv2.cvtColor(letter, cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.show()
