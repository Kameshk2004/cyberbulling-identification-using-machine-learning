import cv2
import pytesseract
import numpy as np

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Tesseract-OCR\\tesseract.exe"

def extract_text_from_image(image_path):
    
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found or cannot be opened.")
        exit()

    # Resize image 
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Apply binary threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    
    return text.strip()

    # Show the processed image (for debugging)
    cv2.imshow("Thresholded Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
