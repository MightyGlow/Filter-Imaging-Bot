import os
import cv2
import numpy as np
import random

# Take user input for source and destination directories
SOURCE_DIR = input("Enter the source directory path: ").strip()
OUTPUT_DIR = input("Enter the destination directory path: ").strip()

# Check if the source directory exists
if not os.path.exists(SOURCE_DIR):
    print(f"Error: The source directory '{SOURCE_DIR}' does not exist.")
    exit()

# Ensure the destination directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define filters
def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def invert(image):
    return cv2.bitwise_not(image)

# def grayscale(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# def rotate_90(image):
#     return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# def flip_horizontal(image):
#     return cv2.flip(image, 1)  # 1 for horizontal flip

# def flip_vertical(image):
#     return cv2.flip(image, 0)  # 0 for vertical flip

# def brightness_contrast(image):
#     # Increase brightness by 30 and contrast by 20%
#     alpha = 1.2  # Contrast control
#     beta = 30    # Brightness control
#     return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# def random_noise(image):
#     # Add random noise (salt and pepper type)
#     output = image.copy()
#     prob = 0.05  # Probability of noise
#     if len(output.shape) == 2:
#         black = 0
#         white = 255            
#     else:
#         colorspace = output.shape[2]
#         if colorspace == 3:  # RGB
#             black = np.array([0, 0, 0], dtype='uint8')
#             white = np.array([255, 255, 255], dtype='uint8')
            
#     probs = np.random.random(output.shape[:2])
#     output[probs < (prob / 2)] = black
#     output[probs > 1 - (prob / 2)] = white
#     return output

# def color_jitter(image):
#     # Randomly adjust hue, saturation, and value
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
    
#     # Adjust hue
#     h = h.astype(np.float32)
#     h += random.randint(-20, 20)
#     h = np.clip(h, 0, 179).astype(np.uint8)
    
#     # Adjust saturation
#     s = s.astype(np.float32)
#     s *= random.uniform(0.7, 1.3)
#     s = np.clip(s, 0, 255).astype(np.uint8)
    
#     # Adjust value/brightness
#     v = v.astype(np.float32)
#     v *= random.uniform(0.7, 1.3)
#     v = np.clip(v, 0, 255).astype(np.uint8)
    
#     hsv = cv2.merge([h, s, v])
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def edge_detection(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
#     edges = cv2.Canny(gray, 100, 200)
#     return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels

# Apply filters to images
def apply_filters(image, base_name):
    # Track which filters have been applied
    applied_filters = set()
    
    filter_names = ['sharpen', 'invert']
    filters = [sharpen, invert]


    for name, filter_func in zip(filter_names, filters):
        # Skip if we've already applied this filter type
        if name in applied_filters:
            continue
            
        filtered_img = filter_func(image)
        # Save directly to the output directory with a descriptive filename
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}.png")
        cv2.imwrite(output_path, filtered_img)
        
        # Mark this filter as applied
        applied_filters.add(name)

# Process images in the source directory
def process_images(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is not None:
                base_name, _ = os.path.splitext(file_name)
                print(f"Processing: {file_name}")
                apply_filters(image, base_name)
            else:
                print(f"Could not read {file_name}. Skipping...")

# Main program
if __name__ == "__main__":
    process_images(SOURCE_DIR)
    print("Processing complete. Filtered images saved.")




# import os
# import cv2
# import numpy as np

# # Take user input for source and destination directories
# SOURCE_DIR = input("Enter the source directory path: ").strip()
# OUTPUT_DIR = input("Enter the destination directory path: ").strip()

# # Check if the source directory exists
# if not os.path.exists(SOURCE_DIR):
#     print(f"Error: The source directory '{SOURCE_DIR}' does not exist.")
#     exit()

# # Ensure the destination directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Define filters
# def sharpen(image):
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# def invert(image):
#     return cv2.bitwise_not(image)

# # Apply filters to images
# def apply_filters(image, base_name):
#     filter_names = ['sharpen', 'invert']
#     filters = [sharpen, invert]

#     for name, filter_func in zip(filter_names, filters):
#         filtered_img = filter_func(image)
#         # Save directly to the output directory with a descriptive filename
#         output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}.png")
#         cv2.imwrite(output_path, filtered_img)

# # Process images in the source directory
# def process_images(folder_path):
#     for file_name in os.listdir(folder_path):
#         if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#             file_path = os.path.join(folder_path, file_name)
#             image = cv2.imread(file_path)
#             if image is not None:
#                 base_name, _ = os.path.splitext(file_name)
#                 print(f"Processing: {file_name}")
#                 apply_filters(image, base_name)
#             else:
#                 print(f"Could not read {file_name}. Skipping...")

# # Main program
# if __name__ == "__main__":
#     process_images(SOURCE_DIR)
#     print("Processing complete. Filtered images saved.")