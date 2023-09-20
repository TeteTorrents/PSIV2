import cv2

def calculate_pixel_density(character_image):
    gray_char = cv2.cvtColor(character_image, cv2.COLOR_BGR2GRAY)
    _, binary_char = cv2.threshold(gray_char, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total_pixels = character_image.shape[0] * character_image.shape[1]
    foreground_pixels = cv2.countNonZero(binary_char)
    pixel_density = foreground_pixels / total_pixels
    return pixel_density

def calculate_feature_vector(character_image):
    feature_vector = []
    
    # Divide the character image into 2x4 zones (2 rows and 4 columns)
    num_rows = 4
    num_cols = 2
    row_height = character_image.shape[0] // num_rows
    col_width = character_image.shape[1] // num_cols
    
    for row in range(num_rows):
        for col in range(num_cols):
            zone = character_image[row*row_height:(row+1)*row_height, col*col_width:(col+1)*col_width]
            pixel_density = calculate_pixel_density(zone)
            feature_vector.append(pixel_density)
    
    return feature_vector

def license_plate_vectors(image, bounding_boxes):
    feature_vectors = []
    
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        character_image = image[y:y+h, x:x+w]

        if character_image.size == 0:
            # Skip empty bounding boxes (if any)
            continue
        
        pixel_density = calculate_feature_vector(character_image)
        feature_vectors.append(pixel_density)

    return feature_vectors