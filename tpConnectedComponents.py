import numpy as np
import cv2
import random

def ccLabel(image):
    rows, cols = image.shape[:2]
    res = np.zeros_like(image, dtype=np.int32)
    current_label = 1
    for i in range(rows):
        for j in range(cols):
            if image[i, j] != 0 and res[i, j] == 0:
                stack = [(i, j)]
                res[i, j] = current_label
                while stack:
                    x, y = stack.pop()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if image[nx, ny] != 0 and res[nx, ny] == 0:
                                res[nx, ny] = current_label
                                stack.append((nx, ny))
                current_label += 1
    return res

def ccAreaFilter(image, size):
    labeled = ccLabel(image)
    labels, counts = np.unique(labeled, return_counts=True)
    valid_labels = labels[(counts >= size) & (labels != 0)]
    mask = np.isin(labeled, valid_labels)
    res = np.where(mask, image, 0)
    return res.astype(image.dtype)

def ccTwoPassLabel(image):
    rows, cols = image.shape[:2]
    res = np.zeros_like(image, dtype=np.int32)
    parent = {}
    current_label = 1

    def find(label):
        if parent[label] != label:
            parent[label] = find(parent[label])  # Path compression
        return parent[label]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    # First pass
    for i in range(rows):
        for j in range(cols):
            if image[i, j] != 0:
                neighbors = []
                if i > 0 and image[i-1, j] != 0:
                    neighbors.append((i-1, j))
                if j > 0 and image[i, j-1] != 0:
                    neighbors.append((i, j-1))
                if not neighbors:
                    res[i, j] = current_label
                    parent[current_label] = current_label
                    current_label += 1
                else:
                    labels = [res[nx, ny] for (nx, ny) in neighbors]
                    min_label = min(labels)
                    res[i, j] = min_label
                    for label in labels:
                        union(label, min_label)

    # Second pass
    for i in range(rows):
        for j in range(cols):
            if res[i, j] != 0:
                res[i, j] = find(res[i, j])

    return res

def save_labeled_image(image, filename):
    # Normalize the labeled image to 8-bit unsigned integer for saving
    labeled_image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(filename, labeled_image_normalized)
    print(f"Labeled image saved as '{filename}'.")

def save_colored_labeled_image(labeled_image, filename):
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background label 0

    # Generate random colors for each label
    colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in unique_labels}

    # Create a color image
    color_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)

    # Assign colors to each label
    for label, color in colors.items():
        color_image[labeled_image == label] = color

    # Save the colored image
    cv2.imwrite(filename, color_image)
    print(f"Colored labeled image saved as '{filename}'.")

def main():
    # Load the image in grayscale mode
    image = cv2.imread("TD4/images/binary.png", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load image.")
        return

    # Label the connected components
    labeled_image = ccLabel(image)
    area_filtered_image = ccAreaFilter(image, 50)
    two_pass_labeled_image = ccTwoPassLabel(image)

    # Save the labeled images
    save_labeled_image(labeled_image, 'labeled_image.png')
    save_labeled_image(area_filtered_image, 'area_filtered_image.png')
    save_labeled_image(two_pass_labeled_image, '2pass_labeled_image.png')

    # Save the colored labeled image
    save_colored_labeled_image(labeled_image, 'colored_labeled_image.png')

if __name__ == "__main__":
    main()