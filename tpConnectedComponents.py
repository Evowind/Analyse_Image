import numpy as np
import cv2
import random

# Finds and labels connected regions in a binary image.
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
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if image[nx, ny] != 0 and res[nx, ny] == 0:
                                res[nx, ny] = current_label
                                stack.append((nx, ny))
                current_label += 1
    return res


# Keeps only the regions whose area (number of pixels) meets the minimum threshold.
# This helps to filter out tiny specks of noise – keeping the big fish and throwing back the small fry.
def ccAreaFilter(image, size):
    labeled = ccLabel(image)
    labels, counts = np.unique(labeled, return_counts=True)
    valid_labels = labels[(counts >= size) & (labels != 0)]
    mask = np.isin(labeled, valid_labels)
    res = np.where(mask, image, 0)
    return res.astype(image.dtype)

# Everyone gets a name tag, then we check who belongs together.
def ccTwoPassLabel(image):
    rows, cols = image.shape[:2]
    res = np.zeros_like(image, dtype=np.int32)
    parent = {}
    current_label = 1

    def find(label):
        if parent[label] != label:
            parent[label] = find(parent[label])  # Path compression for efficiency
        return parent[label]

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    # First pass: assign labels and track equivalences
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

    # Second pass: resolve equivalences
    for i in range(rows):
        for j in range(cols):
            if res[i, j] != 0:
                res[i, j] = find(res[i, j])
    
    return res


# Converts the labelled image into a visual format for easier inspection.
def save_labeled_image(image, filename):
    labeled_image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(filename, labeled_image_normalized)
    print(f"Labeled image saved as '{filename}'.")


# Assigns a unique random colour to each labelled region, making it pop like a bag of Skittles.
def save_colored_labeled_image(labeled_image, filename):
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background label 0

    # Generate random colours for each label
    colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in unique_labels}

    # Create a colour image
    color_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)

    # Assign colours to each label
    for label, color in colors.items():
        color_image[labeled_image == label] = color

    # Save the coloured image
    cv2.imwrite(filename, color_image)
    print(f"Coloured labeled image saved as '{filename}'.")

def main():
    image = cv2.imread("TD4/images/binary.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # Label connected components
    labeled_image = ccLabel(image)
    area_filtered_image = ccAreaFilter(image, 50)
    two_pass_labeled_image = ccTwoPassLabel(image)

    # Save results
    save_labeled_image(labeled_image, 'labeled_image.png')
    save_labeled_image(area_filtered_image, 'area_filtered_image.png')
    save_labeled_image(two_pass_labeled_image, '2pass_labeled_image.png')
    save_colored_labeled_image(labeled_image, 'colored_labeled_image.png')

if __name__ == "__main__":
    main()
