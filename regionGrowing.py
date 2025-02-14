import numpy as np
import cv2
from collections import deque
import random

def region_growing(image, seeds, threshold=10):
    # Initialize the region with the seed points
    region = set()
    for seed in seeds:
        region.add(seed)

    # Initialize the queue with the seed points
    queue = deque(seeds)

    # Define the connectivity (4-connectivity)
    connects = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Calculate initial mean intensity
    mean_intensity = np.mean([image[seed] for seed in seeds])
    print(f"Starting region growing with seeds: {seeds}, initial mean intensity: {mean_intensity}")

    iteration = 0
    while queue:
        iteration += 1
        if iteration % 1000 == 0:  # Print progress every 1000 iterations
            print(f"Iteration {iteration}, queue size: {len(queue)}, region size: {len(region)}")

        # Pop a pixel from the queue
        current_pixel = queue.popleft()

        # Check the neighbors
        for connect in connects:
            neighbor = (current_pixel[0] + connect[0], current_pixel[1] + connect[1])

            # Check if the neighbor is within the image bounds
            if 0 <= neighbor[0] < image.shape[0] and 0 <= neighbor[1] < image.shape[1]:
                if neighbor not in region:
                    # Calculate the homogeneity criterion
                    if abs(int(image[neighbor]) - mean_intensity) < threshold:
                        region.add(neighbor)
                        queue.append(neighbor)
                        mean_intensity = np.mean([image[pixel] for pixel in region])

    print(f"Completed region growing with {len(region)} pixels in the region.")
    return region

def main_region_growing():
    image = cv2.imread("TD4/images/camera.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image")
        return

    seeds_list = [[(185, 295)], [(215, 345)], [(240, 240)], [(100, 300)], [(300, 200)], [(150, 250)]]  # Example seed points (y, x) for different regions
    threshold = 10  # Variance threshold

    # Create an output image with 3 channels for color
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Assign random colors to each region
    for i, seeds in enumerate(seeds_list):
        print(f"Processing region {i+1}/{len(seeds_list)} with seeds: {seeds}")
        region = region_growing(image, seeds, threshold)
        color = [random.randint(0, 255) for _ in range(3)]  # Random color for the region
        for pixel in region:
            output_image[pixel] = color

    cv2.imwrite('region_growing_result_colored_camera.png', output_image)
    print("Region growing result with random colors saved.")

if __name__ == "__main__":
    main_region_growing()