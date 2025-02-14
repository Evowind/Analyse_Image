import numpy as np
import cv2
from collections import deque
import random

def region_growing(image, seeds, threshold=10):
    # Start with the given seed points – every great forest starts with a tiny seed!
    region = set(seeds)
    queue = deque(seeds)

    # Define 4-connectivity – neighbours just a step away
    connects = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Calculate initial mean intensity – setting the tone, literally
    mean_intensity = np.mean([image[seed] for seed in seeds])
    print(f"Starting region growing with seeds: {seeds}, initial mean intensity: {mean_intensity}")

    iteration = 0
    while queue:
        iteration += 1
        if iteration % 1000 == 0:  # Keep tabs on progress
            print(f"Iteration {iteration}, queue size: {len(queue)}, region size: {len(region)}")

        # Fetch the next pixel – let's keep this queue rolling!
        current_pixel = queue.popleft()

        # Inspect the neighbours
        for connect in connects:
            neighbor = (current_pixel[0] + connect[0], current_pixel[1] + connect[1])

            # Stay within the image bounds – no wandering off!
            if 0 <= neighbor[0] < image.shape[0] and 0 <= neighbor[1] < image.shape[1]:
                if neighbor not in region:
                    # Check homogeneity – are we a match?
                    if abs(int(image[neighbor]) - mean_intensity) < threshold:
                        region.add(neighbor)
                        queue.append(neighbor)
                        mean_intensity = np.mean([image[pixel] for pixel in region])

    # Done!, let's return the region
    print(f"Completed region growing with {len(region)} pixels in the region.")
    return region

def main_region_growing():
    image = cv2.imread("TD4/images/camera.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        # Ouch!
        print("Error loading image – this is not a great start!")
        return

    seeds_list = [[(185, 295)], [(215, 345)], [(240, 240)], [(100, 300)], [(300, 200)], [(150, 250)]]  # Sample seeds (y, x)
    threshold = 10  # Variance threshold, keep it reasonable (or not)!

    # Create an output image with 3 channels
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Assign random colours to each region and grow
    for i, seeds in enumerate(seeds_list):
        print(f"Processing region {i+1}/{len(seeds_list)} with seeds: {seeds}")
        region = region_growing(image, seeds, threshold)
        colour = [random.randint(0, 255) for _ in range(3)]  # A splash of randomness
        for pixel in region:
            output_image[pixel] = colour

    cv2.imwrite('region_growing_result_colored_camera.png', output_image)
    print("Region growing result with random colours saved – masterpiece complete!")

if __name__ == "__main__":
    main_region_growing()