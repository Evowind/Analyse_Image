#TODO : Deposer pdf et raport de deux page~

import numpy as np
import cv2

def threshold_otsu(image):
    # Compute the histogram of the image
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Normalize the histogram
    hist = hist.astype(float) / hist.sum()

    # Compute the cumulative sum of the histogram
    cum_sum = np.cumsum(hist)

    # Compute the cumulative mean
    cum_mean = np.cumsum(hist * np.arange(256))

    # Initialize variables
    max_variance = 0
    threshold = 0

    # Iterate over all possible thresholds
    for t in range(1, 256):
        # Background and foreground probabilities
        w_b = cum_sum[t]
        w_f = 1.0 - w_b

        # Background and foreground means
        if w_b > 0 and w_f > 0:
            mean_b = cum_mean[t] / w_b
            mean_f = (cum_mean[-1] - cum_mean[t]) / w_f

            # Calculate between-class variance
            variance = w_b * w_f * (mean_b - mean_f) ** 2

            # Update the threshold if this variance is the highest so far
            if variance > max_variance:
                max_variance = variance
                threshold = t

    # Apply the threshold to the image
    res = np.where(image > threshold, 255, 0).astype(np.uint8)

    return res

def main():
    # Load the image in grayscale mode
    image = cv2.imread("TD4/images/cat.jpg", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load image.")
        return

    # Apply Otsu's thresholding
    otsu_image = threshold_otsu(image)

    # Save the thresholded image
    cv2.imwrite('otsu_thresholded_image.png', otsu_image)
    print("Otsu thresholded image saved as 'otsu_thresholded_image.png'.")

if __name__ == "__main__":
    main()
