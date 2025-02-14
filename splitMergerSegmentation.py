import numpy as np
import cv2

def split(image, threshold):
    # Let's get cracking! Begin by splitting the image into manageable regions – divide and conquer!
    h, w = image.shape
    regions = []
    stack = [(0, 0, w, h)]

    while stack:
        x, y, width, height = stack.pop()
        region = image[y:y+height, x:x+width]
        var = np.var(region)

        # If variance is low enough or the region is too small, keep it as is
        if var <= threshold or width < 2 or height < 2:
            regions.append((x, y, width, height))
        else:
            # Otherwise, split into four quadrants – divide and conquer!
            hw = width // 2
            hh = height // 2
            stack.append((x, y, hw, hh))
            stack.append((x + hw, y, width - hw, hh))
            stack.append((x, y + hh, hw, height - hh))
            stack.append((x + hw, y + hh, width - hw, height - hh))

    return regions

def merge(regions, image, threshold):
    # Time to merge – teamwork makes the dream work!
    merged = True
    while merged:
        merged = False
        new_regions = []
        used = [False] * len(regions)

        for i in range(len(regions)):
            if used[i]:
                continue
            r1 = regions[i]
            x1, y1, w1, h1 = r1
            combined = False

            for j in range(i + 1, len(regions)):
                if used[j]:
                    continue
                r2 = regions[j]
                x2, y2, w2, h2 = r2

                # Check adjacency – are we neighbours in space?
                if (x1 + w1 == x2 or x2 + w2 == x1) and (y1 < y2 + h2 and y2 < y1 + h1):
                    adjacent = True
                elif (y1 + h1 == y2 or y2 + h2 == y1) and (x1 < x2 + w2 and x2 < x1 + w1):
                    adjacent = True
                else:
                    adjacent = False

                if adjacent:
                    # Merge if variance is low enough – a match made in heaven!
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    merged_region = image[y:y+h, x:x+w]
                    merged_var = np.var(merged_region)

                    if merged_var <= threshold:
                        new_regions.append((x, y, w, h))
                        used[i] = used[j] = True
                        merged = True
                        combined = True
                        break

            if not combined and not used[i]:
                new_regions.append(r1)
                used[i] = True

        regions = new_regions

    return regions

def draw_regions(image, regions):
    # Time to showcase our work – I love rectangles!
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in regions:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return output

def main_split_merge():
    image = cv2.imread("TD4/images/camera.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image – not a great start!")
        return

    split_threshold = 150  # Variance threshold for splitting – cut where needed
    merge_threshold = 100  # Variance threshold for merging – bring it all together

    regions = split(image, split_threshold)
    merged_regions = merge(regions, image, merge_threshold)

    # Draw merged regions on the image – an artistic touch!
    result = draw_regions(image, merged_regions)
    cv2.imwrite('split_merge_result.png', result)
    print("Split-merge result saved – another masterpiece!")

if __name__ == "__main__":
    main_split_merge()
