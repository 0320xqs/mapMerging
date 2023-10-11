import cv2

def evaluate_image_stitching(image1, image2, stitched_image):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray_stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between stitched image and original images
    diff1 = cv2.absdiff(gray_image1, gray_stitched_image)
    diff2 = cv2.absdiff(gray_image2, gray_stitched_image)

    # Calculate the mean difference
    mean_diff1 = diff1.mean()
    mean_diff2 = diff2.mean()

    # Higher mean difference indicates poorer stitching quality
    # You can define a threshold to determine whether the stitching is good or bad
    threshold = 10.0

    if mean_diff1 > threshold or mean_diff2 > threshold:
        print("Stitching quality is poor.")
    else:
        print("Stitching quality is good.")

# Example usage
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")
stitched_image = cv2.imread("stitched_image.jpg")

evaluate_image_stitching(image1, image2, stitched_image)