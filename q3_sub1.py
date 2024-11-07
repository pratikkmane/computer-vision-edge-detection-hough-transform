import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

def customHoughTransform(binary_image, num_lines=5):
    # Define a range of angles from -90 to 90 degrees in radians
    theta_values = np.deg2rad(np.arange(-90, 90))
    
    # Calculate the maximum possible distance (rho) based on the image diagonal
    max_rho = int(np.hypot(*binary_image.shape))
    
    # Initialize the Hough accumulator (2D array for rho and theta)
    hough_accumulator = np.zeros((2 * max_rho, len(theta_values)), dtype=int)

    # Extract x and y coordinates of non-zero (edge) pixels from the binary image
    y_indices, x_indices = np.nonzero(binary_image)

    # Populate the Hough accumulator based on the (x, y) edge points
    for idx in range(len(x_indices)):
        x = x_indices[idx]
        y = y_indices[idx]
        # Loop through each theta and compute corresponding rho, then update accumulator
        for theta_idx in range(len(theta_values)):
            rho = int(x * np.cos(theta_values[theta_idx]) + y * np.sin(theta_values[theta_idx]))
            hough_accumulator[rho + max_rho, theta_idx] += 1

    # List to store the detected lines as (rho, theta) pairs
    detected_lines = []
    for _ in range(num_lines):
        # Find the highest peak in the accumulator
        peak_index = np.argmax(hough_accumulator)
        
        # Convert the peak index to corresponding rho and theta indices
        rho_index, theta_index = np.unravel_index(peak_index, hough_accumulator.shape)
        
        # Convert the indices back to actual rho and theta values
        rho_value = rho_index - max_rho
        theta_value = theta_values[theta_index]
        
        # Append the detected line to the list
        detected_lines.append((rho_value, theta_value))
        
        # Set the current peak in the accumulator to 0 to find the next peak
        hough_accumulator[rho_index, theta_index] = 0 

    return detected_lines

def display_lines(image_data, lines):
    # Display the image in grayscale
    plt.imshow(image_data, cmap='gray')
    
    # Iterate over each detected line (rho, theta) and plot it on the image
    for rho, theta in lines:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Compute the x, y coordinates based on rho and theta
        x_origin = cos_theta * rho
        y_origin = sin_theta * rho
        
        # Calculate two points far from each other to extend the line across the image
        x_end1 = int(x_origin + 1000 * (-sin_theta))
        y_end1 = int(y_origin + 1000 * (cos_theta))
        x_end2 = int(x_origin - 1000 * (-sin_theta))
        y_end2 = int(y_origin - 1000 * (cos_theta))
        
        # Plot the line on the image in red
        plt.plot((x_end1, x_end2), (y_end1, y_end2), '-r')
    
    # Set the title and remove axis ticks for clarity
    plt.title('Detected Lines')
    plt.axis('off')
    plt.show()

# Load the image from file path
image_path = "3.png"  
image_data = plt.imread(image_path)

# Convert RGB to grayscale if necessary
if image_data.ndim == 3:
    grayscale_image = np.dot(image_data[..., :3], [0.2989, 0.5870, 0.1140])
else:
    grayscale_image = image_data

# Perform edge detection using Canny edge detector
edge_image = feature.canny(grayscale_image)

# Apply custom Hough Transform to detect lines
detected_lines = customHoughTransform(edge_image, num_lines=5)

# Display the detected lines on the grayscale image
display_lines(grayscale_image, detected_lines)

#=----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

def customHoughTransform(binary_image, num_lines=5):
    angles = np.deg2rad(np.arange(-90, 90))
    max_distance = int(np.hypot(*binary_image.shape))
    hough_space = np.zeros((2 * max_distance, len(angles)), dtype=int)

    y_coords, x_coords = np.nonzero(binary_image)

    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]
        for angle_idx in range(len(angles)):
            rho = int(x * np.cos(angles[angle_idx]) + y * np.sin(angles[angle_idx]))
            hough_space[rho + max_distance, angle_idx] += 1

    detected_lines = []
    for _ in range(num_lines):
        idx = np.argmax(hough_space)
        rho_idx, angle_idx = np.unravel_index(idx, hough_space.shape)
        rho = rho_idx - max_distance
        angle_value = angles[angle_idx]
        detected_lines.append((rho, angle_value))
        hough_space[rho_idx, angle_idx] = 0 

    return detected_lines

def display_lines(image_data, lines):
    plt.imshow(image_data, cmap='gray')
    for rho, theta in lines:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_origin = cos_theta * rho
        y_origin = sin_theta * rho
        x_end1 = int(x_origin + 1000 * (-sin_theta))
        y_end1 = int(y_origin + 1000 * (cos_theta))
        x_end2 = int(x_origin - 1000 * (-sin_theta))
        y_end2 = int(y_origin - 1000 * (cos_theta))
        plt.plot((x_end1, x_end2), (y_end1, y_end2), '-r')
    plt.title('Detected Lines')
    plt.axis('off')
    plt.show()

image_path = "arrow.png"  
img_data = plt.imread(image_path)

if img_data.ndim == 3:
    img_gray = np.dot(img_data[..., :3], [0.2989, 0.5870, 0.1140])
else:
    img_gray = img_data

edge_map = feature.canny(img_gray)

lines_detected = customHoughTransform(edge_map, num_lines=5)

display_lines(img_gray, lines_detected)

#=----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

def customHoughTransform(binary_image, num_lines=5):
    angles = np.deg2rad(np.arange(-90, 90))
    max_distance = int(np.hypot(*binary_image.shape))
    hough_space = np.zeros((2 * max_distance, len(angles)), dtype=int)

    y_coords, x_coords = np.nonzero(binary_image)

    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]
        for angle_idx in range(len(angles)):
            rho = int(x * np.cos(angles[angle_idx]) + y * np.sin(angles[angle_idx]))
            hough_space[rho + max_distance, angle_idx] += 1

    detected_lines = []
    for _ in range(num_lines):
        idx = np.argmax(hough_space)
        rho_idx, angle_idx = np.unravel_index(idx, hough_space.shape)
        rho = rho_idx - max_distance
        angle_value = angles[angle_idx]
        detected_lines.append((rho, angle_value))
        hough_space[rho_idx, angle_idx] = 0  
    return detected_lines

def display_lines(image_data, lines):
    plt.imshow(image_data, cmap='gray')
    for rho, theta in lines:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_origin = cos_theta * rho
        y_origin = sin_theta * rho
        x_end1 = int(x_origin + 1000 * (-sin_theta))
        y_end1 = int(y_origin + 1000 * (cos_theta))
        x_end2 = int(x_origin - 1000 * (-sin_theta))
        y_end2 = int(y_origin - 1000 * (cos_theta))
        plt.plot((x_end1, x_end2), (y_end1, y_end2), '-r')
    plt.title('Detected Lines')
    plt.axis('off')
    plt.show()

image_path = "ques.png"
img_data = plt.imread(image_path)

if img_data.ndim == 3:
    img_gray = np.dot(img_data[..., :3], [0.2989, 0.5870, 0.1140])
else:
    img_gray = img_data

edge_map = feature.canny(img_gray)

lines_detected = customHoughTransform(edge_map, num_lines=5)

display_lines(img_gray, lines_detected)
