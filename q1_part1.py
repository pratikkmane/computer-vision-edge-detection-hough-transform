import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
            
    return kernel / kernel.sum()  # Normalize the kernel

def convolve2d(image, kernel):
    """Perform 2D convolution without using scipy.signal.convolve2d."""
    # Get image and kernel dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Calculate padding
    pad_height = k_height // 2
    pad_width = k_width // 2
    
    # Create padded image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 
                         mode='edge')
    
    # Initialize output
    output = np.zeros_like(image)
    
    # Perform convolution
    for i in range(i_height):
        for j in range(i_width):
            output[i, j] = np.sum(
                padded_image[i:i+k_height, j:j+k_width] * kernel
            )
    
    return output

def gradient_edge_detector(image_path, sigma=1.0, kernel_size=5):
    """
    Detect edges using gradient-based method.
    
    Parameters:
    image_path: str - Path to the input image
    sigma: float - Standard deviation for Gaussian smoothing
    kernel_size: int - Size of the Gaussian kernel (should be odd)
    """
    # Load and convert image to grayscale
    image = Image.open(image_path).convert('L')
    image = np.array(image).astype(float) / 255.0  # Convert to double [0,1]
    
    # Create Gaussian kernel for smoothing
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Smooth the image
    smoothed_image = convolve2d(image, gaussian_kernel)
    
    # Create derivative kernels
    dx_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]) / 8.0  # Sobel x-derivative
    
    dy_kernel = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]]) / 8.0   # Sobel y-derivative
    
    # Compute derivatives
    grad_x = convolve2d(smoothed_image, dx_kernel)
    grad_y = convolve2d(smoothed_image, dy_kernel)
    
    # Compute gradient magnitude and orientation
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_orientation = np.arctan2(grad_y, grad_x)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Gradient magnitude
    plt.subplot(132)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    # Gradient orientation using quiver
    plt.subplot(133)
    y, x = np.mgrid[0:image.shape[0]:20, 0:image.shape[1]:20]
    dx = np.cos(gradient_orientation[::20, ::20])
    dy = np.sin(gradient_orientation[::20, ::20])
    plt.quiver(x, y, dx, dy, gradient_magnitude[::20, ::20],
              angles='xy', scale_units='xy', scale=0.1)
    plt.title('Gradient Orientation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gradient_magnitude, gradient_orientation

# Example usage
if __name__ == "__main__":
    # Example path - replace with your image path
    image_path = "1.png"
    
    # Detect edges with different sigma values
    magnitude, orientation = gradient_edge_detector(image_path, sigma=1.0, kernel_size=5)