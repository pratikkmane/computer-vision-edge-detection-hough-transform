import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class HarrisCornerDetector:
    def __init__(self, k=0.04, window_size=3, threshold=0.01, radius=10):
        self.k = k  # Harris detector free parameter (typically 0.04-0.06)
        self.window_size = window_size  # Size of window for computing covariance
        self.threshold = threshold  # Threshold for corner response
        self.radius = radius  # Non-maximum suppression radius
        
    def compute_gradients(self, image):
        """Compute x and y gradients using Sobel operator"""
        # Define Sobel operators
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]]) / 8.0
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]]) / 8.0
        
        # Compute gradients
        Ix = np.zeros_like(image, dtype=float)
        Iy = np.zeros_like(image, dtype=float)
        
        # Pad image for convolution
        padded = np.pad(image, 1, mode='edge')
        
        # Manual convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+3, j:j+3]
                Ix[i,j] = np.sum(window * sobel_x)
                Iy[i,j] = np.sum(window * sobel_y)
                
        return Ix, Iy
    
    def compute_structure_tensor(self, Ix, Iy):
        """Compute structure tensor components using windowing"""
        # Compute products of gradients
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Initialize structure tensor components
        Sxx = np.zeros_like(Ixx)
        Syy = np.zeros_like(Iyy)
        Sxy = np.zeros_like(Ixy)
        
        # Pad images
        pad = self.window_size // 2
        Ixx_pad = np.pad(Ixx, pad, mode='edge')
        Iyy_pad = np.pad(Iyy, pad, mode='edge')
        Ixy_pad = np.pad(Ixy, pad, mode='edge')
        
        # Sum over windows
        for i in range(Ixx.shape[0]):
            for j in range(Ixx.shape[1]):
                window_xx = Ixx_pad[i:i+self.window_size, j:j+self.window_size]
                window_yy = Iyy_pad[i:i+self.window_size, j:j+self.window_size]
                window_xy = Ixy_pad[i:i+self.window_size, j:j+self.window_size]
                
                Sxx[i,j] = np.sum(window_xx)
                Syy[i,j] = np.sum(window_yy)
                Sxy[i,j] = np.sum(window_xy)
                
        return Sxx, Syy, Sxy
    
    def compute_corner_response(self, Sxx, Syy, Sxy):
        """Compute Harris corner response"""
        # Compute determinant and trace
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy
        
        # Compute corner response
        R = det - self.k * trace * trace
        return R
    
    def non_maximum_suppression(self, R):
        """Perform non-maximum suppression within given radius"""
        corners = []
        R_pad = np.pad(R, self.radius, mode='edge')
        
        for i in range(self.radius, R_pad.shape[0]-self.radius):
            for j in range(self.radius, R_pad.shape[1]-self.radius):
                window = R_pad[i-self.radius:i+self.radius+1, 
                             j-self.radius:j+self.radius+1]
                if R_pad[i,j] == np.max(window) and R_pad[i,j] > self.threshold:
                    corners.append((i-self.radius, j-self.radius))
                    
        return corners
    
    def detect_corners(self, image_path):
        """Main function to detect corners in an image"""
        # Load and preprocess image
        image = np.array(Image.open(image_path).convert('L')).astype(float) / 255.0
        
        # Compute gradients
        Ix, Iy = self.compute_gradients(image)
        
        # Compute structure tensor components
        Sxx, Syy, Sxy = self.compute_structure_tensor(Ix, Iy)
        
        # Compute corner response
        R = self.compute_corner_response(Sxx, Syy, Sxy)
        
        # Perform non-maximum suppression
        corners = self.non_maximum_suppression(R)
        
        return image, R, corners
    
    def visualize_results(self, image, R, corners):
        """Visualize detected corners"""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Corner response
        plt.subplot(132)
        plt.imshow(R, cmap='hot')
        plt.title('Corner Response')
        plt.axis('off')
        
        # Detected corners
        plt.subplot(133)
        plt.imshow(image, cmap='gray')
        corners = np.array(corners)
        if len(corners) > 0:
            plt.plot(corners[:, 1], corners[:, 0], 'r+', markersize=10)
        plt.title(f'Detected Corners ({len(corners)})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize detector with parameters
    detector = HarrisCornerDetector(
        k=0.04,  # Harris response parameter
        window_size=3,  # Window size for structure tensor
        threshold=0.01,  # Corner response threshold
        radius=10  # Non-maximum suppression radius
    )
    
    # Process checkerboard image
    image1, R1, corners1 = detector.detect_corners('2-1.jpg')
    detector.visualize_results(image1, R1, corners1)
    
    # Adjust parameters for cow image (may need different parameters)
    detector.threshold = 0.005  # Lower threshold for natural image
    detector.radius = 5  # Smaller radius for denser corners
    
    # Process cow image
    image2, R2, corners2 = detector.detect_corners('2-2.jpg')
    detector.visualize_results(image2, R2, corners2)