import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button
import matplotlib.patches as patches

class BoundaryTracer:
    def __init__(self, image_path):
        # Load and preprocess image
        self.image = np.array(Image.open(image_path).convert('L')).astype(float) / 255.0
        self.edge_map = None
        self.seed_points = []
        self.boundaries = []
        self.current_object = 0
        
    def detect_edges(self, sigma=1.0, threshold=0.1):
        """Detect edges using Sobel operators"""
        # Gaussian smoothing kernel
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        y = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-(X**2 + Y**2)/(2*sigma**2))
        gaussian = gaussian / gaussian.sum()
        
        # Apply Gaussian smoothing
        smoothed = np.zeros_like(self.image)
        pad = kernel_size//2
        padded_img = np.pad(self.image, pad, mode='edge')
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                smoothed[i,j] = np.sum(padded_img[i:i+kernel_size, j:j+kernel_size] * gaussian)
        
        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
        
        # Compute gradients
        grad_x = np.zeros_like(smoothed)
        grad_y = np.zeros_like(smoothed)
        padded = np.pad(smoothed, 1, mode='edge')
        
        for i in range(smoothed.shape[0]):
            for j in range(smoothed.shape[1]):
                grad_x[i,j] = np.sum(padded[i:i+3, j:j+3] * sobel_x)
                grad_y[i,j] = np.sum(padded[i:i+3, j:j+3] * sobel_y)
        
        # Compute magnitude and threshold
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        self.edge_map = (magnitude > threshold).astype(float)
        
    def trace_boundary(self, seed_point):
        """Trace boundary starting from seed point using 8-connectivity"""
        directions = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        boundary = [seed_point]
        current = seed_point
        visited = set([seed_point])
        
        while True:
            found_next = False
            # Check all 8 directions
            for dy, dx in directions:
                y, x = current[0] + dy, current[1] + dx
                
                # Check bounds
                if (y < 0 or y >= self.edge_map.shape[0] or 
                    x < 0 or x >= self.edge_map.shape[1]):
                    continue
                
                # If edge pixel found and not visited
                if self.edge_map[y,x] > 0 and (y,x) not in visited:
                    boundary.append((y,x))
                    visited.add((y,x))
                    current = (y,x)
                    found_next = True
                    break
            
            # If no unvisited edge pixel found or back to start
            if not found_next or len(boundary) > 1 and boundary[-1] == boundary[0]:
                break
                
            # Prevent infinite loops
            if len(boundary) > 1000:
                break
                
        return boundary
    
    def onclick(self, event):
        """Handle mouse clicks to get seed points"""
        if event.inaxes != self.ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if self.edge_map[y,x] > 0:  # If clicked on edge pixel
            self.seed_points.append((y,x))
            boundary = self.trace_boundary((y,x))
            self.boundaries.append(boundary)
            
            # Plot the boundary
            boundary_y, boundary_x = zip(*boundary)
            self.ax.plot(boundary_x, boundary_y, 'r-', linewidth=2)
            self.fig.canvas.draw()
            
            print(f"Traced boundary {len(self.boundaries)} with {len(boundary)} points")
    
    def process_image(self):
        """Main processing pipeline"""
        # Detect edges
        self.detect_edges(sigma=1.0, threshold=0.1)
        
        # Setup interactive plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.edge_map, cmap='gray')
        self.ax.set_title('Click on edge pixels to trace boundaries')
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Add done button
        ax_done = plt.axes([0.81, 0.05, 0.1, 0.075])
        btn_done = Button(ax_done, 'Done')
        btn_done.on_clicked(lambda event: plt.close())
        
        plt.show()
        
        # Display final results
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(self.image, cmap='gray')
        plt.title('Original Image')
        
        # Edge map
        plt.subplot(132)
        plt.imshow(self.edge_map, cmap='gray')
        plt.title('Edge Map')
        
        # Traced boundaries
        plt.subplot(133)
        plt.imshow(self.image, cmap='gray')
        for boundary in self.boundaries:
            boundary_y, boundary_x = zip(*boundary)
            plt.plot(boundary_x, boundary_y, 'r-', linewidth=2)
        plt.title('Traced Boundaries')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    tracer = BoundaryTracer('1.png')
    tracer.process_image()