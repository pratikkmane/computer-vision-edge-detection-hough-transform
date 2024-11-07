
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature
from scipy.io import loadmat

def trainHoughCircle(binary_img, center_point, boundary_points):
    radii_list = [np.linalg.norm(np.array(pt) - np.array(center_point)) for pt in boundary_points]
    mean_radius = np.mean(radii_list)
    circle_params = {'center': center_point, 'radius': mean_radius}
    print(f"Training: Center={center_point}, Avg Radius={mean_radius}")
    return circle_params

def testHoughCircle(new_binary_img, circle_params):
    center = circle_params['center']
    radius = circle_params['radius']
    
    img_height, img_width = new_binary_img.shape
    vote_accumulator = np.zeros((img_height, img_width), dtype=int)

    edge_y, edge_x = np.nonzero(new_binary_img)
    
    step_angle = 15 
    for x in edge_x:
        for y in edge_y:
            for theta in range(0, 360, step_angle):
                angle_rad = np.deg2rad(theta)
                a = int(x - radius * np.cos(angle_rad))
                b = int(y - radius * np.sin(angle_rad))
                if 0 <= a < img_width and 0 <= b < img_height:
                    vote_accumulator[b, a] += 1

    best_circles = []
    for _ in range(2):
        max_index = np.argmax(vote_accumulator)
        max_position = np.unravel_index(max_index, vote_accumulator.shape)
        best_circles.append(max_position)
        vote_accumulator[max_position] = 0

    print(f"Detected Circles: {best_circles}")
    return best_circles

mat_contents = loadmat('train.mat')

train_center = mat_contents['c'][0]
train_boundary_points = mat_contents['ptlist'][0]

boundary_points_list = [(int(pt[0][0]), int(pt[0][1])) for pt in train_boundary_points]

train_img_path = "train.png"
train_img_data = io.imread(train_img_path)

if train_img_data.ndim == 3:
    train_img_gray = color.rgb2gray(train_img_data)
else:
    train_img_gray = train_img_data

circle_data = trainHoughCircle(train_img_gray, train_center, boundary_points_list)

test_img_path = "test.png"
test_img_data = io.imread(test_img_path)

if test_img_data.ndim == 3:
    test_img_gray = color.rgb2gray(test_img_data)
else:
    test_img_gray = test_img_data

edge_map_test = feature.canny(test_img_gray)

detected_circles = testHoughCircle(edge_map_test, circle_data)

plt.figure(figsize=(10, 10))
plt.imshow(test_img_gray, cmap='gray')
for detected_center in detected_circles:
    plt.plot(detected_center[1], detected_center[0], 'ro', markersize=10)
    detected_circle = plt.Circle((detected_center[1], detected_center[0]), circle_data['radius'], fill=False, color='r')
    plt.gca().add_artist(detected_circle)
plt.title('Detected Circles on Original Image')
plt.axis('off')
plt.show()