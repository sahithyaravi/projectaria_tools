
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R
import json

DATASET_ROOT = "/data/post_intern_sahithya/ariav5/projectaria_tools_ase_data/"  # Specify your own dataset path
SCENE_ID = 1  # Select a scene id

dataset_path = Path(DATASET_ROOT)
print("Chosen ASE data path: ", dataset_path)
print(f"Using Scene {SCENE_ID} for these examples")

scene_path = dataset_path / str(SCENE_ID)

# Load instance to class mapping
instance_mapping_path = scene_path / "object_instances_to_classes.json"
with open(instance_mapping_path, 'r') as f:
    instance_to_class = json.load(f)

# Create a color mapping for classes
unique_classes = list(set(instance_to_class.values()))
class_colors = {}
for i, class_name in enumerate(unique_classes):
    # Generate a unique color for each class
    hue = i / len(unique_classes)
    rgb = colors.hsv_to_rgb([hue, 0.8, 0.8])
    class_colors[class_name] = rgb

from code_snippets.readers import read_points_file, read_trajectory_file, read_language_file



trajectory_path = scene_path / "trajectory.csv"
trajectory = read_trajectory_file(trajectory_path)

language_path = scene_path / "ase_scene_language.txt"
entities = read_language_file(language_path)

from code_snippets.interpreter import language_to_bboxes


entity_boxes = language_to_bboxes(entities)


import sys
sys.path.append("/Users/sahithyaravi/Documents/projectaria_tools")
from projectaria_tools.projects import ase

def transform_3d_points(transform, points):
    N = len(points)
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    transformed_points_h = (transform @ points_h.T).T
    transformed_points = transformed_points_h[:, :-1]
    return transformed_points
    
device = ase.get_ase_rgb_calibration()

trajectory_path = scene_path / "trajectory.csv"
trajectory = read_trajectory_file(trajectory_path)



def random_bright_colormap(num_colors=1024):
    bright_colors = np.random.rand(num_colors, 3)
    bright_colors = (bright_colors + 1) / 2
    return colors.ListedColormap([c for c in bright_colors])

scene_path = dataset_path / str(SCENE_ID)

rgb_dir = scene_path / "rgb"
depth_dir = scene_path / "depth"
instance_dir = scene_path / "instances"

# Choose a random frame to plot from the scene's images
num_frames = len(list(rgb_dir.glob("*.jpg")))
frame_idx = np.random.randint(num_frames)

frame_id = str(frame_idx).zfill(7)

rgb_path = rgb_dir / f"vignette{frame_id}.jpg"
depth_path = depth_dir / f"depth{frame_id}.png"
instance_path = instance_dir / f"instance{frame_id}.png"

rgb = Image.open(rgb_path)
depth = Image.open(depth_path)
instances = Image.open(instance_path)
instance_array = np.array(instances)


# Part 0: Bird Eye View


from projectaria_tools.core.sophus import SE3
from projectaria_tools.core import calibration
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod

def rotate_se3_about_forward_axis(T_scene_camera, theta):
    """
    Rotates an SE3 matrix about its forward (z) axis by an angle theta.

    Parameters:
    T_scene_camera (numpy.ndarray): The original 4x4 SE3 matrix.
    theta (float): The angle by which to rotate about the z-axis (in radians).

    Returns:
    numpy.ndarray: The rotated 4x4 SE3 matrix.
    """
    # Extract the rotation part (3x3) from the SE3 matrix
    R_scene_camera = T_scene_camera.rotation().to_matrix()
    
    # Extract the translation part (3x1) from the SE3 matrix
    t_scene_camera = T_scene_camera.translation()
    
    # Create the rotation matrix for rotation about the z-axis
    R_z = R.from_euler('z', theta).as_matrix()
    
    # Apply the rotation to the original rotation part
    R_scene_camera_rotated = R_scene_camera @ R_z
    
    # Combine the rotated rotation part with the original translation part
    T_scene_camera_rotated = np.eye(4)
    T_scene_camera_rotated[:3, :3] = R_scene_camera_rotated
    T_scene_camera_rotated[:3, 3] = t_scene_camera
    
    return SE3.from_matrix(T_scene_camera_rotated)


import cv2
def undistort_image(file_path, device, is_depth=False):
    
    if is_depth:
        raw_image = np.array(Image.open(file_path)).astype(np.float32)
    else:
        raw_image = Image.open(file_path)
    rectified_array = calibration.distort_by_calibration(raw_image, device, device, InterpolationMethod.BILINEAR)
    return rectified_array
#------------------------------
def render_topdown_from_projected(points_world, colors, grid_resolution=0.01):
    x, y, z = points_world[:, 0], points_world[:, 1], points_world[:, 2]
    valid = (z > 0.1) & (z < 2.5)
    x, y, colors = x[valid], y[valid], colors[valid]

    min_x, min_y = x.min(), y.min()
    max_x, max_y = x.max(), y.max()

    W = int(np.ceil((max_x - min_x) / grid_resolution))
    H = int(np.ceil((max_y - min_y) / grid_resolution))
    img = np.zeros((H, W, 3), dtype=np.uint8)

    xi = ((x - min_x) / grid_resolution).astype(int)
    yi = ((y - min_y) / grid_resolution).astype(int)

    img[H - yi - 1, xi] = colors
    return img

all_points = []
all_colors = []
all_instances = []
all_rgb = []

T_Device_Cam = device.get_transform_device_camera()
image_size = device.get_image_size()
width, height = image_size[0], image_size[1]

rays = np.empty((height, width, 3))
for u in range(width):
    for v in range(height):
        ray = device.unproject([u, v])
        if ray is not None:
            ray = ray / np.linalg.norm(ray)
        rays[v, u] = ray

for frame_idx in range(0, num_frames, 10):
    frame_id = str(frame_idx).zfill(7)
    rgb_path = rgb_dir / f"vignette{frame_id}.jpg"
    depth_path = depth_dir / f"depth{frame_id}.png"
    instance_path = instance_dir / f"instance{frame_id}.png"
    if not rgb_path.exists() or not depth_path.exists():
        continue

    rgb = undistort_image(rgb_path, device, is_depth=False)
    depth = undistort_image(depth_path, device, is_depth=True)
    instances = Image.open(instance_path)
    instance_array = np.array(instances)
    unique_instance_ids = np.unique(instance_array)

    print("Unique instance IDs in instance_array:", np.unique(instance_array))

    rgb = np.rot90(np.array(rgb), 3)
    depth = np.rot90(np.array(depth).astype(np.float32), 3)
    instance_array = np.rot90(instance_array, 3)
    
    T_Scene_Device = trajectory["Ts_world_from_device"][frame_idx]
    T_Device_Cam_rot = rotate_se3_about_forward_axis(T_Device_Cam, 3 * np.pi / 2)
    T_Scene_Cam = T_Scene_Device @ T_Device_Cam_rot.to_matrix()

    valid_mask = (rays is not None) & (depth > 0)
    indices = np.argwhere(valid_mask)
    u_indices, v_indices = indices[:, 1], indices[:, 0]
    rays_selected = rays[v_indices, u_indices]
    depth_selected = depth[v_indices, u_indices] / 1000.0

    p_in_cam = (depth_selected[:, None] * rays_selected)
    p_in_scene = transform_3d_points(T_Scene_Cam, p_in_cam)

    
    instance_ids = instance_array[v_indices, u_indices]
    
    # Convert instance IDs to unique class colors
    instance_colors = np.zeros((len(instance_ids), 3), dtype=np.uint8)
    for i, instance_id in enumerate(instance_ids):
        class_name = instance_to_class.get(str(instance_id), "unknown")
        instance_colors[i] = np.array(class_colors.get(class_name, [0, 0, 0])) * 255

    
    rgb_values = rgb[v_indices, u_indices]

    all_points.append(p_in_scene)
    all_colors.append(instance_colors)
    all_instances.append(instance_ids)
    all_rgb.append(rgb_values) 

    print("Sample instance_ids for valid points:", instance_ids[:20])
    print("Sample instance_colors:", instance_colors[:5])

points_world = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)
instance_ids = np.concatenate(all_instances, axis=0)
rgb_values = np.concatenate(all_rgb, axis=0)


plt.figure(figsize=(20, 10))


plt.subplot(1, 2, 1)
rgb_img = render_topdown_from_projected(points_world, rgb_values)
plt.imshow(rgb_img)
plt.title("Top-Down RGB Map")
plt.axis("off")

# Instance Map
plt.subplot(1, 2, 2)
instance_img = render_topdown_from_projected(points_world, colors)
plt.imshow(instance_img)
plt.title("Top-Down Instance Map")
plt.axis("off")

#legen
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, label=class_name) 
                  for class_name, color in class_colors.items()]
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("topdown_maps.png", dpi=300, bbox_inches='tight')
plt.show()



