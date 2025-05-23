import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R
import cv2
import sys
from projectaria_tools.core.sophus import SE3
from tqdm import tqdm
from collections import defaultdict

# --------------------------- Helper Functions --------------------------- #
def transform_3d_points(transform, points):
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (transform @ points_h.T).T[:, :-1]

def rotate_se3_about_forward_axis(T_scene_camera, theta):
    R_z = R.from_euler('z', theta).as_matrix()
    R_scene_camera = T_scene_camera.rotation().to_matrix()
    t_scene_camera = T_scene_camera.translation()

    R_rotated = R_scene_camera @ R_z
    T_rotated = np.eye(4)
    T_rotated[:3, :3] = R_rotated
    T_rotated[:3, 3] = t_scene_camera
    return SE3.from_matrix(T_rotated)


def camera_to_world(xyz_cam, frame_idx, trajectory, device):
    # Step 1: Camera → Device
    T_device_from_camera = device.get_transform_device_camera()
    T_device_from_camera_rot = rotate_se3_about_forward_axis(T_device_from_camera, 3 * np.pi / 2)

    xyz_cam_h = np.append(xyz_cam, 1.0)
    xyz_device_rot_h = T_device_from_camera_rot.to_matrix() @ xyz_cam_h

    # Step 2: Device → World
    T_world_from_device = trajectory["Ts_world_from_device"][frame_idx]
    xyz_world_h = T_world_from_device @ xyz_device_rot_h

    return xyz_world_h[:3]

def get_clipped_points(points_world, instance_ids, colors, rgb_values,
                       z_bounds=(0.1, 2.5), crop_percentiles=(1, 99)):
    """
    Filters and crops 3D points and corresponding attributes.
    Returns filtered x, y, instance_ids, colors, rgb_values, and bounding box.
    """
    x, y, z = points_world[:, 0], points_world[:, 1], points_world[:, 2]
    valid_z = (z > z_bounds[0]) & (z < z_bounds[1])
    x, y = x[valid_z], y[valid_z]
    instance_ids = instance_ids[valid_z]
    colors = colors[valid_z]
    rgb_values = rgb_values[valid_z]

    x_min, x_max = np.percentile(x, crop_percentiles)
    y_min, y_max = np.percentile(y, crop_percentiles)
    inside_bounds = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    x = x[inside_bounds]
    y = y[inside_bounds]
    instance_ids = instance_ids[inside_bounds]
    colors = colors[inside_bounds]
    rgb_values = rgb_values[inside_bounds]

    return x, y, instance_ids, colors, rgb_values, (x_min, y_min, x_max, y_max)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to ASE dataset root")
    parser.add_argument("--scene_id", type=int, required=True, help="Scene ID to process")
    parser.add_argument("--output_folder", type=Path, default="bird_eye_view", help="Output folder for results")
    parser.add_argument("--qa_file_path", type=str, default="/Users/sahithyaravi/Documents/projectaria_tools/spatial_reasoning_qa_val_natural.json", help="Path to QA file")
    return parser.parse_args()

def render_topdown_from_projected(x, y, colors, bbox, grid_resolution=0.01):
    x_min, y_min, x_max, y_max = bbox
    W = int(np.ceil((x_max - x_min) / grid_resolution))
    H = int(np.ceil((y_max - y_min) / grid_resolution))
    xi = ((x - x_min) / grid_resolution).astype(int)
    yi = ((y - y_min) / grid_resolution).astype(int)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[H - yi - 1, xi] = colors
    return img, (x_min, y_min, H, W)

def undistort_image(file_path, device, is_depth=False):
    from projectaria_tools.core import calibration
    from projectaria_tools.core.image import InterpolationMethod

    image = np.array(Image.open(file_path)).astype(np.float32) if is_depth else Image.open(file_path)
    return calibration.distort_by_calibration(image, device, device, InterpolationMethod.BILINEAR)

def get_ray_directions(device):
    width, height = device.get_image_size()
    rays = np.empty((height, width, 3))
    for u in range(width):
        for v in range(height):
            ray = device.unproject([u, v])
            if ray is not None:
                ray = ray / np.linalg.norm(ray)
            rays[v, u] = ray
    return rays

def convert_instance_to_color(instance_to_class):
    unique_classes = list(set(instance_to_class.values()))
    class_colors = {}
    for i, name in enumerate(unique_classes):
        hue = i / len(unique_classes)
        class_colors[name] = colors.hsv_to_rgb([hue, 0.8, 0.8])
    return class_colors

def plot_camera_trajectory(x, y, instance_ids, rgb_img, class_colors, instance_to_class, trajectory, bbox, grid_resolution, H, min_x, min_y):
    x_min, y_min, H, W = bbox[0], bbox[1], rgb_img.shape[0], rgb_img.shape[1]
    xi = ((x - x_min) / grid_resolution).astype(int)
    yi = ((y - y_min) / grid_resolution).astype(int)

    instance_map = np.zeros((H, W), dtype=np.int32)
    instance_map[H - yi - 1, xi] = instance_ids

    rgb_img_outline = rgb_img.copy()
    unique_ids = np.unique(instance_map)
    centroids = []
    labels = []

    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        mask = (instance_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        class_name = instance_to_class.get(str(inst_id), "unknown")
        color_rgb = np.array(class_colors.get(class_name, [1, 1, 1])) * 255
        color_bgr = tuple(int(x) for x in color_rgb[::-1])
        cv2.drawContours(rgb_img_outline, contours, -1, color_bgr, 2)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            labels.append(class_name)

    # Camera positions
    camera_positions = []
    for T in trajectory["Ts_world_from_device"]:
        if hasattr(T, "translation"):
            t = T.translation()
        else:
            t = T[:3, 3]
        camera_positions.append(t)
    camera_positions = np.array(camera_positions)
    cam_x, cam_y = camera_positions[:, 0], camera_positions[:, 1]
    cam_xi = ((cam_x - min_x) / grid_resolution).astype(int)
    cam_yi = H - ((cam_y - min_y) / grid_resolution).astype(int) - 1

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img_outline)
    plt.plot(cam_xi, cam_yi, color='cyan', linewidth=2, label='Camera Trajectory')
    plt.scatter(cam_xi, cam_yi, color='red', s=10, label='Camera Positions')
    plt.title("Top-Down RGB Map with Instance Outlines and Camera Trajectory")
    plt.axis("off")

    for (cx, cy), class_name in zip(centroids, labels):
        plt.text(cx, cy, class_name, color='yellow', fontsize=8, ha='center', va='center', weight='bold')

    arrow_scale = 20
    camera_positions = []

    for T in trajectory["Ts_world_from_device"]:
        if hasattr(T, "rotation"):
            Rmat = T.rotation().to_matrix()
            t = T.translation()
        else:
            Rmat = T[:3, :3]
            t = T[:3, 3]

        forward = Rmat[:, 2]
        x_img = int((t[0] - min_x) / grid_resolution)
        y_img = H - int((t[1] - min_y) / grid_resolution) - 1
        dx = arrow_scale * forward[0]
        dy = -arrow_scale * forward[1]

        plt.arrow(x_img, y_img, dx, dy, color='magenta', head_width=8, head_length=10,
                length_includes_head=True, alpha=0.7)

        camera_positions.append((x_img, y_img))


    # Offset markers slightly from the path
    # Bright green and red colors
    bright_green = '#00FF00'
    bright_red = '#FF3030'
    bright_blue = '#0000FF'
    offset_px = 0

    # Start marker
    x_start, y_start = camera_positions[0]
    plt.scatter(x_start, y_start, color=bright_green, s=80, marker='^', label='Start')
    plt.text(x_start, y_start - offset_px - 15, 'Start', color=bright_green, fontsize=10, ha='center', weight='bold')

    # End marker
    x_end, y_end = camera_positions[-1]
    plt.scatter(x_end, y_end, color=bright_blue, s=80, marker='X', label='End')
    plt.text(x_end, y_end + offset_px - 15, 'End', color=bright_blue, fontsize=10, ha='center', weight='bold')


    plt.legend(loc='lower right')
    plt.tight_layout()
    #plt.show()

def mark_object(qmeta, rgb_img, trajectory, start_frame, end_frame, min_x, min_y, H, W):
    object_a_xyz = qmeta["bbox_centers_camera_a_frame1"]
    object_b_xyz = qmeta["bbox_centers_camera_b_frame2"]
    # Convert object centers to world
    object_a_world = camera_to_world(object_a_xyz, start_frame, trajectory, device)
    object_b_world = camera_to_world(object_b_xyz, end_frame, trajectory, device)

    # Convert bbox centers to top-down image space
    def project_to_bev(xyz):
        x_img = int((xyz[0] - min_x) / 0.01)
        y_img = H - int((xyz[1] - min_y) / 0.01) - 1
        return x_img, y_img

    a_x, a_y = project_to_bev(object_a_world)
    b_x, b_y = project_to_bev(object_b_world)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img.copy())
    plt.scatter([a_x, b_x], [a_y, b_y], c=["lime", "orange"], s=150, marker='X', label="QA Objects")
    plt.text(a_x, a_y - 5, "Object A", color="lime", fontsize=20, ha='center')
    plt.text(b_x, b_y - 5, "Object B", color="orange", fontsize=20, ha='center')



# --------------------------- Main Pipeline --------------------------- #

def main():
    args = parse_args()
    scene_path = Path(args.dataset_path) / str(args.scene_id)
    print(f"Using dataset: {scene_path}")

    # load qa json
    with open(args.qa_file_path, "r") as f:
        qa_data = json.load(f)
    
    # For each scene_id, get all "frame_ids" and  "bbox_centers_camera_a_frame1" and "bbox_centers_camera_b_frame2" and save it in a map
    question_metadata_per_scene = defaultdict(list)
    for entry in qa_data:
        if entry["scene_id"] == str(args.scene_id):
            frame_id = entry["frame_ids"]
            bboxes_cam_a = entry["bbox_centers_camera_a_frame1"]
            bboxes_cam_b = entry["bbox_centers_camera_b_frame2"]
            meta_data = {
                "frame_id": frame_id,
                "bbox_centers_camera_a_frame1": bboxes_cam_a,
                "bbox_centers_camera_b_frame2": bboxes_cam_b
            }
            question_metadata_per_scene[args.scene_id].append(meta_data)
   

    print(f"Number of questions in scene {args.scene_id}: {len(question_metadata_per_scene[args.scene_id])}")

    # Strip dataset root path until ariav5
    output_folder = args.output_folder
    # Make the output folder if it doesn't exist
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)


    # Imports that rely on sys.path
    sys.path.append("/Users/sahithyaravi/Documents/projectaria_tools")
    from projectaria_tools.projects import ase
    from code_snippets.readers import read_points_file, read_trajectory_file, read_language_file
    from code_snippets.interpreter import language_to_bboxes
   

    # Load device calibration and trajectory
    device = ase.get_ase_rgb_calibration()
    trajectory = read_trajectory_file(scene_path / "trajectory.csv")
    rays = get_ray_directions(device)

    # Load semantic data
    instance_to_class = json.load(open(scene_path / "object_instances_to_classes.json"))
    class_colors = convert_instance_to_color(instance_to_class)

    # Directories
    rgb_dir = scene_path / "rgb"
    depth_dir = scene_path / "depth"
    instance_dir = scene_path / "instances"
    num_frames = len(list(rgb_dir.glob("*.jpg")))

    T_Device_Cam = device.get_transform_device_camera()
    all_points, all_colors, all_instances, all_rgb = [], [], [], []

    for frame_idx in tqdm(range(0, num_frames, 10)):
        frame_id = str(frame_idx).zfill(7)
        paths = {
            "rgb": rgb_dir / f"vignette{frame_id}.jpg",
            "depth": depth_dir / f"depth{frame_id}.png",
            "instance": instance_dir / f"instance{frame_id}.png"
        }

        if not paths["rgb"].exists() or not paths["depth"].exists():
            continue

        rgb = np.rot90(np.array(undistort_image(paths["rgb"], device, False)), 3)
        depth = np.rot90(np.array(undistort_image(paths["depth"], device, True)), 3)
        instance_array = np.rot90(np.array(Image.open(paths["instance"])), 3)

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
        instance_colors = np.array([
            np.array(class_colors.get(instance_to_class.get(str(i), "unknown"), [0, 0, 0])) * 255
            for i in instance_ids
        ]).astype(np.uint8)

        rgb_values = rgb[v_indices, u_indices]
        all_points.append(p_in_scene)
        all_colors.append(instance_colors)
        all_instances.append(instance_ids)
        all_rgb.append(rgb_values)

    # Merge all data
    points_world = np.concatenate(all_points)
    rgb_values = np.concatenate(all_rgb)
    colors = np.concatenate(all_colors)
    instance_ids = np.concatenate(all_instances)

    x, y, instance_ids_filtered, colors_filtered, rgb_values_filtered, bbox = get_clipped_points(
    points_world, instance_ids, colors, rgb_values)

    rgb_img, meta = render_topdown_from_projected(x, y, rgb_values_filtered, bbox)
    inst_img, _   = render_topdown_from_projected(x, y, colors_filtered, bbox)
    min_x, min_y, H, W = meta

    # Plot top-down maps
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Top-Down RGB Map")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(inst_img)
    plt.title("Top-Down Instance Map")
    plt.axis("off")

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, label=cls)
                       for cls, color in class_colors.items()]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{args.output_folder}/{args.scene_id}_overlay.png", dpi=300, bbox_inches='tight')


    # Per question traj plots
    question_metadata = question_metadata_per_scene[args.scene_id]

    for i, qmeta in enumerate(question_metadata):
        question_frames = qmeta["frame_id"]
        # Draw sub-trajectory just for frames in this question
        start_frame = int(question_frames[0])
        end_frame = int(question_frames[-1])  # assuming it's sorted
        frames_range = list(range(start_frame, end_frame + 1))

        traj_subset = {
            "Ts_world_from_device": [trajectory["Ts_world_from_device"][f] for f in frames_range]
        }
        plot_camera_trajectory(
            x, y,
            instance_ids_filtered,
            rgb_img,
            class_colors,
            instance_to_class,
            traj_subset,
            bbox,
            grid_resolution=0.01,
            H=H,
            min_x=min_x,
            min_y=min_y,
        )

        plt.title(f"Scene {args.scene_id} - Question {i}")
        plt.savefig(f"{args.output_folder}/{args.scene_id}_{start_frame}_{end_frame}_bev.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # time taken to complete
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Done!")
