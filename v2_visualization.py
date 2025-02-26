from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor
import imageio.v3 as iio
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from v2_perception import id2rgb, id2label, scale

class Message():
    def __init__(self, recording_img, imgnps, occupancy_data, obs, iou, latency):
        self.recording_img = recording_img
        self.imgnps = imgnps
        self.occupancy_data = occupancy_data
        self.obs = obs
        self.iou = iou
        self.latency = latency

class Visualizer():
    def __init__(self):
        self.q = Queue()
        self.p = Process(target=visualizer, args=(self.q,))
        self.p.start()

    def send(self, recording_img, imgnps, occupancy_data, obs, iou, latency):
        msg = Message(recording_img, imgnps, occupancy_data, obs, iou, latency)
        self.q.put(msg)

    def close(self):
        self.q.put(None)
        self.q.put(None)
        self.p.join()

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_tile_resize(im_list_2d):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

def visualize_occupancy_map(occupancy_data):
    dpi = 100
    fig = plt.figure(figsize=(1080 / dpi, 720 / dpi), dpi=dpi)
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    car_size = 1.0
    car_vertices = np.array([
        [-car_size*2, -car_size, 0],
        [car_size*2, -car_size, 0],
        [car_size*2, car_size, 0],
        [-car_size*2, car_size, 0],
        [-car_size*2, -car_size, car_size*1.5],
        [car_size*2, -car_size, car_size*1.5],
        [car_size*2, car_size, car_size*1.5],
        [-car_size*2, car_size, car_size*1.5]
    ])
    car_faces = [
        [car_vertices[0], car_vertices[1], car_vertices[2], car_vertices[3]],  # bottom
        [car_vertices[4], car_vertices[5], car_vertices[6], car_vertices[7]],  # top
        [car_vertices[0], car_vertices[1], car_vertices[5], car_vertices[4]],  # front
        [car_vertices[2], car_vertices[3], car_vertices[7], car_vertices[6]],  # back
        [car_vertices[0], car_vertices[3], car_vertices[7], car_vertices[4]],  # left
        [car_vertices[1], car_vertices[2], car_vertices[6], car_vertices[5]]   # right
    ]
    car_front_face = Poly3DCollection([car_faces[5]], alpha=0.7, facecolor='blue', edgecolor='black', linewidth=0.5)
    car_poly3d = Poly3DCollection(car_faces, alpha=0.7, facecolor='gray', edgecolor='black', linewidth=0.5)
    ax.add_collection3d(car_poly3d)
    ax.add_collection3d(car_front_face)
    car_poly3d_2 = Poly3DCollection(car_faces, alpha=0.7, facecolor='gray', edgecolor='black', linewidth=0.5)
    ax2.add_collection3d(car_poly3d_2)

    unique_labels = np.unique(occupancy_data[:, 3])
    
    for label in unique_labels:
        mask = occupancy_data[:, 3] == label
        points = occupancy_data[mask][:, :3].astype(float)
        
        vertex_offsets = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        
        for point in points:
            vertices = point + vertex_offsets * scale
            
            faces = [
                [vertices[0], vertices[1], vertices[4], vertices[2]],  # bottom
                [vertices[3], vertices[5], vertices[7], vertices[6]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[3]],  # front
                [vertices[2], vertices[4], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[2], vertices[6], vertices[3]],  # left
                [vertices[1], vertices[4], vertices[7], vertices[5]]   # right
            ]
            
            poly3d = Poly3DCollection(faces, alpha=0.7, facecolor=id2rgb[label], edgecolor='black', linewidth=0.5)
            ax.add_collection3d(poly3d)

            poly3d_2 = Poly3DCollection(faces, alpha=0.7, facecolor=id2rgb[label], edgecolor='black', linewidth=0.5)
            ax2.add_collection3d(poly3d_2)
    
    for ax in [ax, ax2]:
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])
        ax.set_zlim([0, 2])
        ax.set_box_aspect((8, 8, 2))
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Perspectives
    ax2.view_init(45, 180)

    fig.suptitle('Occupancy Map')
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=id2rgb[label], 
                                label=id2label[label]) 
                    for label in unique_labels]
    fig.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout(pad=1)
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return image_flat

def visualize_occupancy_map_voxels(occupancy_data):
    dpi = 100
    fig = plt.figure(figsize=(1080 / dpi, 720 / dpi), dpi=dpi)
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Convert point cloud to voxel grid
    grid_size = 16  # Number of voxels per dimension
    voxel_size = 16.0 / grid_size  # Total space is -8 to 8
    
    # Create voxel grid coordinates
    x = np.floor((occupancy_data[:, 0] + 8) / voxel_size).astype(int)
    y = np.floor((occupancy_data[:, 1] + 8) / voxel_size).astype(int)
    z = np.floor(occupancy_data[:, 2] / voxel_size).astype(int)
    labels = occupancy_data[:, 3]
    
    # Filter valid voxels
    mask = (x >= 0) & (x < grid_size) & (y >= 0) & (y < grid_size) & (z >= 0) & (z < grid_size)
    x, y, z, labels = x[mask], y[mask], z[mask], labels[mask]
    
    # Create coordinate meshgrid for plotting
    edge_coords = np.linspace(-8, 8, grid_size+1)
    z_coords = np.linspace(0, 2, grid_size+1)
    x_coords, y_coords, z_coords = np.meshgrid(
        edge_coords,
        edge_coords,
        z_coords,
        indexing='ij'
    )
    
    for ax_i in [ax, ax2]:
        # Draw voxels for each unique label
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            x_label = x[label_mask]
            y_label = y[label_mask]
            z_label = z[label_mask]
            
            # Create voxel grid
            voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
            voxel_grid[x_label, y_label, z_label] = True
            
            ax_i.voxels(x_coords, y_coords, z_coords, voxel_grid,
                       facecolors=id2rgb[label], alpha=0.7,
                       edgecolor='black', linewidth=0.5)
        
        # Draw car as voxels
        car_size = 1.0
        car_center = np.array([0, 0, car_size * 0.75])
        car_dims = np.array([car_size * 4, car_size * 2, car_size * 1.5])
        
        car_min = np.floor((car_center - car_dims/2 + 8) / voxel_size).astype(int)
        car_max = np.ceil((car_center + car_dims/2 + 8) / voxel_size).astype(int)
        
        car_voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        car_voxels[
            max(0, car_min[0]):min(grid_size, car_max[0]),
            max(0, car_min[1]):min(grid_size, car_max[1]),
            max(0, car_min[2]):min(grid_size, car_max[2])
        ] = True
        
        ax_i.voxels(x_coords, y_coords, z_coords, car_voxels,
                    facecolors='gray', alpha=0.7,
                    edgecolor='black', linewidth=0.5)
        
        # Set view properties
        ax_i.set_xlim([-8, 8])
        ax_i.set_ylim([-8, 8])
        ax_i.set_zlim([0, 2])
        ax_i.set_box_aspect((8, 8, 2))
        ax_i.set_aspect('equal')
        ax_i.set_xlabel('X')
        ax_i.set_ylabel('Y')
        ax_i.set_zlabel('Z')
    
    # Set different perspective for second plot
    ax2.view_init(45, 180)
    
    # Add title and legend
    fig.suptitle('Occupancy Map (Voxelized)')
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=id2rgb[label], 
                                   label=id2label[label]) 
                      for label in unique_labels]
    fig.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout(pad=1)
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(
        fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.savefig('obs_map.png')
    plt.close(fig)
    return image_flat

obs_dpi = 100
obs_fig = plt.figure(figsize=(810 / obs_dpi, 540 / obs_dpi), dpi=obs_dpi)
obs_fig.suptitle('Occupancy Grid')
obs_fig.tight_layout(pad=1)
obs_ax = obs_fig.add_subplot(111)
def visualize_obs(obs):
    obs_ax.clear()
    obs_ax.imshow(obs, cmap='gray', vmin=0.0, vmax=1.0)
    obs_fig.savefig('obs_map.png')
    obs_fig.canvas.draw()
    obs_img = np.frombuffer(obs_fig.canvas.buffer_rgba(), dtype='uint8').reshape(obs_fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    return obs_img

def visualizer_process_msg(msg):
    recording_img = msg.recording_img
    imgnps = msg.imgnps
    occupancy_data = msg.occupancy_data
    obs = msg.obs
    iou = msg.iou
    latency = msg.latency

    image = recording_img if recording_img is not None else np.zeros((720, 1080, 3), dtype=np.uint8)
    height = image.shape[0]
    width = image.shape[1]

    image = cv2.putText(
        image,
        "autonomous, 3x speed",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2
    )
    image = cv2.putText(
        image,
        "IOU: {:.2f}".format(iou),
        (width - 175, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2
    )
    image = cv2.putText(
        image,
        "latency: {}ms".format(latency),
        (width - 275, height - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2
    )

    if imgnps is None:
        imgnps = [
            np.zeros((640, 808, 3), dtype=np.uint8),
            np.zeros((640, 808, 3), dtype=np.uint8),
            np.zeros((640, 808, 3), dtype=np.uint8),
            np.zeros((640, 808, 3), dtype=np.uint8)
        ]
    occupancy_map_img = None
    if occupancy_data is None:
        occupancy_map_img = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        occupancy_map_img = np.zeros((height, width, 3), dtype=np.uint8)
        # occupancy_map_img = visualize_occupancy_map(occupancy_data)

    obs_img = None
    if obs is None:
        obs_img = np.zeros((540, 810, 3), dtype=np.uint8)
    else:
        obs_img = visualize_obs(obs)

    image = concat_tile_resize([
        # [image, occupancy_map_img],
        [image, obs_img],
        [imgnps[0], imgnps[1], imgnps[2], imgnps[3]]
    ])

    return image

def visualizer(q):
    try:
        recording_file = iio.imopen('./vis.mp4', 'w', plugin='pyav')
        recording_file.init_video_stream('vp9', fps=30)
        msg = q.get()
        while msg is not None:
            recording_file.write_frame(visualizer_process_msg(msg), pixel_format='rgb24')
            msg = q.get()
        # batch = []

        # def process_batch():
        #     was_last = False
        #     while len(batch) < 10:
        #         msg = q.get()
        #         if msg is None:
        #             was_last = True
        #             break
        #         batch.append(msg)
        #     if not batch: return was_last
        #     with ProcessPoolExecutor(max_workers=10) as executor:
        #         futures = [executor.submit(visualizer_process_msg, msg) for msg in batch]
        #         for future in futures:
        #             recording_file.write_frame(future.result(), pixel_format='rgb24')
        #     batch.clear()
        #     return was_last

        # while not process_batch(): pass
    except KeyboardInterrupt:
        print('stopping visualizer')
    finally:
        msg = q.get()
        while msg is not None:
            print('flushing remaining items:', q.qsize())
            recording_file.write_frame(visualizer_process_msg(msg), pixel_format='rgb24')
            msg = q.get()
        recording_file.close()
