from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# Function to process a skeleton
def process_skeleton(skeleton):
    """
    function to process a skeleton

    :param skeleton: skeleton
    :return: lines_map
    """
    # Define bones dictionary
    bones = list(skeleton.bones)
    bones_map = {
        "Head": ["Neck"],
        "Neck": ["RShoulder", "LShoulder", "Chest"],
        "RShoulder": ["RUArm"],
        "RUArm": ["RFArm"],
        "RFArm": ["RHand"],
        "RHand": [],
        "LShoulder": ["LUArm"],
        "LUArm": ["LFArm"],
        "LFArm": ["LHand"],
        "LHand": [],
        "Chest": ["Ab"],
        "Ab": ["Hip"],
        "Hip": ["LThigh", "RThigh"],
        "RThigh": ["RShin"],
        "RShin": ["RFoot"],
        "RFoot": ["RToe"],
        "RToe": [],
        "LThigh": ["LShin"],
        "LShin": ["LFoot"],
        "LFoot": ["LToe"],
        "LToe": []
    }

    # Map bone names to indices
    bones_index_map = {bone: bones.index(bone) for bone in bones}

    # Map bone indices to their connections
    lines_map = {bones_index_map[bone]: [bones_index_map[child] for child in children] for bone, children in bones_map.items()}

    return lines_map


# Function to display animation in a single subplot
def plot_single_animation(ax, x, y, z, lines_map, t, n_markers):
    """
    Function to display animation in a single subplot.

    :param ax: Subplot axes
    :param x: x points
    :param y: y points
    :param z: z points
    :param lines_map: Mapping of lines
    :param t: Current frame
    """
    
    ax.clear()
    
    x_noNaNs = x[~np.isnan(x)]
    y_noNaNs = y[~np.isnan(y)]
    z_noNaNs = z[~np.isnan(z)]
    ax.set_xlim(np.min(x_noNaNs), np.max(x_noNaNs))
    ax.set_ylim(np.min(y_noNaNs), np.max(y_noNaNs))
    ax.set_zlim(np.min(z_noNaNs), np.max(z_noNaNs))
    
    # scatter markers
    ax.scatter(x[:n_markers, t][~np.isnan(x[:n_markers, t])], y[:n_markers, t][~np.isnan(y[:n_markers, t])], z[:n_markers, t][~np.isnan(z[:n_markers, t])], color = "blue")
    # scatter particles (if there's any)
    if x.shape[0] > n_markers :
        ax.scatter(x[n_markers:, t][~np.isnan(x[n_markers:, t])], y[n_markers:, t][~np.isnan(y[n_markers:, t])], z[n_markers:, t][~np.isnan(z[n_markers:, t])], s=0.4, color="red")
    
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    # Plot connections
    if len(lines_map) > 0:
        for i in range(len(x[:, t])):
            for id in lines_map[i]:
                ax.plot([x[i, t], x[id, t]], [y[i, t], y[id, t]], [z[i, t], z[id, t]], "--", color="blue")

    # Plot trajectories
    trajectory_frames = 180  # interpolate previous "trajectory_frames" points
    tf = max(0, t - trajectory_frames)

    for i in range(n_markers):
        ax.plot(x[i, tf:t], y[i, tf:t], z[i, tf:t], color="red", alpha=0.2)


# Wrapper function to create and display multiple subplots
def create_plots(x, y, z, lines_map, n_frames):
    """
    Wrapper function to create and display multiple subplots.

    :param x: x points
    :param y: y points
    :param z: z points
    :param lines_map: Mapping of lines
    :param n_frames: Total number of frames
    """
    # Create a figure and axes for the plots
    fig = plt.figure(figsize=(15, 8))  # Adjust the figure size as needed

    # Use GridSpec to customize subplot layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    # List to store subplot axes
    axes = []

    # Iterate over subplots
    for i in range(3):
        ax = fig.add_subplot(gs[i // 3, i % 3], projection='3d')
        ax.view_init(elev=[90.001, 0, 60][i], azim=[0, 0, 20][i], roll=[90, 90, 110][i])
        axes.append(ax)

    # Adjust margins and spacing between subplots
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)

    # Iterate over frames and update plot for each subplot
    for t in range(n_frames):
        for ax in axes:
            plot_single_animation(ax, x, y, z, lines_map, t)
        plt.pause(0.1)  # Adjust the pause time as needed

    plt.show()

def create_single_plot(x, y, z, lines_map, n_frames, n_markers) :
    """
    Wrapper function to create and display multiple subplots.

    :param x: x points
    :param y: y points
    :param z: z points
    :param lines_map: Mapping of lines
    :param n_frames: Total number of frames
    """

    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(elev=60, azim=20, roll=110)

    # Iterate over frames and update plot for each subplot
    for t in range(n_frames):
        plot_single_animation(ax, x, y, z, lines_map, t, n_markers)
        plt.pause(0.01)  # Adjust the pause time as needed

    plt.show()
    


# Function to create and save an animation as GIF
def create_animation(x, y, z, lines_map, n_frames, n_markers, filename='animation.gif'):
    """
    Function to create and save an animation as GIF.

    :param x: x points
    :param y: y points
    :param z: z points
    :param lines_map: Mapping of lines
    :param n_frames: Total number of frames
    :param filename: Name of the output GIF file
    """
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(60,20,110)

    def update(frame):
        ax.clear()
        plot_single_animation(ax, x, y, z, lines_map, frame, n_markers)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100)
    anim.save("animations/"+filename, writer='pillow', fps=60)


# Function to display animation in a single subplot using Open3D
def plot_single_animation_open3d(x, y, z, lines_map):
    """
    Function to display animation in a single subplot using Open3D.

    :param x: x points (numpy array)
    :param y: y points (numpy array)
    :param z: z points (numpy array)
    :param lines_map: Mapping of lines (dict)
    """
    # Create a point cloud
    point_cloud = o3d.geometry.PointCloud()
    
    # Plot connections
    line_set = o3d.geometry.LineSet()
    
    # Plot trajectories
    trajectory_frames = 180  # displace the previous "trajectory_frames" as an interpolation

    # Create visualizer and window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 7  # Adjust the size as needed

    for t in range(x.shape[1]):
        # Update point cloud
        point_cloud.points = o3d.utility.Vector3dVector(np.column_stack((x[:, t], y[:, t], z[:, t])))
        
        # Update connections if lines_map is provided and not empty
        if lines_map and any(lines_map.values()):
            lines = []
            for i in range(len(x[:, t])):
                for id in lines_map[i]:
                    lines.append([i, id])
            line_set.points = o3d.utility.Vector3dVector(point_cloud.points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.0, 0.0, 1.0])  # Adjust color as needed
        else:
            # Clear LineSet if lines_map is empty or contains no lines
            line_set.clear()

        # Update trajectories
        trajectories = []
        tf = max(0, t - trajectory_frames)
        for i in range(len(x)):
            trajectory = o3d.geometry.LineSet()
            trajectory.points = o3d.utility.Vector3dVector(np.column_stack((x[i, tf:t], y[i, tf:t], z[i, tf:t])))
            trajectory.lines = o3d.utility.Vector2iVector([[k, k+1] for k in range(len(x[i, tf:t])-1)])
            trajectory.paint_uniform_color([1.0, 0.0, 0.0])  # Adjust color as needed
            trajectories.append(trajectory)
        
        # Clear geometries
        vis.clear_geometries()
        
        # Add geometries to visualizer
        vis.add_geometry(point_cloud)
        vis.add_geometry(line_set)
        for traj in trajectories:
            vis.add_geometry(traj)
        
        # Update visualizer
        vis.poll_events()
        vis.update_renderer()
    # Destroy window
    vis.destroy_window()



def print_c3d_info(frames_data, labels):
    """
    Function to print the marker information from frames_data.

    :param frames_data: A list of dictionaries containing marker information for each frame.
    :param labels: A list of marker labels.
    :return: None
    """
    for frame_info in frames_data:
        print(f"Frame Number: {frame_info['frame_number']}")
        for label, coordinates in frame_info['marker_coordinates'].items():
            print(f"{label} - Coordinates: {coordinates}")
        if frame_info['analog_data']:
            print("Analog Data:", frame_info['analog_data'])


def write_c3d_info_to_txt(frames_data, labels, output_file):
    """
    Function to write marker information from frames_data to a text file.

    :param frames_data: A list of dictionaries containing marker information for each frame.
    :param labels: A list of marker labels.
    :param output_file: The path to the output text file.
    :return: None
    """
    print("writing c3d information...")
    with open(output_file, 'w') as txtfile:
        # Write header
        txtfile.write("Frame Number\t")
        txtfile.write("\t".join(labels))
        txtfile.write("\tAnalog Data\n")
        
        # Write data
        for frame_info in frames_data:
            txtfile.write(str(frame_info['frame_number']) + "\t")
            for label in labels:
                if label in frame_info['marker_coordinates']:
                    txtfile.write(str(frame_info['marker_coordinates'][label]) + "\t")
                else:
                    txtfile.write("\t")  # If marker not present in this frame
            txtfile.write(str(frame_info['analog_data']) + "\n")
    print("done!")


def print_bvh_info(rotations, positions, edges, offsets, joint_names):
    print("Joint Names:")
    print(joint_names)

    print("\nRelative Rotations for Each Joint:")
    for i, rotation in enumerate(rotations):
        print(f"Joint {i}: {rotation}")

    print("\nGlobal Positions for Each Joint:")
    for i, position in enumerate(positions):
        print(f"Joint {i}: {position}")

    print("\nEdges (List of Pairs of Joints that are Connected):")
    for edge in edges:
        print(edge)

    print("\nOffsets (Length of Each Bone):")
    for i, offset in enumerate(offsets):
        print(f"Joint {i}: {offset}")


def write_bvh_info_to_txt(rotations, positions, edges, offsets, joint_names, output_file):
    print("writing bvh information...")
    with open(output_file, 'w') as file:
        file.write("Joint Names:\n")
        file.write('\n'.join(joint_names))
        
        file.write("\n\nRelative Rotations for Each Joint:\n")
        for i, rotation in enumerate(rotations):
            file.write(f"Joint {i}: {rotation}\n")
        
        file.write("\nGlobal Positions for Each Joint:\n")
        for i, position in enumerate(positions):
            file.write(f"Joint {i}: {position}\n")
        
        file.write("\nEdges (List of Pairs of Joints that are Connected):\n")
        for edge in edges:
            file.write(f"{edge}\n")
        
        file.write("\nOffsets (Length of Each Bone):\n")
        for i, offset in enumerate(offsets):
            file.write(f"Joint {i}: {offset}\n")
    print("done!")

