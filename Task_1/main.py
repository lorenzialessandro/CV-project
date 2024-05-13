import numpy as np
import c3d
import sys

from utility import * # utilities functions: see utility.py
import lib.optitrack.csv_reader as csv # optitrack 
from lib.optitrack.geometry import * # optitrack
import lib.BVH_reader.BVH_FILE as bhv

def main(file_type):
    if file_type == "CSV_SKELETON":
        # Handle CSV file with skeleton data
        filename = "../material/60fps/skeleton.csv"
        x,y,z, lines_map, n_frames = read_csv(filename)

        #create_plots(x,y,z, lines_map, n_frames)
        #create_animation(x, y, z, lines_map, n_frames, filename='skeleton.gif')
        plot_single_animation_open3d(x, y, z, lines_map)
        #plot_single_animation_open3d_new(x, y, z, lines_map)
        
    elif file_type == "CSV_RIGID":
        # Handle CSV file with rigid body data
        filename = "../material/60fps/rigidbody.csv"
        x,y,z, lines_map, n_frames = read_csv(filename)
        
        create_plots(x,y,z, lines_map, n_frames)
        #create_animation(x, y, z, lines_map, n_frames, filename='rigidbody.gif')
        #plot_single_animation_open3d(x, y, z, lines_map)
        

    elif file_type == "BVH":
        # Handle BVH file
        filename = "../material/60fps/animation.bvh"
        rotations, positions, edges, offsets, joint_names = read_bvh(filename)
        
        # print_bvh_info(rotations, positions, edges, offsets, joint_names) # print bvh info on prompt 
        # write_bvh_info_to_txt(rotations, positions, edges, offsets, joint_names, 'bvh_info.txt') # write bvh info on .txt file
        
    elif file_type == "C3D":
        # Handle C3D file
        filename = "../material/60fps/marker.c3d"
        frames_data, labels = read_c3d(filename)

        # print_c3d_info(frames_data, labels) # print c3d info on prompt 
        # write_c3d_info_to_txt(frames_data, labels, 'c3d_info.txt') # write c3d info on .txt file
        
    else:
        print("Invalid file type.")


# Function to read data from a .csv file.
def read_csv(filename):
    """
    Function to read data from a .csv file

    :param filename: The name of the .csv file.
    :return: x, y, z points (coordinates), lines_map (map of skeleton lines), n_frames (total number of frames).
    """

    # Read the file.
    take = csv.Take().readCSV(filename)

    # Print out some statistics
    # print("Found rigid bodies:", take.rigid_bodies.keys())

    # Process the first rigid body into a set of planes.
    bodies = take.rigid_bodies
    skeletons = take.skeletons

    bones_pos = []
    rigid_body_markers_pos = []
    markers_pos = []

    if len(bodies) > 0:
        for body in bodies: 
            bones = take.rigid_bodies[body]
            n_frames = bones.num_total_frames()
            
            bones_pos.append(bones.positions)
            for marker in bones.rigid_body_markers.values():
                rigid_body_markers_pos.append(marker.positions)
                
            for marker in bones.rigid_body_markers.values():
                markers_pos.append(marker.positions)

            lines_map = {}

    if len(skeletons) > 0:
        for body in skeletons: 
            skeleton = take.skeletons[body]
            n_frames = skeleton.bones["Hip"].num_total_frames()
            
            for marker in skeleton.bones.values():
                rigid_body_markers_pos.append(marker.positions)
                
            for marker in skeleton.bone_markers.values():
                markers_pos.append(marker.positions)

            lines_map = process_skeleton(skeleton)
            

    #points = bones_pos + rigid_body_markers_pos + markers_pos
    points = rigid_body_markers_pos

    #points = [[0.0,0.0,0.0] if elem is None else elem for elem in markers for markers in points ]
    points = [[[0.0, 0.0, 0.0] if frame is None else frame for frame in markers] for markers in points]
    np_points = np.array(points)

    x = np_points[:,:,0]
    y = np_points[:,:,1]
    z = np_points[:,:,2]

    return x,y,z, lines_map, n_frames



# Function to read data from a .bvh file.
def read_bvh(filename):
    animation, joint_names, frametime = bhv.read_bvh(filename) # Call lib function

    # Relative rotation for each joint
    rotations = animation.rotations

    # Global Position for each joint
    positions = animation.positions

    # Edges (list of pairs of joints that are connected)
    edges = [(child, parent) for child, parent in enumerate(animation.parents) if parent >= 0]

    # Offsets (define the length of each bone)
    offsets = animation.offsets

    return rotations, positions, edges, offsets, joint_names
    
  

# Function to read data from a .c3d file.
def read_c3d(filename):
    """
    Function to read data from a .c3d file.

    :param filename: The path to the .c3d file.
    :return: A tuple containing frames_data (a list of dictionaries containing marker information for each frame) and labels (a list of marker labels).
    """
    # Lists to store marker information
    frames_data = []
    labels = []

    reader = c3d.Reader(open(filename, 'rb'))
    for i, points, analog in reader.read_frames():
        # Store frame number
        frame_info = {'frame_number': i}
        
        # Store marker coordinates
        marker_info = {}
        for j, label in enumerate(reader.point_labels):
            marker_info[label] = points[j]
        
        frame_info['marker_coordinates'] = marker_info
        
        # Store analog data (if any)
        frame_info['analog_data'] = analog
        
        # Append frame data to the list
        frames_data.append(frame_info)
        
        # Store labels
        if i == 0:  # Assuming labels are same for all frames
            labels = reader.point_labels     
    
    return frames_data, labels



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_type>")
        sys.exit(1)
    
    file_type = sys.argv[1]
    main(file_type)