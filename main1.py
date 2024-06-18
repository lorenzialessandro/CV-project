import numpy as np
import c3d
import sys

from utils.utility import * # utilities functions: see utility.py
import lib.optitrack.csv_reader as csv # optitrack 
from lib.optitrack.geometry import * # optitrack
import lib.BVH_reader.BVH_FILE as bhv


# Define bones dictionary
bones_map_skeleton = {
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
# Define markers dictionary
marker_map_ragnetto = {
    'Marker1' : ['Marker2'],
    'Marker2' : ['Marker4'],
    'Marker3' : ['Marker1'],
    'Marker4' : ['Marker3']
}

def main(file_type):
    
    # set fps source
    fps = 360
    # set perspective for 3D matplotlib visualization (elev, azim, roll)
    rot = (60, 20, 110)

    if file_type == "CSV_SKELETON":
        # Handle CSV file with skeleton data
        filename = "resources/"+str(fps)+"fps/skeleton.csv"
        x, y, z, component_names = read_csv(filename)
        n_markers = x.shape[0]
        n_frames = x.shape[-1]

        #create_animation(x, y, z, lines_map, 1000, n_markers, filename='ragnetto-PF.gif')
        create_single_plot(x, y, z, component_names, bones_map_skeleton, n_frames, n_markers, rot)
        
    elif file_type == "CSV_RIGID":
        # Handle CSV file with rigid body data
        filename = "resources/"+str(fps)+"fps/rigidbody.csv"
        x, y, z, component_names = read_csv(filename)
        n_markers = x.shape[0]
        n_frames = x.shape[-1]

        #create_animation(x, y, z, lines_map, 1000, n_markers, filename='ragnetto-PF.gif')
        create_single_plot(x, y, z, component_names, marker_map_ragnetto, n_frames, n_markers, rot)

    elif file_type == "BVH":
        # Handle BVH file
        filename = "resources/"+str(fps)+"fps/animation.bvh"
        rotations, positions, edges, offsets, joint_names = read_bvh(filename)
        
        print_bvh_info(rotations, positions, edges, offsets, joint_names) # print bvh info on prompt 
        #write_bvh_info_to_txt(rotations, positions, edges, offsets, joint_names, 'bvh_info.txt') # write bvh info on .txt file
        
    elif file_type == "C3D":
        # Handle C3D file
        filename = "resources/"+str(fps)+"fps/marker.c3d"
        frames_data, labels = read_c3d(filename)

        print_c3d_info(frames_data, labels) # print c3d info on prompt 
        #write_c3d_info_to_txt(frames_data, labels, 'c3d_info.txt') # write c3d info on .txt file
        
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

    # Print out file content
    print("Rigid bodies : ", take.rigid_bodies.keys())
    print("Skeletons : ", take.skeletons.keys())

    # Process the first rigid body into a set of planes.
    bodies = take.rigid_bodies
    skeletons = take.skeletons

    ragnetto_pos = []
    rigid_body_markers_pos = []
    markers_pos = []
    
    if len(bodies) > 0:
        for body in bodies: 
            ragnetto = take.rigid_bodies[body]
            n_frames = ragnetto.num_total_frames()
            
            ragnetto_pos.append(ragnetto.positions)
            for marker in ragnetto.rigid_body_markers.values():
                rigid_body_markers_pos.append(marker.positions)
                
            for marker in ragnetto.markers.values():
                markers_pos.append(marker.positions)

            component_names = list(ragnetto.markers)            
            points = rigid_body_markers_pos

    bone_markers = []
    bones=[]
    
    if len(skeletons) > 0:
        for body in skeletons: 
            skeleton = take.skeletons[body]
            n_frames = skeleton.bones["Hip"].num_total_frames()
            
            for marker in skeleton.bones.values():
                bones.append(marker.positions)
                
            for marker in skeleton.bone_markers.values():
                bone_markers.append(marker.positions)

            component_names = list(skeleton.bones)            
            points = bones

    points = [[[np.nan, np.nan, np.nan] if frame is None else frame for frame in markers] for markers in points]
    np_points = np.array(points)

    # np_points shape is : (rigid body, frame, axis)
    x = np_points[:,:,0]
    y = np_points[:,:,1]
    z = np_points[:,:,2]

    return x,y,z, component_names



# Function to read data from a .bvh file.
def read_bvh(filename):
    animation, joint_names, _ = bhv.read_bvh(filename) # Call lib function

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
    nargs = len(sys.argv)

    if nargs != 2 :
        print("Arguments error")
        print("Expected arguments: 1 mandatory")
        print("     Arg1 : { CSV_SKELETON | CSV_RIGID | BVH | C3D }")
        sys.exit(1)
        
    file_type = sys.argv[1]
            
    main(file_type)