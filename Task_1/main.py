import numpy as np
import c3d
import sys
import cv2

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
        #plot_single_animation_open3d(x, y, z, lines_map)
        create_single_plot(x,y,z, lines_map, n_frames)
        #plot_single_animation_open3d_new(x, y, z, lines_map)
        
    elif file_type == "CSV_RIGID":
        # Handle CSV file with rigid body data
        filename = "../material/60fps/rigidbody.csv"
        x,y,z, lines_map, n_frames = read_csv(filename)
        
        x,y,z = apply_Kallman(x, y, z, n_frames, 4)
        
        #create_plots(x,y,z, lines_map, n_frames)
        create_single_plot(x,y,z, lines_map, n_frames)
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

def concat_diagonally(L):
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype=np.float32)
    out[mask] = np.concatenate(L).ravel()
    return out

def apply_Kallman(x, y, z, n_frames, n_markers) :
    
    kalman = cv2.KalmanFilter(7*n_markers, 3*n_markers) # (d,n_params)      
    
    measurementMatrix = []
    transitionMatrix = []
    processNoiseCov = []
    measurementNoiseCov = []
    
    for _ in range(n_markers) :
        # maps the state vector to the observed measurements
        measurementMatrix_i = \
        np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0]],np.float32)
        
        # maps the current state vector to its next state 
        # vector based on the defined motion model
        #
        # Defines "Legge oraria (?)"
        t = 1/60
        transitionMatrix_i = \
            np.array([
                [1, 0, 0, t, 0, 0, 0],
                [0, 1, 0, 0, t, 0, 0],
                [0, 0, 1, 0, 0, t, 0],
                [0, 0, 0, 1, 0, 0, t],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]], np.float32)
        
        # Models the uncertainty in the motion model
        processNoiseCov_i = \
            np.array([
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]], np.float32) * 0.003
        
        # Models the uncertainty of mesurements themselves
        measurementNoiseCov_i = \
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]], np.float32) * 1
        
        measurementMatrix.append(measurementMatrix_i)
        transitionMatrix.append(transitionMatrix_i)
        processNoiseCov.append(processNoiseCov_i)
        measurementNoiseCov.append(measurementNoiseCov_i)
        
    print("measurementMatrix : ", concat_diagonally(tuple(measurementMatrix)).shape)
    print("transitionMatrix : ", concat_diagonally(tuple(transitionMatrix)).shape)
    print("processNoiseCov : ", concat_diagonally(tuple(processNoiseCov)).shape)
    print("measurementNoiseCov : ", concat_diagonally(tuple(measurementNoiseCov)).shape)
    
    kalman.measurementMatrix = concat_diagonally(tuple(measurementMatrix))
    kalman.transitionMatrix = concat_diagonally(tuple(transitionMatrix))
    kalman.processNoiseCov = concat_diagonally(tuple(processNoiseCov))
    kalman.measurementNoiseCov = concat_diagonally(tuple(measurementNoiseCov))
    
    
    for t in range(0, n_frames):
        
        point = np.array([x[:,t], y[:,t], z[:,t]], np.float32).T # (n_markers x 3)
        point = np.array([point.flatten()]).T
        
        kalman.correct(point)
        prediction = kalman.predict()

        #print(prediction)

    return x, y, z

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

            lines_map = {}

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

            lines_map = process_skeleton(skeleton)
            
            points = bones

    points = [[[np.nan, np.nan, np.nan] if frame is None else frame for frame in markers] for markers in points]
    np_points = np.array(points)

    # np_points shape is : (rigid body, frame, axis)
    x = np_points[:,:,0]
    y = np_points[:,:,1]
    z = np_points[:,:,2]

    return x,y,z, lines_map, n_frames



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
    if len(sys.argv) != 2:
        print("argv error")
        print("Usage: python script.py CSV_SKELETON | CSV_RIGID | BVH | C3D")
        sys.exit(1)
    
    file_type = sys.argv[1]
    main(file_type)