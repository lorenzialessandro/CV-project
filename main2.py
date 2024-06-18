import numpy as np
import sys

from utils.utility import * # utilities functions: see utility.py
import lib.optitrack.csv_reader as csv # optitrack 
from lib.optitrack.geometry import * # optitrack

from utils.KF_utils import apply_KallmanFilter
from utils.PF_utils import apply_ParticleFilter


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

def main(file_type, filter_type):
    
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
        
        if filter_type is not None :
            if filter_type == "KF" :
                x, y, z = apply_KallmanFilter(x, y, z, n_frames, n_markers, fps)
            elif filter_type == "PF" :
                x, y, z = apply_ParticleFilter(x, y, z, n_frames, n_markers)

        #create_animation(x, y, z, lines_map, 1000, n_markers, filename='ragnetto-PF.gif')
        create_single_plot(x, y, z, component_names, bones_map_skeleton, n_frames, n_markers, rot)
        
    elif file_type == "CSV_RIGID":
        # Handle CSV file with rigid body data
        filename = "resources/"+str(fps)+"fps/rigidbody.csv"
        x, y, z, component_names = read_csv(filename)
        n_markers = x.shape[0]
        n_frames = x.shape[-1]

        if filter_type is not None :
            if filter_type == "KF" :
                x, y, z = apply_KallmanFilter(x, y, z, n_frames, n_markers, fps)
            elif filter_type == "PF" :
                x, y, z = apply_ParticleFilter(x, y, z, n_frames, n_markers)

        #create_animation(x, y, z, lines_map, 1000, n_markers, filename='ragnetto-PF.gif')
        create_single_plot(x, y, z, component_names, marker_map_ragnetto, n_frames, n_markers, rot)

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


if __name__ == "__main__":
    nargs = len(sys.argv)
    if nargs < 2 or nargs > 3 :
        print("Arguments error")
        print("Expected argument 1 mandatory + 1 optional")
        print("     Arg1 : { CSV_SKELETON | CSV_RIGID }")
        print("     Arg2 : { KF , PF }")
        sys.exit(1)
        
    file_type = sys.argv[1]
    filter_type = None
    
    if nargs == 3 :
        filter_type = sys.argv[2]
        
    main(file_type, filter_type)