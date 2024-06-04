import json
import cv2
import numpy as np

from utility import * # utilities functions: see utility.py

# Define bones dictionary + list
guy_bones = []
guy_bones_map = {
    "Hips": ["LeftUpLeg", "RightUpLeg", "Spine"],
    "LeftUpLeg": ["LeftLeg"],
    "LeftLeg": ["LeftFoot"],
    "LeftFoot": ["LeftToeBase"],
    "LeftToeBase": ["LeftToe_End"],
    "LeftToe_End": [],

    "RightUpLeg": ["RightLeg"],
    "RightLeg": ["RightFoot"],
    "RightFoot": ["RightToeBase"],
    "RightToeBase": ["RightToe_End"],
    "RightToe_End": [],

    "Spine": ["Spine1"],
    "Spine1": ["Spine2"],
    "Spine2": ["LeftShoulder", "Neck", "RightShoulder"],
    "LeftShoulder": ["LeftArm"],
    "LeftArm": ["LeftForeArm"],
    "LeftForeArm": ["LeftHand"],
    "LeftHand": ["LeftHandIndex1", "LeftHandMiddle1", "LeftHandPinky1", "LeftHandRing1", "LeftHandThumb1"],
    "LeftHandIndex1": ["LeftHandIndex2"],
    "LeftHandIndex2": ["LeftHandIndex3"],
    "LeftHandIndex3": ["LeftHandIndex4"],
    "LeftHandIndex4": [],

    "LeftHandMiddle1": ["LeftHandMiddle2"],
    "LeftHandMiddle2": ["LeftHandMiddle3"],
    "LeftHandMiddle3": ["LeftHandMiddle4"],
    "LeftHandMiddle4": [],

    "LeftHandPinky1": ["LeftHandPinky2"],
    "LeftHandPinky2": ["LeftHandPinky3"],
    "LeftHandPinky3": ["LeftHandPinky4"],
    "LeftHandPinky4": [],

    "LeftHandRing1": ["LeftHandRing2"],
    "LeftHandRing2": ["LeftHandRing3"],
    "LeftHandRing3": ["LeftHandRing4"],
    "LeftHandRing4": [],

    "LeftHandThumb1": ["LeftHandThumb2"],
    "LeftHandThumb2": ["LeftHandThumb3"],
    "LeftHandThumb3": ["LeftHandThumb4"],
    "LeftHandThumb4": [],

    "Neck": ["Neck1"],
    "Neck1": ["Head"],
    "Head": ["HeadTop_End", "LeftEye", "RightEye"],
    "HeadTop_End": [],
    "LeftEye": [],
    "RightEye": [],

    "RightShoulder": ["RightArm"],
    "RightArm": ["RightForeArm"],
    "RightForeArm": ["RightHand"],
    "RightHand": ["RightHandIndex1", "RightHandMiddle1", "RightHandPinky1", "RightHandRing1", "RightHandThumb1"],
    "RightHandIndex1": ["RightHandIndex2"],
    "RightHandIndex2": ["RightHandIndex3"],
    "RightHandIndex3": ["RightHandIndex4"],
    "RightHandIndex4": [],

    "RightHandMiddle1": ["RightHandMiddle2"],
    "RightHandMiddle2": ["RightHandMiddle3"],
    "RightHandMiddle3": ["RightHandMiddle4"],
    "RightHandMiddle4": [],

    "RightHandPinky1": ["RightHandPinky2"],
    "RightHandPinky2": ["RightHandPinky3"],
    "RightHandPinky3": ["RightHandPinky4"],
    "RightHandPinky4": [],

    "RightHandRing1": ["RightHandRing2"],
    "RightHandRing2": ["RightHandRing3"],
    "RightHandRing3": ["RightHandRing4"],
    "RightHandRing4": [],

    "RightHandThumb1": ["RightHandThumb2"], 
    "RightHandThumb2": ["RightHandThumb3"],
    "RightHandThumb3": ["RightHandThumb4"],
    "RightHandThumb4": []
}


# Function to process a skeleton
def get_skeleton_lines(bones, bones_map):
    """
    function to process a skeleton

    :param skeleton: skeleton
    :return: lines_map
    """
    # Map bone names to indices
    bones_index_map = {bone: bones.index(bone) for bone in bones}
    # Map bone indices to their connections
    lines_map = {bones_index_map[bone]: [bones_index_map[child] for child in children] for bone, children in bones_map.items()}

    return lines_map


if __name__ == "__main__":
    # Opening JSON file
    src_guy = open('guy.json')
    src_cam = open('camera.json')
    # Returns JSON object as a dictionary
    data_guy = json.load(src_guy)
    data_cam = json.load(src_cam)
    # Close files
    src_guy.close()
    src_cam.close()

    # Camera vals
    tvec_cam = np.array(data_cam["Camera_pos"], dtype=np.float32)
    rvec_cam = np.array(data_cam["Camera_rot"], dtype=np.float32)
    fov = np.float32(data_cam["Camera_FOV"])
    cam_ar =  np.float32(data_cam["Camera_AspectRatio"])

    # Skeleton vals
    x = [[pos["X"] for pos in val["Positions"]] for val in data_guy["Body"]]
    y = [[pos["Y"] for pos in val["Positions"]] for val in data_guy["Body"]]
    z = [[pos["Z"] for pos in val["Positions"]] for val in data_guy["Body"]]
    x = np.array(x).T
    y = np.array(y).T
    z = np.array(z).T

    #n_markers = x.shape[0]
    #n_frames = x.shape[1]
    #guy_bones = [bone["name"] for bone in data_guy["Body"][0]["Positions"]]
    #rot = (15, 45, 0)
    #create_single_plot(x,y,z, guy_bones, guy_bones_map, n_frames, n_markers, rot)

    t = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        objectPoints = np.array((x[:,t],y[:,t],z[:,t]), dtype=np.float32).T

        imagePoints, _ = cv2.projectPoints(objectPoints, rvec_cam, tvec_cam, cameraMatrix, None, aspectRatio=cam_ar)

        imagePoints = imagePoints.squeeze().astype(int)
        for point in imagePoints:
            cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)

        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break        

        print(f"Frame {t}: {imagePoints}")
        if ret == False:
            break

        t = t+1

    cap.release()
    cv2.destroyAllWindows()