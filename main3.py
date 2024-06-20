import json
import cv2
import numpy as np
from numpy.linalg import inv
from utils.cameraCalibSquares import *
import time
import scipy
from scipy.spatial.transform import Rotation

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

def projectionError(imagePoints, groundTruth, frame):
    """
    :param imagePoints: computed 2d camera coordinates of joints
    :param groundTruth: true values 2d camera coordinates of joints
    """
    error = np.mean(groundTruth - imagePoints, axis=0)
    print("Average projection error")
    print(f"    frame : {frame}")
    print(f"        X : {error[0]}")
    print(f"        Y : {error[1]}\n")


def rtvec_to_matrix(rvec, tvec):
    "Convert rotation vector and translation vector to 4x4 matrix"
    T = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    return T


def matrix_to_rtvec(matrix):
    "Convert 4x4 matrix to rotation vector and translation vector"
    rvec, _ = cv2.Rodrigues(matrix[:3, :3])
    tvec = matrix[:3, 3]
    return rvec.squeeze(), tvec


def leftToRightHanded (quat, tvec):
    # Left handed (Unreal Enginge) :  (+X: forward, +Y: right, +Z: up) 
    # Right handed (openCV) : (+X: right, +Y: down, +Z: forward)
    rvec = Rotation.from_quat(quat).as_rotvec()
    mat = rtvec_to_matrix(rvec, tvec)

    C = np.array([
        [0,  1,   0, 0],
        [0,  0,  -1, 0],
        [1,  0,   0, 0],
        [0,  0,   0, 1]
    ], dtype=np.float32)
    
    # Apply transformation
    mat = C @ mat @ inv(C)
    
    rvec, tvec = matrix_to_rtvec(mat)
    quat = Rotation.from_rotvec(rvec).as_quat()

    return quat, tvec


def worldToPixel(objectPoints, rvec_obj, tvec_obj, rvec_cam, tvec_cam, cameraMatrix, cameraDistortion):
    """
    :param objectPoints: object's point in world frame coordinates
    :param rvec_obj: rotation vector world frame -> object frame
    :param tvec_obj: translation vector world frame -> object frame
    :param rvec_cam: rotation vector world frame -> camera frame
    :param tvec_cam: translation vector world frame -> camera frame
    :param cameraMatrix: intrinsic camera parameters
    :param cameraMatrix: camera distortion parameters
    """
    T_cam_world = rtvec_to_matrix(rvec_cam, tvec_cam)   # camera to world transform
    rvec, tvec = matrix_to_rtvec(inv(T_cam_world))      # get (world -> cam) traslation & rotation vector

    # Obtain pixel convertion
    imagePoints, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, cameraDistortion)
    imagePoints = imagePoints.squeeze().astype(int)

    return imagePoints


if __name__ == "__main__":
    # Opening JSON file
    src_guy = open('resources/ue5/skeleton.json')
    src_cam = open('resources/ue5/camera.json')
    # Returns JSON object as a dictionary
    data_guy = json.load(src_guy)
    data_cam = json.load(src_cam)
    # Close files
    src_guy.close()
    src_cam.close()

    #
    ## Data gathering
    #
    ## !!! IMPORTANT !!! > axis need to be swapped as Unreal is Left-Handed, while OpenCV is Right-Handed
    # 

    # Camera base frame location & rotation (in our scenario the camera is fixed)
    quat_cam = np.deg2rad(np.array([data_cam["Camera_rot"]["x"], data_cam["Camera_rot"]["y"], data_cam["Camera_rot"]["z"], data_cam["Camera_rot"]["w"]], dtype=np.float32))
    tvec_cam = np.array([data_cam["Camera_pos"]["x"], data_cam["Camera_pos"]["y"], data_cam["Camera_pos"]["z"]], dtype=np.float32)
    quat_cam, tvec_cam = leftToRightHanded(quat_cam, tvec_cam)
    # Camera intrinsics
    fov = np.deg2rad(np.float32(data_cam["Camera_FOV"]))
    aspectRatio = np.float32(data_cam["Camera_AspectRatio"])

    # Skeleton
    # 3D coordinates
    points3d = np.array([[[pos["loc"]["x"], pos["loc"]["y"], pos["loc"]["z"]] for pos in val["Positions"]] for val in data_guy["Body"]], dtype=np.float32).T
    for i in range(points3d.shape[1]) :
        for j in range(points3d.shape[2]) : 
            _ , points3d[:,i,j] = leftToRightHanded(np.array((0,0,0,1)), points3d[:,i,j])
    x = points3d[0,:,:]
    y = points3d[1,:,:]
    z = points3d[2,:,:]

    # 2D coordinates
    points2d = np.array([[[pos["pixel"]["x"], pos["pixel"]["y"]] for pos in val["Positions"]] for val in data_guy["Body"]], dtype=np.float32).T
    u = points2d[0,:,:]
    v = points2d[1,:,:]

    # skeleton base frame location & rotation
    #   only first frame is taken as in our case the
    #   skeleton base is fixed
    quat_guy = np.deg2rad(np.array([data_guy["Body"][0]["Guy_frame_rotation"]["x"], data_guy["Body"][0]["Guy_frame_rotation"]["y"], data_guy["Body"][0]["Guy_frame_rotation"]["z"], data_guy["Body"][0]["Guy_frame_rotation"]["w"]], dtype=np.float32))
    tvec_guy = np.array([data_guy["Body"][0]["Guy_frame_location"]["x"], data_guy["Body"][0]["Guy_frame_location"]["y"], data_guy["Body"][0]["Guy_frame_location"]["z"]], dtype=np.float32)
    quat_guy, tvec_guy = leftToRightHanded(quat_guy, tvec_guy)

    #
    ## Plotting data
    #
    n_markers = x.shape[0]
    n_frames = x.shape[1]
    guy_bones = [bone["name"] for bone in data_guy["Body"][0]["Positions"]]
    bones_index_map = {bone: guy_bones.index(bone) for bone in guy_bones}
    lines_map = {bones_index_map[bone]: [bones_index_map[child] for child in children] for bone, children in guy_bones_map.items()}

    #
    ## Video Plotting
    #
    cap = cv2.VideoCapture("media/Level.avi")
    #out = cv2.VideoWriter('guyWithSkel.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (width,height))
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Defining camera intrinsics
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx = np.float32((width)/2)
    cy = np.float32((height)/2)
    fx = fy = width/(2*np.tan(fov/2))
    cameraMatrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ], dtype = np.float32)
    cameraDistortion = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    # Uncomment to extract camera intrinsics via calibrating inside the virtual environment
    #print("Calibrating camera...")
    #cameraMatrix, cameraDistortion, _, _ = get_camera_params(display=True)

    for t in range(0, n_frames) :

        res, frame = cap.read()
        if res == False:
            break

        object3dPoints = np.array((x[:,t], y[:,t], z[:,t]), dtype=np.float32).T
        object2dPoints = np.array((u[:,t], v[:,t]), dtype=np.float32).T     # ground truth of joints' pixel coordinates
        rvec_guy = Rotation.from_quat(quat_guy).as_rotvec()
        rvec_cam = Rotation.from_quat(quat_cam).as_rotvec()

        imagePoints = worldToPixel(object3dPoints, rvec_guy, tvec_guy, rvec_cam, tvec_cam, cameraMatrix, cameraDistortion)
        error = projectionError(imagePoints, object2dPoints, t)

        for i, point in enumerate(imagePoints.astype(int)):
            u_i, v_i = point
            cv2.circle(frame, (u_i, v_i), 5, (0, 0, 255), 1)
            for connection in lines_map[i] :
                u_j, v_j = imagePoints[connection].astype(int)
                cv2.line(frame, (u_i, v_i), (u_j, v_j), (255, 0, 0), 1)

        #out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #out.release()
    cap.release()
    cv2.destroyAllWindows()