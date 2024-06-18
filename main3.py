import json
import cv2
import numpy as np
from numpy.linalg import inv
from utils.cameraCalibSquares import *
import time

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

def rotX (theta) :
    matrix = np.array([
        [1,             0,              0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0,              0,             0, 1]
    ], dtype=np.float32)
    return matrix
    
def rotY (theta) :
    matrix = np.array([
        [np.cos(theta),  0,  np.sin(theta), 0],
        [0,              1,              0, 0],
        [-np.sin(theta), 0,  np.cos(theta), 0],
        [0,               0,              0,1]
    ], dtype=np.float32)
    return matrix

def rotZ (theta) :
    matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [            0,              0, 1, 0],
        [            0,              0, 0, 1]

    ], dtype=np.float32)
    return matrix

def rtvec_to_matrix(rvec, tvec):
    "Convert rotation vector and translation vector to 4x4 matrix"
    T = np.eye(4)
    R, jac = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    return T

def matrix_to_rtvec(matrix):
    "Convert 4x4 matrix to rotation vector and translation vector"
    rvec, jac = cv2.Rodrigues(matrix[:3, :3])
    tvec = matrix[:3, 3]
    return rvec, tvec



def leftToRightHanded (rvec, tvec):
    # Left handed (Unreal Enginge) :  (+X: forward, +Y: right, +Z: up) 
    # Right handed (openCV) : (+X: right, +Y: down, +Z: forward)
    mat = rtvec_to_matrix(rvec, tvec)

    C = np.array([
        [0,  1,   0,  0],
        [0,  0,  -1,  0],
        [1,  0,   0,  0],
        [0,  0,   0,  1]
    ], dtype=np.float32)
    
    # Apply transformation
    mat = (C) @ mat @ inv(C)

    rvec, tvec = matrix_to_rtvec(mat)

    return rvec, tvec


def worldToPixel(objectPoints, rvec_obj, tvec_obj, rvec_cam, tvec_cam, cameraMatrix, cameraDistortion):
    """
    :param objectPoints: object's point in world frame coordinates
    :param tvec_obj: translation vector for object in world frame coordinates
    :param tvec_cam: translation vector for camera in world frame coordinates
    :param rvec_obj: rotation vector for object in world frame coordinates
    :param rvec_cam: rotation vector for camera in world frame coordinates
    :param cameraMatrix: intrinsic camera parameters
    """
    T_world_cam = rtvec_to_matrix(rvec_cam, tvec_cam)    # Transformation matrix world -> cam
    T_world_obj = rtvec_to_matrix(rvec_obj, tvec_obj)    # Transformation matrix world -> obj
    T_cam_obj = inv(T_world_cam) @ T_world_obj           # Transformation matrix cam -> obj

    # Get mapping cam -> object
    rvec, tvec  = matrix_to_rtvec(inv(T_world_cam))

    #objectPoints = np.hstack((objectPoints, np.ones((objectPoints.shape[0], 1))))
    #objectPoints = (inv(T_world_obj) @ objectPoints.T).T
    #objectPoints = objectPoints[:,:3]

    # Obtain pixel convertion
    imagePoints, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, cameraDistortion)
    imagePoints = imagePoints.squeeze().astype(int)

    return imagePoints

    def refWorldPixel (objectPoints, UVs) :
        return

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

    # Camera tranform
    rvec_cam = np.deg2rad(np.array([data_cam["Camera_rot"]["roll"], data_cam["Camera_rot"]["pitch"], data_cam["Camera_rot"]["yaw"]], dtype=np.float32))
    tvec_cam = np.array([data_cam["Camera_pos"]["x"], data_cam["Camera_pos"]["y"], data_cam["Camera_pos"]["z"]], dtype=np.float32)
    rvec_cam, tvec_cam = leftToRightHanded(rvec_cam, tvec_cam)
    # Camera intrinsics
    fov = np.deg2rad(np.float32(data_cam["Camera_FOV"]))
    aspectRatio = np.float32(data_cam["Camera_AspectRatio"])
    cameraDistortion = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    # Skeleton
    # 3D coordinates
    points3d = np.array([[[pos["loc"]["x"], pos["loc"]["y"], pos["loc"]["z"]] for pos in val["Positions"]] for val in data_guy["Body"]], dtype=np.float32).T
    for i in range(points3d.shape[1]) :
        for j in range(points3d.shape[2]) : 
            _ , points3d[:,i,j] = leftToRightHanded(np.zeros(3), points3d[:,i,j])
    x = points3d[0,:,:]
    y = points3d[1,:,:]
    z = points3d[2,:,:]

    # 2D coordinates
    points2d = np.array([[[pos["pixel"]["x"], pos["pixel"]["y"]] for pos in val["Positions"]] for val in data_guy["Body"]], dtype=np.float32).T
    u = points2d[0,:,:]
    v = points2d[1,:,:]

    # skeleton rots
    rvec_obj = np.deg2rad(np.array([data_guy["Body"][0]["Guy_frame_rotation"]["roll"], data_guy["Body"][0]["Guy_frame_rotation"]["pitch"], data_guy["Body"][0]["Guy_frame_rotation"]["yaw"]], dtype=np.float32))
    tvec_obj = np.array([data_guy["Body"][0]["Guy_frame_location"]["x"], data_guy["Body"][0]["Guy_frame_location"]["y"], data_guy["Body"][0]["Guy_frame_location"]["z"]], dtype=np.float32)
    rvec_obj, tvec_obj = leftToRightHanded(rvec_obj, tvec_obj)

    #
    ## Animation Plotting
    #

    n_markers = x.shape[0]
    n_frames = x.shape[1]
    guy_bones = [bone["name"] for bone in data_guy["Body"][0]["Positions"]]
    bones_index_map = {bone: guy_bones.index(bone) for bone in guy_bones}
    lines_map = {bones_index_map[bone]: [bones_index_map[child] for child in children] for bone, children in guy_bones_map.items()}
    #rot = (0, 0, 0)
    #create_single_plot(x,y,z, guy_bones, guy_bones_map, n_frames, n_markers, rot)
    #exit()

    #
    ## Video Plotting
    #

    cap = cv2.VideoCapture("media/video.avi")
    #out = cv2.VideoWriter('guyWithSkel.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (width,height))
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cx = np.float32((width)/2)
    cy = np.float32((height)/2)
    fx = width/(2*np.tan(fov/2))
    fy = fx * height/width
    
    cameraMatrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype = np.float32)

    viewMatrix = np.array([[-0.544639, -0.131197, -0.828345, 0],[-0.838671, 0.0852003, 0.537934, 0],[0, 0.987688, -0.156434, 0],[1395.66, -198.661, 828.027, 1]], dtype=np.float32)
    projectionMatrix = np.array([[2.94613, 0, 0, 0],[0, 5.23756, 0, 0],[0, 0, 0, 1],[0, 0, 10, 0]], dtype=np.float32)
    viewProjectionMatrix = np.array([[-1.60458, -0.687152, 0, -0.828345],[-2.47083, 0.446242, 0, 0.537934],[0, 5.17308, 0, -0.156434],[4111.79, -1040.5, 10, 828.027]], dtype=np.float32)
    #print(cameraMatrix)
    #exit()
    #print("Calibrating camera...")
    #cameraMatrix, cameraDistortion, _, _ = get_camera_params(display=True)

    K = 10
    object_points = points3d.T
    image_points = points2d.T
    ret, cameraMatrix, cameraDistortion, r_vecs, t_vecs = cv2.calibrateCamera(
        object_points[:K,:,:], image_points[:K,:,:], (width,height), cameraMatrix, cameraDistortion, flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_FOCAL_LENGTH)
    ) 
    print("Camera Matrix:\n", cameraMatrix)
    print("Camera Distortion:\n", cameraDistortion)

    #print("A : rvec : ", np.rad2deg(rvec_cam).T)
    #print("A : tvec : ", tvec_cam)
    #_, rvec_cam, tvec_cam = cv2.solvePnP(object_points[0,:,:], image_points[0,:,:], cameraMatrix, cameraDistortion)
    #print("B : rvec : ", np.rad2deg(rvec_cam).T)
    #print("B : tvec : ", tvec_cam.T)
    #exit()

    for t in range(0, n_frames) :

        res, frame = cap.read()
        if res == False:
            break

        objectPoints = np.array((x[:,t], y[:,t], z[:,t]), dtype=np.float32).T
        imagePoints = np.array((u[:,t], v[:,t]), dtype=np.float32).T
        imagePoints = worldToPixel(objectPoints, rvec_obj, tvec_obj, rvec_cam, tvec_cam, cameraMatrix, cameraDistortion)
        
        #objectPoints = np.hstack((objectPoints, np.zeros((objectPoints.shape[0],1))))
        #imagePoints = (inv(projectionMatrix) @ viewMatrix @ objectPoints.T).T
        #print(imagePoints)
        #exit()
        
        for i, point in enumerate(imagePoints.astype(int)):
            u_i, v_i = point
            cv2.circle(frame, (u_i, v_i), 5, (0, 0, 255), -1)
            for connection in lines_map[i] :
                u_j, v_j = imagePoints[connection].astype(int)
                cv2.line(frame, (u_i, v_i), (u_j, v_j), (255, 0, 0), 1)

        #out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break        

        print(f"Frame {t} {imagePoints}")

    #out.release()
    cap.release()
    cv2.destroyAllWindows()