import json
import cv2
import numpy as np
from numpy.linalg import inv
from utility import * # utilities functions: see utility.py
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
        [0,  1,  0,  0],
        [0,  0, -1,  0],
        [1,  0,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)

    # Apply transformation
    mat = C @ mat @ inv(C)

    #print(f"A rvec - {rvec}")
    #print(f"A tvec - {tvec}")
    rvec, tvec = matrix_to_rtvec(mat)
    #print(f"B rvec - {rvec.T}")
    #print(f"B tvec - {tvec}")
    #exit()
    return rvec, tvec



#def leftToRightHanded (rvec, tvec):
#    # Left handed (Unreal Enginge) :  (+X: forward, +Y: right, +Z: up) 
#    # Right handed (openCV) : (+X: right, +Y: down, +Z: forward)
#    mat = rtvec_to_matrix(rvec, tvec)
#    cpy = np.copy(mat)
#    
#    # Swap Z & Y Rots the matrix right handed
#    mat[1,:] = cpy[2,:]
#    mat[:,1] = cpy[:,2]
#    mat[2,:] = cpy[1,:]
#    mat[:,2] = cpy[:,1]
#    # swap positions independently
#    #mat[0,3] = cpy[1,3]
#    #mat[1,3] = -cpy[2,3]
#    #mat[2,3] = cpy[0,3]
#
#    # apply chain of transforms
#    #mat = rotX(np.deg2rad(180)) @ rotY(np.deg2rad(90)) @ mat
#
#    rvec, tvec = matrix_to_rtvec(mat)
#
#    return rvec, tvec

def worldToPixel(objectPoints, tvec_obj, tvec_cam, rvec_obj, rvec_cam, cameraMatrix):
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
    rvec, tvec  = matrix_to_rtvec(T_world_cam)
    # Obtain pixel convertion
    imagePoints, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, None)
    imagePoints = imagePoints.squeeze().astype(int)

    return imagePoints


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

    #
    ## Data gathering
    #
    ## !!! IMPORTANT !!! > axis need to be swapped as Unreal is Left-Handed, while OpenCV is Right-Handed ?check well?
    # 

    # Camera tranform
    rvec_cam = np.deg2rad(np.array(data_cam["Camera_rot"], dtype=np.float32))
    tvec_cam = np.array(data_cam["Camera_pos"], dtype=np.float32)
    rvec_cam, tvec_cam = leftToRightHanded(rvec_cam, tvec_cam)
    # Camera intrinsics
    fov = np.deg2rad(np.float32(data_cam["Camera_FOV"]))
    cam_ar =  np.float32(data_cam["Camera_AspectRatio"])
    
    # Skeleton poses
    points = np.array([[[pos["X"], pos["Y"], pos["Z"]] for pos in val["Positions"]] for val in data_guy["Body"]], dtype=np.float32).T
    for i in range(points.shape[1]) :
        for j in range(points.shape[2]) : 
            _ , points[:,i,j] = leftToRightHanded(np.zeros(3), points[:,i,j])
    x = points[0,:,:]
    y = points[1,:,:]
    z = points[2,:,:]

    # skeleton rots
    rvec_obj = np.deg2rad(np.array(data_guy["Body"][0]["Guy_frame_rotation"], dtype=np.float32))
    tvec_obj = np.array(data_guy["Body"][0]["Guy_frame_location"], dtype=np.float32)
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

    cap = cv2.VideoCapture("video.avi")
    #out = cv2.VideoWriter('guyWithSkel.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (width,height))
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fx = width/(2*np.tan(fov/2))
    fy = height/(2*np.tan(fov/2))
    cx = np.float32(width/2)
    cy = np.float32(height/2)

    cameraMatrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype = np.float32)

    for t in range(0, n_frames) :

        res, frame = cap.read()
        if res == False:
            break

        objectPoints = np.array((x[:,t], y[:,t], z[:,t]), dtype=np.float32).T
        imagePoints = worldToPixel(objectPoints, tvec_obj, tvec_cam, rvec_obj, rvec_cam, cameraMatrix)

        for key, point in enumerate(imagePoints):
            u, v = point
            cv2.circle(frame, (u,v), 5, (0, 0, 255), 1)
            
            for connection in lines_map[key] :
                u_2, v_2 = imagePoints[connection]
                cv2.line(frame, (u,v), (u_2, v_2), (255, 0, 0), 1)

        #out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break        

        print(f"Frame {t} {imagePoints}")
        #print(f"pos: {objectPoints[0,:]}")

    #out.release()
    cap.release()
    cv2.destroyAllWindows()