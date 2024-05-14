import numpy as np
import cv2

def duplicate_diagonally(pattern, repetitions):
    
    L = tuple([pattern.copy() for _ in range(repetitions)])
    
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype=np.float32)
    out[mask] = np.concatenate(L).ravel()
    
    return out


def apply_KallmanFilter(x, y, z, n_frames, n_markers, fps) :
    
    measure_params = 3
    dynam_params = 7
    total_measure_params = measure_params * n_markers
    total_dynam_params = dynam_params * n_markers
    
    kalman = cv2.KalmanFilter(total_dynam_params, total_measure_params) # (d,n_params)      
    
    # maps the state vector to the observed measurements
    kalman.measurementMatrix = np.eye(total_measure_params, total_dynam_params, dtype=np.float32)
    # maps the current state vector to its next state vector based on the defined motion model
    dt = 1/fps
    single_point_transitionMat = np.array([
        [1, 0, 0, dt,  0,  0,  0],
        [0, 1, 0,  0, dt,  0,  0],
        [0, 0, 1,  0,  0, dt,  0],
        [0, 0, 0,  1,  0,  0, dt],
        [0, 0, 0,  0,  1,  0,  0],
        [0, 0, 0,  0,  0,  1,  0],
        [0, 0, 0,  0,  0,  0,  1],
    ], dtype=np.float32)
    
    kalman.transitionMatrix = duplicate_diagonally(single_point_transitionMat, n_markers)
    # Models the uncertainty in the motion model
    kalman.processNoiseCov = np.identity(total_dynam_params, dtype=np.float32) * 1e-5
    # Models the uncertainty of mesurements themselves
    kalman.measurementNoiseCov = np.identity(total_measure_params, dtype=np.float32) * 1e-5  
    
    for t in range(0, n_frames):
        
        point = np.array([x[:,t], y[:,t], z[:,t]], np.float32).T
        point = np.array([point.flatten()]).T
        
        prediction = kalman.predict()
        
        # Correct point with Kallman prediction if it's absent
        for i in range(point.shape[0]) :
            if np.isnan(point[i,0]) :
                point[i,0] = prediction[i,0]
        
        # Update the model
        kalman.correct(point)
        
        point = point.reshape(4,3)
        x[:,t] = point[:,0]
        y[:,t] = point[:,1]
        z[:,t] = point[:,2]
        
    return x, y, z