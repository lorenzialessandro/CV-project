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
    
    # Setup variable to include prediction in the output too
    x_out = np.zeros((n_markers + n_markers, n_frames), dtype=np.float32)
    y_out = np.zeros((n_markers + n_markers, n_frames), dtype=np.float32)
    z_out = np.zeros((n_markers + n_markers, n_frames), dtype=np.float32)
    
    for t in range(0, n_frames):
        
        point = np.array([x[:,t], y[:,t], z[:,t]], np.float32).T
        point = np.array([point.flatten()]).T
        
        prediction = kalman.predict()

        for i in range(total_measure_params):
            if np.isnan(prediction[i,0]):
                exit()
        
        # Correct point with Kallman prediction if it's absent
        for i in range(point.shape[0]) :
            if np.isnan(point[i,0]) :
                point[i,0] = prediction[i,0]
        
        # Update the model
        kalman.correct(point)
        
        point = point.reshape(n_markers, 3)
        x[:,t] = point[:,0]
        y[:,t] = point[:,1]
        z[:,t] = point[:,2]
        
        prediction = prediction[:total_measure_params, 0].reshape(n_markers, 3)
        # Append prediction value to the output
        x_out[:,t] = np.hstack((x[:,t], prediction[:,0]))
        y_out[:,t] = np.hstack((y[:,t], prediction[:,1]))
        z_out[:,t] = np.hstack((z[:,t], prediction[:,2]))  
        
    return x_out, y_out, z_out