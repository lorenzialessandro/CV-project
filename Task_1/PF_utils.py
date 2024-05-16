import numpy as np

# Initialize particles uniformly inside the specified lmits
def initialize_particles(NUM_PARTICLES, VEL_RANGE, LIM_X, LIM_Y, LIM_Z):
    x_min, x_max = LIM_X
    y_min, y_max = LIM_Y
    z_min, z_max = LIM_Z
    
    particles = np.random.rand(NUM_PARTICLES, 6)
    particles = particles * np.array((np.abs(x_max-x_min), np.abs(y_max-y_min), np.abs(z_max-z_min) ,VEL_RANGE, VEL_RANGE, VEL_RANGE))
    
    particles[:,0] += x_min
    particles[:,1] += y_min
    particles[:,2] += z_min
    particles[:,3:6] -= VEL_RANGE/2.0

    return particles

# Move particles according to the specified velocities
def apply_velocity(particles):
    particles[:,0] += particles[:,3]
    particles[:,1] += particles[:,4]
    particles[:,2] += particles[:,5]

    return particles

# Make sure particles don't get out the defined domain after step
def enforce_edges(particles, NUM_PARTICLES, LIM_X, LIM_Y, LIM_Z):
    x_min, x_max = LIM_X
    y_min, y_max = LIM_Y
    z_min, z_max = LIM_Z
    
    for i in range(NUM_PARTICLES):
        particles[i,0] = max(x_min, min(x_max, particles[i,0]))
        particles[i,1] = max(y_min, min(y_max, particles[i,1]))
        particles[i,2] = max(z_min, min(z_max, particles[i,2]))
    return particles


def compute_errors(x_t, y_t, z_t, particles, ref_marker, NUM_PARTICLES, N_MARKERS): 
    
    errors = np.ones(NUM_PARTICLES) * np.finfo(np.float32).max
    
    for idx, p in enumerate(particles):
        for mark in range(N_MARKERS) :
            # comute particles's loss w.r.t. each marker
            err = (x_t[mark] - p[0])**2 + (y_t[mark] - p[1])**2 + (z_t[mark] - p[2])**2
            
            # associate minimum error found and store correscponding marker ID
            if errors[idx] > err :
                errors[idx] = err
                ref_marker[idx] = mark
    
    return errors, ref_marker

def compute_weights(errors):
    weights = np.max(errors) - errors
    # Add small term to avoid division by zero
    weights += 1e-8
    
    return weights

def resample(particles, weights,NUM_PARTICLES):
    probabilities = weights / (np.sum(weights))

    idxs = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=probabilities)
    
    particles = particles[idxs, :]
    
    return particles


def apply_noise(particles, POS_SIGMA, VEL_SIGMA, NUM_PARTICLES):
    noise= np.concatenate(
    (
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1))
    ),
    axis=1)

    particles += noise
    return particles


# If the input values are absent, substitute them with the best fitting particle
def predict_pos(x_t, y_t, z_t, particles, weights, ref_marker, n_markers) :
    for i in range(n_markers) :
        if np.isnan(x_t[i]) or np.isnan(y_t[i]) or np.isnan(z_t[i]) :
            
            best_fit = np.finfo(np.float32).min

            for key, p in enumerate(particles) :
                if ref_marker[6] == i :
                    fit = weights[key]
                    
                    if fit > best_fit :
                        best_fit = fit
                        x_t[i] = p[0]
                        y_t[i] = p[1]
                        z_t[i] = p[2]
    
    return x_t, y_t, z_t


def apply_ParticleFilter (x, y, z, n_frames, n_markers):
    
    # Particle Filter parameters
    NUM_PARTICLES = 100
    VEL_RANGE = 0.005

    x_no_Nan = x[~np.isnan(x)]
    y_no_Nan = y[~np.isnan(y)]
    z_no_Nan = z[~np.isnan(z)]
    X_LIM = (np.min(x_no_Nan), np.max(x_no_Nan))
    Y_LIM = (np.min(y_no_Nan), np.max(y_no_Nan))
    Z_LIM = (np.min(z_no_Nan), np.max(z_no_Nan))
    
    # modules population area
    POS_SIGMA = 0.1
    # alterates new population velocities
    VEL_SIGMA = 0.00025
    
    # Output which contains both markers positions and particles
    x_out = np.zeros((n_markers + NUM_PARTICLES, n_frames), dtype=np.float32)
    y_out = np.zeros((n_markers + NUM_PARTICLES, n_frames), dtype=np.float32)
    z_out = np.zeros((n_markers + NUM_PARTICLES, n_frames), dtype=np.float32)
    
    # Initialize filter
    particles = initialize_particles(NUM_PARTICLES, VEL_RANGE, X_LIM, Y_LIM, Z_LIM)
    weights = np.ones(NUM_PARTICLES)
    ref_marker = np.ones(NUM_PARTICLES) * -1.0
    #print("INIT : ", particles[:,:3])
    
    for t in range(n_frames):
        # update current frame value according to the current best fits for each marker
        x_t, y_t, z_t = predict_pos(x[:,t], y[:,t], z[:,t], particles, weights, ref_marker, n_markers)
        # Save back the updated values
        x[:,t] = x_t
        y[:,t] = y_t
        z[:,t] = z_t
        
        x_out[:,t] = np.hstack((x[:,t], particles[:,0]))
        y_out[:,t] = np.hstack((y[:,t], particles[:,1]))
        z_out[:,t] = np.hstack((z[:,t], particles[:,2]))
        
        particles = apply_noise(particles,POS_SIGMA,VEL_SIGMA,NUM_PARTICLES)
        #print("AFTER NOISE : ", particles[:,:3])
        
        # Particle filter step
        particles = apply_velocity(particles)
        #print("VELOCITY APPLIED : ", particles[:,:3])
        
        particles = enforce_edges(particles, NUM_PARTICLES, X_LIM, Y_LIM, Z_LIM)
        #print("ENFORCE : ", particles[:,:3])
        
        errors, ref_marker = compute_errors(x_t, y_t, z_t, particles, ref_marker, NUM_PARTICLES, n_markers)
        weights = compute_weights(errors)
        
        particles = resample(particles, weights, NUM_PARTICLES)
        #print("RESAMPLING : ", particles[:,:3])
        
    return x_out, y_out, z_out
