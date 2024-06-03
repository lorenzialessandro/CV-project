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


def compute_errors(x_t, y_t, z_t, particles, ref_marker, NUM_PARTICLES): 
    
    errors = np.ones(NUM_PARTICLES) * np.inf
    
    for idx, p in enumerate(particles):     
        # index of the marker associated to "p"
        m = ref_marker[idx]
        # comute particles's loss w.r.t. the associated marker
        errors[idx] = (x_t[m] - p[0])**2 + (y_t[m] - p[1])**2 + (z_t[m] - p[2])**2

    return errors


def compute_weights(errors):
    weights = np.max(errors) - errors
    # Add small term to avoid division by zero
    weights += 1e-8
    
    return weights


def resample(particles, weights, ref_marker, N_MARKERS):
    resampled_particles = np.empty_like(particles)
    
    for marker_idx in range(N_MARKERS):
        # Take all particles associated to the a marker
        marker_indices = np.where(ref_marker == marker_idx)[0]
        marker_weights = weights[marker_indices]
        
        # Compute a probability distribution among them
        probabilities = marker_weights / np.sum(marker_weights)
        
        resampled_indices = np.random.choice(marker_indices, size=len(marker_indices), p=probabilities)
        resampled_particles[marker_indices] = particles[resampled_indices]
        
    return resampled_particles


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
def predict_pos(x_t, y_t, z_t, particles, weights, ref_marker, N_MARKERS) :
    
    for i in range(N_MARKERS) :
        if np.isnan(x_t[i]) or np.isnan(y_t[i]) or np.isnan(z_t[i]) :            
            
            best_fit = -np.inf
            
            for key, p in enumerate(particles) :
                if ref_marker[key] == i :
                    fit = weights[key]
                    
                    if fit > best_fit :
                        best_fit = fit
                        x_t[i], y_t[i], z_t[i] = p[0], p[1], p[2]
    
    return x_t, y_t, z_t


def apply_ParticleFilter (x, y, z, n_frames, n_markers):
    
    # Particle Filter parameters
    NUM_PARTICLES = 250
    VEL_RANGE = 0.01

    # Consider as domain limit for the particles min - max values found in x, y, z with some tollerance
    x_no_Nan = x[~np.isnan(x)]
    y_no_Nan = y[~np.isnan(y)]
    z_no_Nan = z[~np.isnan(z)]
    X_LIM = (1.25 * np.min(x_no_Nan), 1.25 * np.max(x_no_Nan))
    Y_LIM = (1.25 * np.min(y_no_Nan), 1.25 * np.max(y_no_Nan))
    Z_LIM = (1.25 * np.min(z_no_Nan), 1.25 * np.max(z_no_Nan))
    
    # modules population area
    POS_SIGMA = 0.005
    # alterates new population velocities
    VEL_SIGMA = 0.0005
    
    # Output which contains both markers positions and particles
    x_out = np.zeros((n_markers + NUM_PARTICLES, n_frames), dtype=np.float32)
    y_out = np.zeros((n_markers + NUM_PARTICLES, n_frames), dtype=np.float32)
    z_out = np.zeros((n_markers + NUM_PARTICLES, n_frames), dtype=np.float32)
    
    # Initialize filter
    particles = initialize_particles(NUM_PARTICLES, VEL_RANGE, X_LIM, Y_LIM, Z_LIM)
    weights = np.ones(NUM_PARTICLES)
    # Randomly assign each particle to a specific marker
    ref_marker = np.array(np.random.rand(NUM_PARTICLES) * n_markers, dtype=np.uint8)

    
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
        
        # Particle filter step
        particles = apply_noise(particles, POS_SIGMA, VEL_SIGMA, NUM_PARTICLES)
        particles = apply_velocity(particles)
        particles = enforce_edges(particles, NUM_PARTICLES, X_LIM, Y_LIM, Z_LIM)
        errors = compute_errors(x_t, y_t, z_t, particles, ref_marker, NUM_PARTICLES)
        weights = compute_weights(errors)
        particles = resample(particles, weights, ref_marker, n_markers)

    return x_out, y_out, z_out
