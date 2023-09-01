import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    # Inverse of camera intrinsic matrix
    K_inv = np.linalg.inv(K)
    
    # Convert pixels to homogeneous coordinates
    pixels_homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    
    # Transform pixel coordinates to normalized image coordinates
    x_normalized = np.dot(K_inv, pixels_homogeneous.T).T
    
    # Create the direction vectors in camera coordinates (z = 0 in the world)
    direction_c = np.hstack((x_normalized[:, :2], np.zeros((x_normalized.shape[0], 1))))
    
    # Transform direction vectors to world coordinates
    direction_w = np.dot(R_wc, direction_c.T).T
    
    # Calculate the scale factor to account for the z-coordinate being 0
    scale = t_wc[2] / direction_w[:, 2]
    
    # Calculate world coordinates
    Pw = t_wc + scale[:, np.newaxis] * direction_w
    ##### STUDENT CODE END #####
    return Pw
