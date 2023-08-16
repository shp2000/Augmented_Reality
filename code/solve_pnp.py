from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    H = est_homography(Pw[:, :-1], Pc)
    print(Pw[:, :-1])
    H = H/H[2][2]
    col = np.matmul(np.linalg.inv(K), H)
    h1_prime = col[:,0] 
    h2_prime = col[:,1] 
    h3_prime = col[:,2] 
    cros = np.cross(h1_prime, h2_prime)
    A2 = [[h1_prime[0], h2_prime[0], cros[0]],
            [h1_prime[1], h2_prime[1], cros[1]],
            [h1_prime[2], h2_prime[2], cros[2]]]
    [U, S , Vt ] = np.linalg.svd(A2)
    mamul = np.matmul(U, Vt)
    int1 = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, np.linalg.det(np.matmul(U, Vt))]]
    R1 = np.matmul(U,int1 )
    R2 = np.matmul(R1, Vt)
    R = np.transpose(R2)
    t1 = np.linalg.norm(h1_prime)
    t2 = np.transpose(h3_prime/t1)
    t = np.matmul(R,-t2)
    



    ##### STUDENT CODE END #####

    return R, t
