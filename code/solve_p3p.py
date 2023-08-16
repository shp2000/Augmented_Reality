
import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    ##### STUDENT CODE END #####
    #R, t = Procrustes(Pc_3d, Pw[1:4])
    f = (K[0][0]+K[1][1])/2
    Pc = (Pc.T -np.array( [[K[0][2],K[1][2]]]).T).T
    #print(Pc)
    #matri=[[0],[0],[0]]
    matri=[0,0,0]
    uni_mat = [[0],[0],[0]]
    uni=[[0],[0],[0]]
    j1=[[0],[0],[0]]
    j2=[[0],[0],[0]]
    j3=[[0],[0],[0]]
    for i in range(0,3):
        matri = np.array([[float(Pc[i][0])],
                    [float(Pc[i][1])],
                    [f]])
        
        
        #print(matri)
        matri = matri.reshape(3,1)
        uni = np.sqrt(Pc[i][0]*Pc[i][0] + Pc[i][1]*Pc[i][1] + f*f)
        uni_mat = matri/uni
        if(i==0):
            j1 = uni_mat
            #print(j1)
        if(i==1):
            j2 = uni_mat
            #print(j2)
        if(i==2):
            j3 = uni_mat
            #print(j3)

    # print(matri)
    #print(uni_mat)
    #print(j2)
    #print(j3)
    #cosalpha = uni_mat[1][0]@uni_mat[2][0]
    #cosbeta = uni_mat[0][0]@uni_mat[2][0]
    #cosgamma = uni_mat[0][0]@uni_mat[1][0]
    cosalpha = float(j2.T@j3)
    cosbeta = float(j1.T@j3)
    cosgamma = float(j1.T@j2)
    #print(cosalpha)
    a = float(np.sqrt((Pw[1][0]-Pw[2][0])*(Pw[1][0]-Pw[2][0]) + (Pw[1][1]-Pw[2][1])*(Pw[1][1]-Pw[2][1])))
    b = float(np.sqrt((Pw[0][0]-Pw[2][0])*(Pw[0][0]-Pw[2][0]) + (Pw[0][1]-Pw[2][1])*(Pw[0][1]-Pw[2][1])))
    c = float(np.sqrt((Pw[0][0]-Pw[1][0])*(Pw[0][0]-Pw[1][0]) + (Pw[0][1]-Pw[1][1])*(Pw[0][1]-Pw[1][1])))
    #print(a)
    #print(b)
    #print(c)

    A0 = (1+ ((a**2-c**2)/(b**2)))**2 - (4*(a**2)*(cosgamma**2))/(b**2)
    A1 = 4*(-((a**2-c**2)/(b**2))*(1 + ((a**2-c**2)/(b**2)))*cosbeta + (2*(a**2)*(cosgamma**2)*cosbeta)/(b**2) - (1-((a**2+c**2)/(b**2)))*cosalpha*cosgamma)
    A2 = 2*(((a**2-c**2)/(b**2))**2 - 1 + 2*(((a**2-c**2)/(b**2))**2)*(cosbeta**2) + 2*(((b**2-c**2)/(b**2)))*(cosalpha**2) - 4*((a**2+c**2)/(b**2))*cosalpha*cosbeta*cosgamma + 2*((b**2-a**2)/(b**2))*(cosgamma**2))
    
    t1=((a**2-c**2)/(b**2))*(1-((a**2-c**2)/(b**2)))*cosbeta
    t2 = (1-((a**2+c**2)/(b**2)))*cosalpha*cosgamma
    t3 = (2*(c**2)*(cosalpha**2)*cosbeta)/(b**2)
    
    #A3 = 4*(((a**2-c**2)/(b**2))*(1-((a**2-c**2)/(b**2)))*cosbeta - (1-((a**2-c**2)/(b**2)))*cosalpha*cosgamma + (2*(c**2)*(cosalpha**2)*cosbeta)/(b**2))
    A3 = 4*(t1-t2+t3)
    A4 = (((a**2-c**2)/(b**2)) - 1)**2 - (4*(c**2)*(cosalpha**2))/(b**2)

    coeff = [A4, A3, A2, A1, A0]
    #print(coeff)
    
    roots4 = np.roots(coeff)

    roots4 = roots4[np.isreal(roots4)]
    #print(roots4)

    v = roots4[0]

    u = ((-1 + ((a**2-c**2)/(b**2)))*(v**2) - 2*((a**2-c**2)/(b**2))*cosbeta*v + 1 + ((a**2-c**2)/(b**2)))/(2*(cosgamma-v*cosalpha))

    s1 = np.sqrt((c**2)/(1 + (u**2) -2*u*cosgamma))
    s2 = u*s1
    s3 = v*s1
    #print(s1)
    #print(s2)
    #print(s3)
    Y=[[0,0,0],
        [0,0,0],
        [0,0,0]]
    Y[0][0] = s1*Pc[0][0]/(np.sqrt((Pc[0][0])**2 + (Pc[0][1])**2 + f**2))
    Y[0][1] = s1*Pc[0][1]/(np.sqrt((Pc[0][0])**2 + (Pc[0][1])**2 + f**2))
    Y[0][2] = s1*f/(np.sqrt((Pc[0][0])**2 + (Pc[0][1])**2 + f**2))
    Y[1][0] = s2*Pc[1][0]/(np.sqrt((Pc[1][0])**2 + (Pc[1][1])**2+ f**2))
    Y[1][1] = s2*Pc[1][1]/(np.sqrt((Pc[1][0])**2 + (Pc[1][1])**2+ f**2))
    Y[1][2] = s2*f/(np.sqrt((Pc[1][0])**2 + (Pc[1][1])**2+ f**2))
    Y[2][0] = s3*Pc[2][0]/(np.sqrt((Pc[2][0])**2 + (Pc[2][1])**2+ f**2))
    Y[2][1] = s3*Pc[2][1]/(np.sqrt((Pc[2][0])**2 + (Pc[2][1])**2+ f**2))
    Y[2][2] = s3*f/(np.sqrt((Pc[2][0])**2 + (Pc[2][1])**2+ f**2))
    Y = np.real(Y)
    print(Y)
    #print(Pw[])
    
    R, t = Procrustes(Y,Pw[0:3])
    #print(R)
    #print(t)
    
    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X_c: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    X_c = np.mean(X,axis=0)
    
    Y_c = np.mean(Y,axis=0)
    
    X_new1 =X-X_c
    Y_new1 =Y-Y_c
    X_new =  np.transpose(X_new1)
    Y_new = np.transpose(Y_new1)
    R_prime=np.matmul(Y_new,np.transpose(X_new))
    # print(R_prime)
    U, S , Vt = np.linalg.svd(R_prime)
    #mamul = np.matmul(U, Vt)
    # print(Vt)
    # print(U)
    int1 = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, np.linalg.det(np.matmul(np.transpose(Vt), np.transpose(U)))]]
    R1 = np.matmul(U,int1)
    R2 = np.matmul(R1, Vt)
    t = Y_c - np.matmul(R2, X_c)
    R=R2
    ##### STUDENT CODE END #####

    return R, t