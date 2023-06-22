import numpy as np
import scipy.optimize

class Optimize_Return:
    def __init__(self, A: np.ndarray, D: np.ndarray, points: list[np.ndarray]) -> None:
        self.A = A
        self.D = D
        self.points = points

class Maximum_Likelihood:
    @staticmethod
    def optimize(all_image_corners: list[np.ndarray], world_corners: np.ndarray, all_rt_matricies: list[np.ndarray], A_init: np.ndarray, all_H_init: list[np.ndarray]) -> "Optimize_Return":
        """Optimizes the camera parameters and distortion coefficient by maximum likelihood estimation

        Args:
            all_image_corners (list[np.ndarray]): All image corners from all images. Each image is a np.ndarray of shape (n,2) where n is the number of points (for chess board of size 6x9, this would be 54)
            world_corners (np.ndarray): ndarray of shape (n,2) where n is the number of points each image would have (for chess board of size 6x9, this would be 54)
            all_rt_matricies (list[np.ndarray]): list of rotation&translation matricies for each image. np.ndarray of shape (3,4)
            A_init (np.ndarray): A matrix of shape (3,3)
            all_H_init (list[np.ndarray]): list of all homography matricies for each image. np.ndarray of shape (3,3)

        Returns:
            Optimize_Return: Returns the optimized A matrix, distortion coefficient and reprojection points. Ready to be used for cv2.undistort(image, A, D)
        """
        # now we optimize by maximum likelihood estimation
        print("Init Kc")
        # initializing distortion coefficient as [0,0]
        kc_init = np.array([0,0]).reshape(2,1)
        print("Initialized kc =\n", kc_init)
         # optimization
        print("Optimizing ...")
        # x0 are all parameters of A and kc as a 1d array
        # since scipy only takes 1d arrays as input
        x0 = Maximum_Likelihood.__extractParamFromA(A_init, kc_init)
        # optimize by least square
        # this takes ages btw
        res = scipy.optimize.least_squares(fun=Maximum_Likelihood.__lossFunc, x0=x0, method="lm", args=[all_rt_matricies, all_image_corners, world_corners]) # type: ignore
        # get the results
        x1 = res.x
        # retrieve A from the x0 1d array
        AK = Maximum_Likelihood.__retrieveA(x1)
        A_new = AK[0]
        # and the new distortion coefficient
        kc_new = AK[1]
        
        # calculate some error statistics
        previous_error, _ = Maximum_Likelihood.__reprojectPointsAndGetError(A_init, kc_init, all_rt_matricies, all_image_corners, world_corners)
        att_RT_new = Maximum_Likelihood.__A(A_new, all_H_init)
        new_error, points = Maximum_Likelihood.__reprojectPointsAndGetError(A_new, kc_new, att_RT_new, all_image_corners, world_corners)
        
        print("The error befor optimization was ", previous_error)
        print("The error after optimization is ", new_error)
        print("The A matrix is:\n", A_new)
        print("The kc matrix is:\n", kc_new)
        
        # show the results
        # reshape so cv understands A and kc
        K = np.array(A_new, np.float32).reshape(3,3)
        D = np.array([kc_new[0,0], kc_new[1,0], 0, 0], np.float32)
        return Optimize_Return(K, D, points)
        
    @staticmethod 
    def __extractParamFromA(init_A, init_kc):
        alpha = init_A[0,0]
        gamma = init_A[0,1]
        beta = init_A[1,1]
        u0 = init_A[0,2]
        v0 = init_A[1,2]
        k1 = init_kc[0,0]
        k2 = init_kc[1,0]

        x0 = np.array([alpha, gamma, beta, u0, v0, k1, k2])
        return x0

    @staticmethod
    def __lossFunc(x0, init_all_RT, all_image_corners, world_corners):
        # get A and kc from x0
        A, _ = Maximum_Likelihood.__retrieveA(x0)
        # and all the parameters it has
        _, _, _, u0, v0, k1, k2 = x0

        # this will be our output
        error_mat = []
        for i, image_corners in enumerate(all_image_corners):

            # Rotation and Translation matrix for 3d world points
            # this is a 3x3 matrix
            RT = init_all_RT[i]
            # get A * RT for 2d world points
            RT3 = np.array([RT[:,0], RT[:,1], RT[:,3]]).reshape(3,3)
            RT3 = RT3.T
            ART3 = np.dot(A, RT3)

            # accumulation error for each image
            image_total_error = 0
            # for each point in the world corners
            for j in range(world_corners.shape[0]):

                world_point_2d = world_corners[j]
                world_point_2d_homo = np.array([world_point_2d[0], world_point_2d[1], 1]).reshape(3,1)
                world_point_3d_homo = np.array([world_point_2d[0], world_point_2d[1], 0, 1]).reshape(4,1)

                #get radius of distortion
                XYZ = np.dot(RT, world_point_3d_homo)
                x =  XYZ[0] / XYZ[2]
                y = XYZ[1] / XYZ[2]
                # x = alpha * XYZ[0] / XYZ[2] #assume gamma as 0 
                # y = beta * XYZ[1] / XYZ[2] #assume gamma as 0
                r = np.sqrt(x**2 + y**2) #radius of distortion

                #observed image co-ordinates
                mij = image_corners[j]
                mij = np.array([mij[0], mij[1], 1], dtype = 'float').reshape(3,1)

                #projected image co-ordinates
                uvw = np.dot(ART3, world_point_2d_homo)
                u = uvw[0] / uvw[2]
                v = uvw[1] / uvw[2]

                u_dash = u + (u - u0) * (k1 * r**2 + k2 * r**4)
                v_dash = v + (v - v0) * (k1 * r**2 + k2 * r**4)
                # this is a 1d array, weird, but we need values to initialze the resulting
                # array
                mij_dash = np.array([u_dash[0], v_dash[0], 1], dtype = 'float').reshape(3,1)

                # get error
                e = np.linalg.norm((mij - mij_dash), ord=2)
                image_total_error = image_total_error + e

            error_mat.append(image_total_error / 54)
        
        return np.array(error_mat)
    
    @staticmethod
    def __retrieveA(x0):
        # returns the A and distortion of a given x0
        alpha, gamma, beta, u0, v0, k1, k2 = x0
        A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]).reshape(3,3)
        kc = np.array([k1, k2]).reshape(2,1)
        return A, kc
    @staticmethod
    def __reprojectPointsAndGetError(A, kc, all_RT, all_image_corners, world_corners):
        # basically lossFunc again
        # without x0 as input
        error_mat = []
        _, _, _, u0, v0, k1, k2 = Maximum_Likelihood.__extractParamFromA(A, kc)
        all_reprojected_points = []
        for i, image_corners in enumerate(all_image_corners):

            RT = all_RT[i]
            RT3 = np.array([RT[:,0], RT[:,1], RT[:,3]]).reshape(3,3)
            RT3 = RT3.T
            ART3 = np.dot(A, RT3)

            image_total_error = 0
            reprojected_points = []
            for j in range(world_corners.shape[0]):

                world_point_2d = world_corners[j]
                world_point_2d_homo = np.array([world_point_2d[0], world_point_2d[1], 1]).reshape(3,1)
                world_point_3d_homo = np.array([world_point_2d[0], world_point_2d[1], 0, 1]).reshape(4,1)

                XYZ = np.dot(RT, world_point_3d_homo)
                x =  XYZ[0] / XYZ[2]
                y = XYZ[1] / XYZ[2]
                r = np.sqrt(x**2 + y**2)

                mij = image_corners[j]
                mij = np.array([mij[0], mij[1], 1], dtype = 'float').reshape(3,1)

                uvw = np.dot(ART3, world_point_2d_homo)
                u = uvw[0] / uvw[2]
                v = uvw[1] / uvw[2]

                u_dash = u + (u - u0) * (k1 * r**2 + k2 * r**4)
                v_dash = v + (v - v0) * (k1 * r**2 + k2 * r**4)
                reprojected_points.append([u_dash, v_dash])

                mij_dash = np.array([u_dash[0], v_dash[0], 1], dtype = 'float').reshape(3,1)

                e = np.linalg.norm((mij - mij_dash), ord=2)
                image_total_error = image_total_error + e
            
            all_reprojected_points.append(reprojected_points)
            error_mat.append(image_total_error)
        error_mat = np.array(error_mat)
        error_average = np.sum(error_mat) / (len(all_image_corners) * world_corners.shape[0])
        return error_average, all_reprojected_points
    @staticmethod
    def __A(A,all_H):
        D=[]
        for B in all_H:E=B[:,0];H=B[:,1];I=B[:,2];C=np.linalg.norm(np.dot(np.linalg.inv(A),E),2);F=np.dot(np.linalg.inv(A),E)/C;G=np.dot(np.linalg.inv(A),H)/C;J=np.cross(F,G);K=np.dot(np.linalg.inv(A),I)/C;L=np.vstack((F,G,J,K)).T;D.append(L)
        return D

    