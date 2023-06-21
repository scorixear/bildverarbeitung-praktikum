import os
import cv2 as cv
import numpy as np
import scipy.optimize

def main():
    # load in images
    images= loadImages("Woche_12/chess_images")
    
    # chessboard size
    h,w = [6,9]
    # get corners via cv2
    all_image_corners = getImagesPoints(images, h, w)

    # length of chessboard paper in mm
    square_side = 12.5
    # get world coordinates
    world_corners = getWorldPoints(square_side, h, w)
    
    # diplay images with found points
    displayCorners(images, all_image_corners, h, w)
    
    # initializations
    print(f"Calculating H for {len(images)} images")
    # calculate all homography matricies
    all_H_init = getAllH(all_image_corners, square_side, h, w)
    print("Calculating B")
    # calculate the B matricies
    B_init = getB(all_H_init)
    print("Estimated B =\n", B_init)
    print("Calculating A")
    # deduce the A matrix from B
    A_init = getA(B_init)
    print("Initialized A =\n",A_init)
    print("Calculating rotation and translation")
    # and calculate the rotation and translation matricies
    all_RT_init = getRotationAndTrans(A_init, all_H_init)
    # now we optimize by maximum likelihood estimation
    print("Init Kc")
    # initializing distortion coefficient as [0,0]
    kc_init = np.array([0,0]).reshape(2,1)
    print("Initialized kc =\n", kc_init)
    
    # optimization
    print("Optimizing ...")
    # x0 are all parameters of A and kc as a 1d array
    # since scipy only takes 1d arrays as input
    x0 = extractParamFromA(A_init, kc_init)
    # optimize by least square
    # this takes ages btw
    res = scipy.optimize.least_squares(fun=lossFunc, x0=x0, method="lm", args=[all_RT_init, all_image_corners, world_corners]) # type: ignore
    # get the results
    x1 = res.x
    # retrieve A from the x0 1d array
    AK = retrieveA(x1)
    A_new = AK[0]
    # and the new distortion coefficient
    kc_new = AK[1]
    
    # calculate some error statistics
    previous_error, _ = reprojectPointsAndGetError(A_init, kc_init, all_RT_init, all_image_corners, world_corners)
    att_RT_new = getRotationAndTrans(A_new, all_H_init)
    new_error, points = reprojectPointsAndGetError(A_new, kc_new, att_RT_new, all_image_corners, world_corners)
    
    print("The error befor optimization was ", previous_error)
    print("The error after optimization is ", new_error)
    print("The A matrix is:\n", A_new)
    print("The kc matrix is:\n", kc_new)
    
    # show the results
    # reshape so cv understands A and kc
    K = np.array(A_new, np.float32).reshape(3,3)
    D = np.array([kc_new[0,0], kc_new[1,0], 0, 0], np.float32)
    for i, image_points in enumerate(points):
        # apply undistortion
        image = cv.undistort(images[i], K, D)
        # draw the points
        for point in image_points:
            x = int(point[0])
            y = int(point[1])
            image = cv.circle(image, (x,y), 5, (0,0,255), 3)
        # and show the image
        # as images are quite big, we resize them
        cv.imshow('frame', cv.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3))))
        cv.waitKey()
    cv.destroyAllWindows()
    

def loadImages(folder_name):
    files = os.listdir(folder_name)
    print("Loading images from ", folder_name)
    images = []
    for f in files:
        # print(f)
        image_path = folder_name + "/" + f
        image = cv.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def displayCorners(images, all_corners, h, w):
    for i, image in enumerate(images):
        corners = all_corners[i]
        corners = np.float32(corners.reshape(-1, 1, 2))
        cv.drawChessboardCorners(image, (w, h), corners, True)
        img = cv.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
        cv.imshow('frame', img)
        # filename = save_folder + str(i) + "draw.png"
        # cv.imwrite(filename, img)
        cv.waitKey()
    cv.destroyAllWindows()

def getImagesPoints(imgs, h, w):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images = imgs.copy()
    all_corners = []
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (w, h), None)
        if ret == True:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            corners2 = corners2.reshape(-1,2)
            # corners2 = np.hstack((corners2.reshape(-1,2), np.ones((corners2.shape[0], 1))))
            all_corners.append(corners2)
    return all_corners

def getWorldPoints(square_side, h, w):
    # h, w = [6, 9]
    Yi, Xi = np.indices((h, w)) 
    offset = 0
    lin_homg_pts = np.stack(((Xi.ravel() + offset) * square_side, (Yi.ravel() + offset) * square_side)).T
    return lin_homg_pts

def getAllH(all_corners, square_side, h, w):
    set1 = getWorldPoints(square_side, h, w)
    # denotes all homography matricies for each image
    all_H = []
    for corners in all_corners:
        set2 = corners
        H = getH(set1, set2)
        # H, _ = cv2.findHomography(set1, set2, cv2.RANSAC, 5.0)
        all_H.append(H)
    return all_H


def getH(set1, set2):
    nrows = set1.shape[0]
    if (nrows < 4):
        print("Need atleast four points to compute SVD.")
        return 0
    # the world coordinates x and y
    x = set1[:, 0]
    y = set1[:, 1]
    # the image coordinates xp and yp
    xp = set2[:, 0]
    yp = set2[:,1]
    # the homography calculation matrix
    A = []
    for i in range(nrows):
        row1 = np.array([x[i], y[i], 1, 0, 0, 0, -x[i]*xp[i], -y[i]*xp[i], -xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, x[i], y[i], 1, -x[i]*yp[i], -y[i]*yp[i], -yp[i]])
        A.append(row2)

    A = np.array(A)
    # calculate Singular Value Decomposition
    _, _, V = np.linalg.svd(A, full_matrices=True)
    # the homography should be the last row of V converted to 3x3 matrix
    H = V[-1, :].reshape((3, 3))
    # and scaled by the middle
    H = H / H[2,2]
    return H

def getB(all_H):
    # get the V matrix defined by h_i^T*B*h_j = v_ij^T*b
    # where b is the underlying construction of B
    # b = [B_11, B_12, B_22, B_13, B_23, B_33]^T
    # and h_i is the ith column of H
    # h_i = [h_i1, h_i2, h_i3]^T
    
    # v_ij is defined as
    # [h_i1*h_j1, 
    #  h_i1*h_j2 + h_i2*h_j1, 
    #  h_i2*h_j2,
    #  h_i3*h_j1 + h_i1*h_j3,
    #  h_i3*h_j2 + h_i2*h_j3, 
    #  h_i3*h_j3]
    
    # and V is defined as [v_12^T, (v_11 - v_22)^T]^T for n images stacked vertically
    # resulting in a 2n*6 matrix
    v = getV(all_H)
    # vb = 0
    # estimate b
    _, _, V = np.linalg.svd(v)
    b = V[-1, :]
    print("B matrix is\n", b)
    # and reshape to 3x3 matrix
    B = arrangeB(b)  
    return B

def getV(all_H):
    # calculates V matrix from given H matricies
    v = []
    # for each H matrix
    for H in all_H:
        # get h_i and h_j columns
        h1 = H[:,0]
        h2 = H[:,1]
        # get the V_ij values as defined above
        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        # calculate each V element
        v.append(v12.T)
        v.append((v11 - v22).T)
    # and transform back to matrix of size 2n x 6
    return np.array(v)

def getVij(hi, hj):
    # calculates the v_ij values as defined above
    Vij = np.array([ hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], hi[1]*hj[1], hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], hi[2]*hj[2] ])
    return Vij.T
def arrangeB(b):
    # aranges the b vector into a 3x3 matrix
    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,0] = b[3]
    B[2,1] = b[4]
    B[2,2] = b[5]
    return B

def getA(B):
    # since B = A^T^-1 * A^-1
    # we can deduce alpha, beta, gamme, v0 and u0 from those equations
    # this is taken from the original microsoft paper
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf
    v0 = (B[0,1] * B[0,2] - B[0,0] * B[1,2])/(B[0,0] * B[1,1] - B[0,1]**2)
    lamb = B[2,2] - (B[0,2]**2 + v0 * (B[0,1] * B[0,2] - B[0,0] * B[1,2]))/B[0,0]
    alpha = np.sqrt(lamb/B[0,0])
    beta = np.sqrt(lamb * (B[0,0]/(B[0,0] * B[1,1] - B[0,1]**2)))
    gamma = -(B[0,1] * alpha**2 * beta) / lamb 
    u0 = (gamma * v0 / beta) - (B[0,2] * alpha**2 / lamb)
    # and there we have our A matrix
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    return A

def getRotationAndTrans(A, all_H):
    # calculates r1, r2, r3 and t from A and H
    all_RT = []
    for H in all_H:
        # get the columns h1, h2, and h3
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        # lambda is defined as 1/||A^-1*h1||
        lamb = np.linalg.norm(np.dot(np.linalg.inv(A), h1), 2)
        # and r1, r2, r3 and t are defined as
        # r1 = 1/lambda * A^-1 * h1
        r1 = np.dot(np.linalg.inv(A), h1) / lamb
        # r2 = 1/lambda * A^-1 * h2
        r2 = np.dot(np.linalg.inv(A), h2) / lamb
        # r3 = r1 x r2
        r3 = np.cross(r1, r2)
        # t = 1/lambda * A^-1 * h3
        t = np.dot(np.linalg.inv(A), h3) / lamb
        # rotation and translation matrix for a given H matrix
        RT = np.vstack((r1, r2, r3, t)).T
        all_RT.append(RT)

    return all_RT
def extractParamFromA(init_A, init_kc):
    alpha = init_A[0,0]
    gamma = init_A[0,1]
    beta = init_A[1,1]
    u0 = init_A[0,2]
    v0 = init_A[1,2]
    k1 = init_kc[0,0]
    k2 = init_kc[1,0]

    x0 = np.array([alpha, gamma, beta, u0, v0, k1, k2])
    return x0

def lossFunc(x0, init_all_RT, all_image_corners, world_corners):
    # get A and kc from x0
    A, _ = retrieveA(x0)
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

def retrieveA(x0):
    # returns the A and distortion of a given x0
    alpha, gamma, beta, u0, v0, k1, k2 = x0
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]).reshape(3,3)
    kc = np.array([k1, k2]).reshape(2,1)
    return A, kc
def reprojectPointsAndGetError(A, kc, all_RT, all_image_corners, world_corners):
    # basically lossFunc again
    # without x0 as input
    error_mat = []
    _, _, _, u0, v0, k1, k2 = extractParamFromA(A, kc)
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

if __name__ == "__main__":
    main()
