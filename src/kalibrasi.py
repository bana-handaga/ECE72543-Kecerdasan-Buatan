import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(50)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Menghitung faktor kesalahan 

tot_error = 0

# for i in xrange(len(objpoints)):  #versi 2.7
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

# print "mean error: ", tot_error/len(objpoints) versi 2.7
print("mean error: %5.2f" % ( 100*tot_error/len(objpoints)) )

# Menyimpan parameter hasil kalibrasi
np.savez_compressed('B', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs )




print("Camera Matrix:")
print(mtx)
print("-----------------------------------")
print("Distortion Coefficient")
print("k1={}, k2={}, p1={}, p2={}, k3={}".format(dist[0,0],dist[0,1],dist[0,2],dist[0,3],dist[0,4] ) )

# menyimpan variabel kamera: camera matrix (mtx), distortion coefficient (dist)
# print ( dist )
