import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(10,150)
width,height =640,480
def prprocess_img(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(imgGray,200,200)
    kernel = np.ones((4,4))
    img_dil = cv2.dilate(img_canny,kernel,iterations=2)
    img_erode = cv2.erode(img_dil,kernel,iterations=1)


    return img_erode


def get_contour(img):
    maxarea =0
    biggest = np.array([])
    contour,heirarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area >5000:
            peri = cv2.arcLength(cnt,True)
            aprox = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area>maxarea and len(aprox) == 4:
                biggest = aprox
                maxarea = area
    cv2.drawContours(img_cont, biggest, -1, (0, 0, 255), 20)
    return biggest

def reorder(mypoint):
    mypoint=mypoint.reshape((4,2))
    mypoint_new = np.zeros((4,1,2),np.int32)
    add = mypoint.sum(1)
    mypoint_new[0] = mypoint[np.argmin(add)]
    mypoint_new[3] = mypoint[np.argmax(add)]
    diff = np.diff(mypoint,axis=1)
    mypoint_new[1] = mypoint[np.argmin(diff)]
    mypoint_new[2] = mypoint[np.argmax(diff)]
    return mypoint_new


def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts_1 = np.float32(biggest)
    pts_2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts_1,pts_2)
    img_warp = cv2.warpPerspective(img,matrix,(width,height))
    return img_warp

while True:
    success,img = cam.read()
    cv2.resize(img,(width,height))
    img_cont = img.copy()
    img_erode = prprocess_img(img)
    biggest = get_contour(img_erode)

    if biggest.size!=0:
        imgwarp = getWarp(img,biggest)
        cv2.imshow("image", imgwarp)
    else:
        cv2.imshow("original",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break