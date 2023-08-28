import cv2 as cv
import numpy as np

def getContours(img, cThr= [100,100], showcanny=False, minArea= 1000, filter = 0, draw =False):
    imgGrey= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur= cv.GaussianBlur(imgGrey,(5,5),1)
    imgCanny = cv.Canny(imgBlur,cThr[0],cThr[1])
    kernel= np.ones((5,5))
    imgdial = cv.dilate(imgCanny, kernel, iterations=3 )
    imgthres = cv.erode(imgdial, kernel, iterations=2)
    if showcanny:cv.imshow('canny',imgthres)

    contours, hiearchy =  cv.findContours(imgthres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    finalcontours = []
    for i in contours:
        area = cv.contourArea(i)
        if area > minArea:
            peri = cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,0.02*peri, True)
            bbox= cv.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    finalcontours.append((len(approx), area, approx, bbox, i))
            else:
                finalcontours.append((len(approx), area, approx, bbox, i))


    finalcontours= sorted(finalcontours, key = lambda x:x[1], reverse= True)
    if draw:
        for con in finalcontours:
            cv.drawContours(img,con[4],-1 ,(0,255,0),3)

    return img, finalcontours

def reorder(mypoints):
    print(mypoints.shape)
    mypointsnew = np.zeros_like(mypoints)
    mypoints= mypoints.reshape((4,2))
    add = mypoints.sum(1)
    mypointsnew[0] = mypoints[np.argmin(add)]
    mypointsnew[3] = mypoints[np.argmax(add)]
    diff= np.diff(mypoints, axis= 1)
    mypointsnew[1] = mypoints[np.argmin(diff)]
    mypointsnew[2] = mypoints[np.argmax(diff)]

    return mypointsnew

def wrapimg(img,points,w,h,pad=20):
    print(points)
    points=reorder(points)

    pts1= np.float32(points)
    pts2= np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix= cv.getPerspectiveTransform(pts1,pts2)
    imgwrap = cv.warpPerspective(img,matrix,(w,h))
    imgwrap = imgwrap[pad:imgwrap.shape[0]-pad,pad:imgwrap.shape[1]-pad]

    return imgwrap

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
