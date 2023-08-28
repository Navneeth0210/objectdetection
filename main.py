import cv2 as cv
import numpy as np
import utlis

path = 'image.jpeg'
img = cv.imread(path)
scale = 3
width = 210 * scale
height = 297 * scale

img = cv.resize(img, (0, 0), None, 0.5, 0.5)
img, contours = utlis.getContours(img, minArea=50000, filter=4)
if len(contours) != 0:
    biggest = contours[0][2]
    print(biggest)
    img_warp_result = utlis.wrapimg(img, biggest, width, height)
    cv.imshow('a4', img_warp_result)

    img2, contours2 = utlis.getContours(img_warp_result,showcanny=True, minArea=2800, filter=4, cThr=[10, 10], draw=False)

    if len(contours)!=0:
        for obj in contours2:
            cv.polylines(img2,[obj[2]], True, (0,0,255),2)
            npoints = utlis.reorder(obj[2])
            nw=utlis.findDis(npoints[0][0]//scale, npoints[1][0]//scale)
            nh=utlis.findDis(npoints[0][0]//scale, npoints[2][0]//scale)
            cv.putText(
                img2,
                f'Width: {nw:.2f} mm',
                tuple(npoints[0][0]),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv.putText(
                img2,
                f'Height: {nh:.2f} mm',
                tuple(npoints[2][0]),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
    cv.imshow('object', img2)

cv.imshow('output', img)

cv.waitKey(0)
