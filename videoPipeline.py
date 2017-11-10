import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import numpy as np
import os
import params
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def debugDislay(x):
    if params.DEBUG_MODE:
        plt.imshow(x)
        plt.show()

def debugDislay2(x,y):
    if params.DEBUG_MODE:
        fig = plt.figure()
        plt.subplot(211)
        plt.imshow(x)
        plt.subplot(212)
        plt.imshow(y)
        plt.show()

def parseVideo(iFile,oFile):
    """filename= /path/to/file
    Reutrns
    """
    clip = VideoFileClip(iFile)
    new_clip = clip.fl_image(lambda x: frameTransform(x))
    new_clip.write_videofile(oFile,audio=False)

def parseTestImages(imgDir,resultDir):
    "Loops through test images"
    outputImages=[]
    for root,_,filenames in os.walk(imgDir):
        for img in filenames:
            image = cv2.imread(img)
            image_ = np.copy(image)
            outputImages.append(frameTransform(image_))
def frameTransform(x):

    #   grey,blur
    frame = np.copy(x)
    width = frame.shape[1]
    height = frame.shape[0]
    # Color filter (get segments
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2HLS)
    maxLum = np.max(frame[:,:int(height/2),1])
    select_yellow= (frame[:,:,0] < params.hls_YellowHigh_thresholds[0]) & (frame[:, :, 0] > params.hls_YellowLow_thresholds[0])\
                &  (frame[:,:,1] < params.hls_YellowHigh_thresholds[1]*maxLum) & (frame[:, :, 1] > params.hls_YellowLow_thresholds[1]*maxLum) \
                &  (frame[:,:,2] < params.hls_YellowHigh_thresholds[2]) & (frame[:, :, 2] > params.hls_YellowLow_thresholds[2])
    select_white = (frame[:, :, 0] < params.hls_WhiteHigh_thresholds[0]) & (
    frame[:, :, 0] > params.hls_WhiteLow_thresholds[0]) \
                    & (frame[:, :, 1] < params.hls_WhiteHigh_thresholds[1] * maxLum) & (
                    frame[:, :, 1] > params.hls_WhiteLow_thresholds[1] * maxLum) \
                    & (frame[:, :, 2] < params.hls_WhiteHigh_thresholds[2]) & (
                    frame[:, :, 2] > params.hls_WhiteLow_thresholds[2])

    color_drop = np.copy(frame)
    color_drop = cv2.cvtColor(color_drop,cv2.COLOR_HLS2RGB)

    color_drop[~select_yellow]=[0,0,0]
    color_drop[select_white] = [255, 255, 255]
    debugDislay(color_drop)

    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)


    shape = np.shape(frame)
    height = shape[0]
    width = shape[1]
    white = np.max(frame[int(height/2):,:])
    kernel = (params.gaussKernel,params.gaussKernel)

    frame = cv2.GaussianBlur(
        frame,
        kernel,
        params.gaussSigma)

    # Canny Edge Detection

    frame_canny = cv2.Canny(frame,
                      params.cannyLowThreshMul * white,
                      params.cannyLowThreshMul * white)
    frame = np.copy(frame_canny)

    # Mask
    mask_points = [[width * params.maskBaseX, params.maskBaseY * height], \
                   [width * params.maskHorizonX, params.maskHorizonY * height], \
                   [width - (width * params.maskHorizonX), params.maskHorizonY * height], \
                   [width - (width * params.maskBaseX), params.maskBaseY * height], \
                   ]
    numMaskPoints = len(mask_points)

    model_line = []
    for point_number in range(numMaskPoints):
        model_line.append(np.polyfit((mask_points[point_number][0], mask_points[(point_number + 1) % numMaskPoints][0]),
                                     (mask_points[point_number][1], mask_points[(point_number + 1) % numMaskPoints][1]),
                                     1
                                     ))

    # meshgrid
    XX , YY = np.meshgrid(np.arange(0,width),np.arange(0,height))

    select_space = (YY < (XX * model_line[0][0] + model_line[0][1])) \
                   | (YY < (XX * model_line[1][0] + model_line[1][1])) \
                   | (YY < (XX * model_line[2][0] + model_line[2][1])) \
                   | (YY > (XX * model_line[3][0] + model_line[3][1]))

    frame[select_space] = 0
    select_WY = select_white | select_yellow
    frame[~select_WY] = 0

    lines = cv2.HoughLinesP(frame,
                            params.houghLineResolution,
                            params.houghAngleResolution,
                            threshold=params.houghPointVotes,
                            lines=np.array([]),
                            minLineLength=params.houghLineMinLength,
                            maxLineGap=params.houghMaxLineGap)

    gradients = []
    intercept = []
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:

                if abs((y2-y1)/(x2-x1))>params.lineHighGradientLimit or \
                                abs((y2 - y1) / (x2 - x1)) < params.lineLowGradientLimit:
                    continue
                gradients.append((y2-y1)/(x2-x1))
                intercept.append(y1-gradients[-1]*x1)
        error_flag = False
        print(len(lines))
    except:
        plt.imshow(x)
        plt.show()
        plt.imshow(frame_canny)
        plt.show()
        plt.imshow(color_drop)
        plt.show()
        params.DEBUG_MODE = True
        error_flag= True
    # Average lines
    linePoints = np.asarray([gradients,intercept])
    linePoints.transpose()
    print(linePoints.shape)
    km=KMeans(2)
    km.fit(linePoints.transpose())
    centroids = km.cluster_centers_
    print(centroids)

    # Sample Averages : Top and bottom of visibility space
    average_lines = []
    lineImage = np.zeros_like(x)
    for i in range(len(centroids)):


        y1 = int(mask_points[i*3][1])
        y2 = int(mask_points[i+1][1])
        x1 = int((y1 - centroids[i][1])/centroids[i][0])
        x2 = int((y2 - centroids[i][1])/centroids[i][0])
        cv2.line(lineImage, (x2, y2), (x1, y1), (0, 127, 255), 3)
    lineImage[select_space] = [0,0,0]
    output = cv2.addWeighted(x,1,
                             lineImage,1,0)
    maskHighlighter = np.zeros_like(x)
    #maskHighlighter[~select_space] = [127,127,0]
    output = cv2.addWeighted(maskHighlighter,0.5,output,1,0)
    debugDislay2(output,frame_canny)
    if not error_flag:
        if (params.frame_number % 30) == 0:
            params.DEBUG_MODE = True
        else:
            params.DEBUG_MODE = False
        #   C

    # Mask
    # Hough Transform
    params.frame_number += 1
    return output

if __name__=="__main__":
    parseVideo('test.mp4','out.mp4')