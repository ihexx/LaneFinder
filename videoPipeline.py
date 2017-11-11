import argparse
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
    if params.DISPLAY_DATA and params.DEBUG_MODE:
        plt.imshow(x)
        plt.show()

def debugDislay2(x,y):
    if params.DISPLAY_DATA and params.DEBUG_MODE:
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
    frame = np.copy(x)
    width = frame.shape[1]
    height = frame.shape[0]

    # Color filter (get segments)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2HLS) #To HLS so we can select yellow hue and white luminosity easily
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

    select_WY = select_white | select_yellow
    color_drop[~select_WY]=[0,0,0]

    # Noise Removal
    frame = cv2.cvtColor(color_drop,cv2.COLOR_RGB2GRAY)
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

    frame_canny[select_space] = 0
    debugDislay2(frame, frame_canny)

    # Hough Transform
    lines = cv2.HoughLinesP(frame_canny,
                            params.houghLineResolution,
                            params.houghAngleResolution,
                            threshold=params.houghPointVotes,
                            lines=np.array([]),
                            minLineLength=params.houghLineMinLength,
                            maxLineGap=params.houghMaxLineGap)
    if params.DEBUG_MODE:
        print(len(lines))
        if len(lines)!=0:
            lineImage = np.zeros_like(x)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(lineImage, (x2, y2), (x1, y1), (0, 0, 255), 3)
                    showHough = cv2.addWeighted(np.dstack((frame,frame,frame)), 1,lineImage, 1, 0)
            debugDislay2(showHough,frame_canny)
    gradients = []
    intercept = []
    numLines = 0
    for line in lines:
        for x1,y1,x2,y2 in line:

            if abs((y2-y1)/(x2-x1))>params.lineHighGradientLimit or \
                            abs((y2 - y1) / (x2 - x1)) < params.lineLowGradientLimit:
                continue
            gradients.append((y2-y1)/(x2-x1))
            intercept.append(y1-gradients[-1]*x1)
            numLines+=1
    error_flag = False


    # Average lines
    lineParams = params.previousLineParams

    if not numLines<2:
        linePoints = np.asarray([gradients,intercept])
        km=KMeans(2)
        km.fit(linePoints.transpose())
        centroids = km.cluster_centers_


        for centroid in centroids:
            if params.smoothing_init:
                if centroid[0] <0:
                    lineParams[1] = centroid
                    params.previousLeft = centroid
                else:
                    lineParams[0] = centroid
                    params.previousRight = centroid
            else:
                if centroid[0] <0:
                    lineParams[1] = np.mean(np.dstack((centroid,np.asarray(params.previousLineParams[1]))),axis=2)
                    params.previousLineParams[1] = centroid
                else:
                    lineParams[0] = np.mean(np.dstack((centroid,np.asarray(params.previousLineParams[0]))),axis=2)
                    params.previousLineParams[0]= centroid



    # Sample Averages : Top and bottom of visibility space
    average_lines = []

    lineImage = np.zeros_like(x)
    for i in range(len(lineParams)):


        y1 = int(mask_points[i*3][1]*params.lineHorizonScalar)
        y2 = int(mask_points[i+1][1]*params.lineHorizonScalar)
        x1 = int((y1 - lineParams[i][1])/lineParams[i][0])
        x2 = int((y2 - lineParams[i][1])/lineParams[i][0])
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
            params.DISPLAY_DATA = True
        else:
            params.DISPLAY_DATA = False
        #   C

    # Mask
    # Hough Transform
    params.frame_number += 1
    return output


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process Video pipeline and print results.')
    parser.add_argument('--input', '-i', type=str,
                        help='Input video path',default='test_videos/solidWhiteRight.mp4')
    parser.add_argument('--output', '-o', type=str,
                        help='Output video path', default='output_videos/out.mp4')
    args = parser.parse_args()
    print(args.input)

    parseVideo(args.input,args.output)