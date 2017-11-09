from numpy import pi
frame_number =0
DEBUG_MODE = True

hls_YellowLow_thresholds = [10, 0.2, 100]
hls_YellowHigh_thresholds = [40, 1, 256]

hls_WhiteLow_thresholds = [0, 0.9, 0]
hls_WhiteHigh_thresholds = [255, 10, 256]

gaussKernel= 7
gaussSigma= 2

cannyLowThreshMul = 0.275
cannyHighThreshMul = 0.3

maskHorizonY = 0.63
maskSubHorizonY = 0.7
maskHorizonX = 0.48
maskSubHorizonX = 0.4
maskBaseY = 0.9
maskBaseX = 0


houghLineResolution = 60
houghAngleResolution = 30*(pi/180)
houghPointVotes = 150
houghLineMinLength = 30
houghMaxLineGap = 100
lineHighGradientLimit = 0.7
lineLowGradientLimit = 0.3