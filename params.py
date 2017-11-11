from numpy import pi
frame_number =0
DEBUG_MODE = False
DISPLAY_DATA = False

# Color Select
hls_YellowLow_thresholds = [13, 0.2, 100]
hls_YellowHigh_thresholds = [40, 1, 256]

hls_WhiteLow_thresholds = [0, 0.9, 0]
hls_WhiteHigh_thresholds = [255, 10, 256]

# Noise Filtering
gaussKernel= 21
gaussSigma= 3

# Canny
cannyLowThreshMul = 0.1
cannyHighThreshMul = 0.15

# Mask
maskHorizonY = 0.64
maskHorizonX = 0.43
maskBaseY = 0.95
maskBaseX = 0.1

# Hough
houghLineResolution = 3
houghAngleResolution = 3*(pi/180)
houghPointVotes = 40
houghLineMinLength = 30
houghMaxLineGap = 100
lineHighGradientLimit = 0.7
lineLowGradientLimit = 0.3

# Show line
lineHorizonScalar = 1.1

# Temporal Smoothing
smoothing_init = True
previousLineParams = [[0.1,0.1],[0.1,0.1]]
