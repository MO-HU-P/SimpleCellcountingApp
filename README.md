# Simple Cellcounting App

This Python application utilizes Streamlit for building a web-based interface to analyze cell density in images. The app allows users to upload a cell image in JPG format, preprocesses it, and performs cell counting using morphological transformations.

## Features

1.　Image Preprocessing: The uploaded image undergoes preprocessing, including conversion from BGR to RGB, Gaussian blur, and conversion to grayscale.

2.　Histogram Analysis: Displays the histogram of the grayscale image to provide insights into pixel intensity distribution.

3.　Cell Density Analysis: Allows users to analyze cell density using different morphological transformations and a user-defined threshold value for binary image conversion.

4.　Visual Results: Displays the original image, binary image after morphological transformation, and the source image with drawn cell contours. Additionally, presents the analysis results, including the number of detected cells.

## How to Use

1.　Upload a cell image in JPG format.

2.　Adjust the threshold value and choose a morphological transformation (closing or opening).

3.　Click the "Analyze" button to see the results.

4.　Analyzed results include images for contour extraction and the source image with drawn cell outlines, along with cell confluency percentage and cell count.
