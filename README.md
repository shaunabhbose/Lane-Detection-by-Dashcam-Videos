## Lane Detection Using Computer Vision (Python + OpenCV)

This project implements a real-time **lane detection system** using Python and OpenCV, designed to demonstrate core perception techniques used in autonomous driving and ADAS systems. The program processes driving footage frame-by-frame to detect road lane boundaries and overlays them onto the original video.

### Key Features

* Uses **Canny edge detection** and **Gaussian blurring** for robust edge extraction
* Applies a **region-of-interest mask** to isolate the roadway
* Detects lane lines using the **Probabilistic Hough Transform**
* Outputs an annotated MP4 video with detected lane lines highlighted in green

### Inputs & Outputs

* **Input:** Path to a dashcam-style driving video
* **Output:** Processed video showing detected lane boundaries

### Why It Matters

Lane detection is a foundational problem in autonomous vehicle perception. This project provides an educational, modular implementation that illustrates how raw visual data is transformed into structured roadway information in real time.


## For additional information, please refer to the report and/or the code in .github
