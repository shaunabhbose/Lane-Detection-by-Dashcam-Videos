"""
Course Number: ENGR 13300
Semester: e.g. Spring 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     18.4 IP
    Team ID:        LC1 - 13
    Author:         Shaunabh Bose, bose45@purdue.edu
    Date:           12/7/25  

Contributors:
    name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""
import cv2
import numpy as np
import ip_helper_python_file
# Full preprocessing pipeline for edge extraction.
def prepare_edges(frame):
    prep = ip_helper_python_file.preprocess_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blr = cv2.GaussianBlur(gray, (5, 5), 0) #reduce brightness noise
    edges = cv2.Canny(blr, 50, 150) #detect edges
    norm = ip_helper_python_file.normalize_edges(edges) #normalize edges
    smoothed = ip_helper_python_file.extra_smoothing(norm) #final smoothing pass
    return smoothed

# Main lane detection pipeline that processes every frame of the input video.
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path) #open video file
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # get frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # get frame height
    fps = int(cap.get(cv2.CAP_PROP_FPS))                 # get frames per second

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')             # codec for MP4 files
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (width, height))             

    while True:
        frame, ret = ip_helper_python_file.safe_read(cap)              
        if not ret:
            break

        _ = ip_helper_python_file.diagnostic(frame)                          

        prepared = prepare_edges(frame)               
        cropped = ip_helper_python_file.region_of_interest(prepared)     
        lines = ip_helper_python_file.detect_lines(cropped)                 
        final_frame = ip_helper_python_file.draw_lines(frame, lines)        

        out.write(final_frame)                

    cap.release()
    out.release()                                        
    print("Processing complete. Output saved:", output_path)

# Main function to prompt for user input and start processing.
def main():
    path = input("Input Video file path: ")
    while True:
        out = input("Output Video file path(words only): ")
        if(out.isalpha()):
            break
        print("Output Filename Contains numbers")
    out = out + ".mp4"
    process_video(path, out)

if __name__ == "__main__":
    main()
