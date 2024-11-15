import cv2
import os
import numpy as np
import streamlit as st

# Function to load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    return model

# Function to process the image
def process_image(image, model):
    classIds, scores, boxes = model.detect(image, confThreshold=0.6, nmsThreshold=0.4)
    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)
    return image

# Main function to run the Streamlit app
def main():
    st.title("Pothole Detection using YOLO")
    
    # Load the YOLO model
    model = load_yolo_model()
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Process the image
        result_img = process_image(img, model)
        
        # Display the result
        st.image(result_img, channels="BGR", caption="Processed Image")
        
        # Save the result
        cv2.imwrite("result1.jpg", result_img)  # Save the result image

if __name__ == "__main__":
    main()