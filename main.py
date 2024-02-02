import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read and preprocess the image
def read_and_preprocess_image(temp_img_path):
    # Convert BGR to RGB
    img = cv2.imread(temp_img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply Gaussian Blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    
    return img, img_gray


# Function to display histogram of the grayscale image
def image_analysis(img_gray):
    # Display histogram of the grayscale image
    fig, ax = plt.subplots()
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    ax.plot(hist)
    st.pyplot(fig)


# Function to analyze cell counting based on specified parameters
def analyze_cell_counting(img, img_gray, morphology_method, threshold_value, display_numbers):
    _, th = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    
    # Apply morphological transformations
    if morphology_method == "closing":
       morphology = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    if morphology_method == "opening":
       morphology = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    if morphology_method == "opening followed by closing":
       morphology = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
       morphology = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel)
    if morphology_method == "closing followed by opening":
       morphology = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
       morphology = cv2.morphologyEx(morphology, cv2.MORPH_CLOSE, kernel)

    # Find contours in the processed image
    contours, _ = cv2.findContours(morphology, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    img_with_numbers = img.copy()  # Create a copy to avoid modifying the original image

    for i, cnt in enumerate(contours):
        cnt = cnt.squeeze(axis=1)
        if display_numbers:
           cv2.putText(img_with_numbers, f'{i+1}', (cnt[0][0], cnt[0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    
    img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    cell_count = len(contours)

    return img, morphology, img_contour, cell_count, img_with_numbers


# Main function for the Streamlit app
def main():
    st.title("Simple Cellcounting App")

    uploaded_file = st.file_uploader("Upload a cell image (JPG only)", type=["jpg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Save the uploaded file to a temporary location
        temp_img_path = "temp_image.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img, img_gray = read_and_preprocess_image(temp_img_path)

        st.image(img_gray, caption="Grayscale image.", use_column_width=True)

        image_analysis(img_gray)

        threshold_value = st.text_input("Threshold Value: ", 150)

        morphology_method = st.radio(label='Morphological Transformations',
                 options=('closing', 'opening', "closing followed by opening", "opening followed by closing"),
                 index=0,
                 horizontal=True,
                )
        
        display_numbers = st.checkbox('Display Numbers')
     
        if st.button("Analyze"):
            st.subheader("Analysis Results")

            img, morphology, img_contour, cell_count, img_with_numbers = analyze_cell_counting(img, img_gray, morphology_method, int(threshold_value), display_numbers)

            st.image(morphology, caption="Image used for Contour Extraction", use_column_width=True)
            
            if display_numbers:
                st.image(img_with_numbers, caption="Numbered Image", use_column_width=True)
            else:
                st.image(img_contour, caption="Draw the outline on the source image", use_column_width=True)
            
            st.markdown(f"Cell Count: {cell_count}")

if __name__ == "__main__":
    main()
