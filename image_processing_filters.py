import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def download(img,img_name):
    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    col1, spacer, col2 = st.columns([5,3,5])
    with spacer:
        st.download_button(
            label=". Download Image.",
            data=buffer,
            file_name=img_name,
            mime="image/jpeg"
        )

def view_old_new(img1,img2):
    col1, spacer, col2 = st.columns([10,1,10])
    with col1:
        st.image(img1, channels="BGR",use_column_width=True, caption="original image")
    with col2:
        st.image(img2, channels="BGR",use_column_width=True, caption="new image")

def apply(k_start, k_end, k_default):
    col1, spacer, col2 = st.columns([2,5,2])
    with spacer:
        kernal_size = st.slider('select kernel size', k_start, k_end, k_default, 2)
    col1, spacer, col2 = st.columns([5,2,5])
    with spacer:
        button=st.button('Show image')
    while(not button):
        pass
    return kernal_size



def lpf(img,k_size):
    # Get the selected kernel size from the slider
    kernel_size = k_size
    # Apply Gaussian blur to the grayscale image with the selected kernel size
    blurred_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) 
    return(blurred_image)

def hpf(img,a):
    # Get the selected kernel size from the slider
    kernel_size = a
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image with the selected kernel size
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    # Subtract the blurred image from the grayscale image to obtain high-pass filtered image
    hpf_image = cv2.subtract(gray_image, blurred_image)
    # Convert the high-pass filtered image to BGR format 
    return(cv2.cvtColor(hpf_image, cv2.COLOR_GRAY2BGR))

def mean_filter(img,a):
    # Get the selected kernel size from the slider
    kernel_size = a
    # Apply mean filter to the original image with the selected kernel size
    mean_image = cv2.blur(img, (kernel_size, kernel_size))
    return(mean_image)

def median_filter(img,a):
    # Get the selected kernel size from the slider
    kernel_size = a
    # Apply median filter to the original image with the selected kernel size
    median_image = cv2.medianBlur(img, kernel_size)
    return(median_image)

def roberts_edge_detector(img,min,max):
    # Apply Canny edge detection to the original image with fixed thresholds
    roberts_image = cv2.Canny(img, min, max)
    # Update the image display with the Roberts edge detected image
    return(roberts_image)

def prewitt_edge_detector(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray_image, (3,3), 0)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewitt_x = cv2.filter2D(img_gaussian, -1, kernelx)
    prewitt_y = cv2.filter2D(img_gaussian, -1, kernely)
    # Compute the magnitude of gradients
    prewitt_image = prewitt_x+prewitt_y
    return(prewitt_image)

def sobel_edge_detector(img,k_size):
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute the horizontal and vertical gradients using Sobel operators
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=k_size)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=k_size)
    # Compute the magnitude of gradients
    sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalize the gradient magnitude image
    sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Update the image display with the Sobel edge detected image
    return(sobel_image)

def erosion(img,k_size,iteration):
    # Get the selected kernel size from the slider
    kernel_size = k_size
    # Create a kernel for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform erosion on the original image
    erosion_image = cv2.erode(img, kernel, iterations=iteration)
    # Update the image display with the eroded image
    return(erosion_image)

def dilation(img,k_size,iteration):
    # Get the selected kernel size from the slider
    kernel_size = k_size
    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation on the original image
    dilation_image = cv2.dilate(img, kernel, iterations=iteration)
    # Update the image display with the dilated image
    return(dilation_image)

def open(img,a):
    # Get the selected kernel size from the slider
    kernel_size = a
    # Create a kernel for opening
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform opening on the original image
    open_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # Update the image display with the opened image
    return(open_image)

def close(img,a):
    # Get the selected kernel size from the slider
    kernel_size = a
    # Create a kernel for closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform closing on the original image
    close_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Update the image display with the closed image
    return(close_image)

def hough_circle_transform(img):
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect circles using Hough circle transform with specified parameters
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    # Check if any circles are detected
    if circles is not None:
        # Convert the circle parameters to integer
        circles = np.uint16(np.around(circles))
        # Create a copy of the original image for drawing circles
        hough_image = img.copy()
        # Draw detected circles on the image
        for i in circles[0, :10]:
            cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the outer circle
            cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)       # Draw the center of the circle
        # Update the image display with the circles drawn
        return(hough_image)

def thresholding_segmentation(img,thre):
    threshold_value =thre
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return(cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR))




def main():
    st.markdown("<h1 style='text-align: center;'>Image Processing Filters</h1>", unsafe_allow_html=True)

    link='''https://github.com/AhmedMOM3/Image-Processing-Filters'''
    st.write(f'#<a target="_blank" href="{link}">`github link`</a>', unsafe_allow_html=True)


    # uploading photo
    data = st.file_uploader('Upload your photo here in one of these formats (jpg, jpeg, png, webp)')

    #checkin if the file is valied or not
    x=False
    if data:
        if data.name.split('.')[-1] == 'jpg' or data.name.split('.')[-1] == 'jpeg' or data.name.split('.')[-1] == 'png' or data.name.split('.')[-1] == 'webp':
            file_bytes = np.asarray(bytearray(data.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.success('uploaded successfully')
            x=True
        else:
            st.error('please enter a valied photo')
            x=False
    st.write('---')
    
    
    if x:
        selected_filter_type=st.selectbox('Pick the filter you want to apply',['','LBF','HPF','Mean Filter','Median Filter',"Robert's Edge Detedtor",'Prewitt Edge Detedtor','Sobel Edge Detedtor','Erosion','Dilation','Openning','Closeing','Hough Transform for Circle','Segmentation using Thresholding'] )

        #LPF
        if selected_filter_type=='LBF':
            kernal_size=apply(3, 21, 7)    
            lpf_img=lpf(img,kernal_size)
            view_old_new(img,lpf_img)
            lpf_img=cv2.cvtColor(lpf_img,cv2.COLOR_BGR2RGB)
            download(lpf_img,'lpf_img.jpg')

        #HPF
        elif selected_filter_type=='HPF':
            kernal_size=apply(11, 81, 21)    
            hpf_img=hpf(img,kernal_size)
            view_old_new(img,hpf_img)
            download(hpf_img,'hpf_image.jpg')

        #Mean Filter
        elif selected_filter_type=='Mean Filter':
            kernal_size=apply(3, 81, 21)
            mean_filtered_img=mean_filter(img,kernal_size)
            view_old_new(img,mean_filtered_img)
            mean_filtered_img=cv2.cvtColor(mean_filtered_img,cv2.COLOR_BGR2RGB)
            download(mean_filtered_img,'mean_filtered_image.jpg')
        
        #median Filter
        elif selected_filter_type=='Median Filter':
            kernal_size=apply(3, 81, 21)    
            median_filtered_img=median_filter(img,kernal_size)
            view_old_new(img,median_filtered_img)
            median_filtered_img=cv2.cvtColor(median_filtered_img,cv2.COLOR_BGR2RGB)
            download(median_filtered_img,'median_filtered_img.jpg')

        #robert's edge detector
        elif selected_filter_type=="Robert's Edge Detedtor":
            x,y = st.slider('Select min and max thresholds values',0, 255, (100, 200))
            col1, spacer, col2 = st.columns([5,2,5])
            with spacer:
                button=st.button('Show image')
            while(not button):
                pass
            roberts_edge_detector_img=roberts_edge_detector(img,x,y)
            roberts_edge_detector_img=cv2.cvtColor(roberts_edge_detector_img,cv2.COLOR_GRAY2RGB)
            view_old_new(img,roberts_edge_detector_img)
            download(roberts_edge_detector_img,'roberts_edge_detector_img.jpg')

        #prewitt edge detector
        elif selected_filter_type=='Prewitt Edge Detedtor':
            col1, spacer, col2 = st.columns([5,2,5])
            with spacer:
                button=st.button('Show image')
            while(not button):
                pass
            prewitt_edge_detector_img=prewitt_edge_detector(img)
            col1, spacer, col2 = st.columns([10,1,10])
            with col1:
                st.image(img, channels="BGR",use_column_width=True, caption="original image")
            with col2:
                st.image(prewitt_edge_detector_img, use_column_width=True, caption="new image")
            
            prewitt_edge_detector_img=cv2.cvtColor(prewitt_edge_detector_img,cv2.COLOR_GRAY2RGB)
            download(prewitt_edge_detector_img,'prewitt_edge_detector_img.jpg')

        #sobel edge detector
        elif selected_filter_type=='Sobel Edge Detedtor':
            kernal_size=apply(3, 21, 7)    
            sobel_edge_detector_img=sobel_edge_detector(img,kernal_size)
            sobel_edge_detector_img=cv2.cvtColor(sobel_edge_detector_img,cv2.COLOR_GRAY2RGB)
            view_old_new(img,sobel_edge_detector_img)
            download(sobel_edge_detector_img,'sobel_edge_detector_img.jpg')

        #erosion filter
        elif selected_filter_type=='Erosion':   
            col1, spacer, col2 = st.columns([2,5,2])
            with spacer:
                iterations = st.slider('select num of iteration ', 1, 5, 1)
            kernal_size=apply(3, 21, 7)    
            erosion_img=erosion(img,kernal_size,iterations)
            view_old_new(img,erosion_img)
            erosion_img=cv2.cvtColor(erosion_img,cv2.COLOR_BGR2RGB)
            download(erosion_img,'erosion_img.jpg')

        #dilation filter
        elif selected_filter_type=='Dilation':    
            col1, spacer, col2 = st.columns([2,5,2])
            with spacer:
                iterations = st.slider('select num of iteration ', 1, 5, 1)
            kernal_size=apply(3, 21, 7)    
            dilate_img=dilation(img,kernal_size,iterations)
            view_old_new(img,dilate_img)
            dilate_img=cv2.cvtColor(dilate_img,cv2.COLOR_BGR2RGB)
            download(dilate_img,'dilate_img.jpg')

        #openning filter
        elif selected_filter_type=='Openning':
            kernal_size=apply(3, 21, 7)    
            openning_img=open(img,kernal_size)
            view_old_new(img,openning_img)
            openning_img=cv2.cvtColor(openning_img,cv2.COLOR_BGR2RGB)
            download(openning_img,'openning_img.jpg')

        #clossing filter
        elif selected_filter_type=='Closeing':
            kernal_size=apply(3, 21, 7)    
            closeing_img=close(img,kernal_size)
            view_old_new(img,closeing_img)
            closeing_img=cv2.cvtColor(closeing_img,cv2.COLOR_BGR2RGB)
            download(closeing_img,'closeing_img.jpg')

        #hough transform for Circle
        elif selected_filter_type=='Hough Transform for Circle':
            col1, spacer, col2 = st.columns([5,2,5])
            with spacer:
                button=st.button('Show image')
            while(not button):
                pass
            hough_circle_transform_img=hough_circle_transform(img)
            view_old_new(img,hough_circle_transform_img)
            hough_circle_transform_img=cv2.cvtColor(hough_circle_transform_img,cv2.COLOR_BGR2RGB)
            download(hough_circle_transform_img,'hough_circle_transform_img.jpg')

        #segmentation using thresholding
        elif selected_filter_type=='Segmentation using Thresholding':
            thre=apply(0, 255, 127)    
            thresholding_segmentation_img=thresholding_segmentation(img,thre)
            view_old_new(img,thresholding_segmentation_img)
            download(thresholding_segmentation_img,'thresholding_segmentation_img.jpg')



if __name__ == "__main__":
    main()
