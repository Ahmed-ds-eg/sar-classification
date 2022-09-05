from cProfile import label
from tkinter import font
from turtle import color
import streamlit as st
from streamlit_image_comparison import image_comparison
import base64
import sys 

# importing other files
sys.path.append('../')
import data_info
import models

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode() 

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def masking_image(data):
    st.write("")
    st.write('## Choose the Masking you want')
    options = st.multiselect(
    '',
    data["names"],
    )#default = data["names"][:2])
    img = data_info.mask(data, options)
    st.image(img)

def page_header():
    set_background(r'assets\background2.jpg')
    col1, mid, col2 = st.columns([1,4,20])
    with col1:
        st.image(
        "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/telescope_1f52d.png",
        width=120,
    )
    with col2:
        st.header("Classification of Polarimetric SAR Images using Compact Adaptive Convolutional Neural Networks", )
    st.write("")


# page title
st.set_page_config("Classification Polarsar Images", "🔭")

streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Cinzel', sans-serif;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)


radio = st.sidebar.radio("Choose the Dataset", ["Hello" ,"Flevo_l","Flevo_c","Sfbay_l", "Sfbay_c", "Classify"], )

if radio == "Hello":
    set_background(r'assets\background.png')
    st.write('')
    st.write('')
    st.write('')
    original_title = '<p style="font-family:inter; color:#cc8800; font-size: 50px; text-align:center">Classification of Polarimetric SAR </br> Images using Compact Adaptive Convolutional Neural Networks</p>'
    st.markdown(original_title, unsafe_allow_html=True)

    #st.write("# Classification of polarimetric SAR images using compact Adaptive convolutional neural networks", color = 'brown')
    
if radio == "Flevo_l":
    page_header()
    st.markdown("### Flevo_l")
    image_comparison(
        img1="pics/flevo_l.jpg",
        img2="img/file_flevo_l.jpg",
        label1="Before",
        label2="After",
    )
    st.image('labels/flevo_l_classes_colors.png')
    masking_image(data_info.flevo_l)
    

elif radio == "Flevo_c":
    page_header()
    st.markdown("### Flevo_c")
    image_comparison(
        img1="pics/flevo_c.jpg",
        img2="img/file_flevo_c.jpg",
        label1="Before",
        label2="After",
    )
    st.image('labels/flevo_c_classes_colors.png')
    masking_image(data_info.flevo_c)

elif radio == "Sfbay_l":
    page_header()
    st.markdown("### sfbay_l")
    image_comparison(
        img1="pics/sfbay_l.jpg",
        img2="img/file_sfbay_l.jpg",
        label1="Before",
        label2="After",
    )
    st.image('labels/sfbay_l_classes_colors.png')
    masking_image(data_info.sfbay_l)

elif radio == "Sfbay_c":
    page_header()
    st.markdown("### sfbay_c")
    image_comparison(
        img1="pics/sfbay_c.jpg",
        img2="img/file_sfbay_c.jpg",
        label1="Before",
        label2="After",
    )
    st.image('labels/sfbay_c_classes_colors.png')
    masking_image(data_info.sfbay_c)

elif radio == "Classify":
    page_header()
    st.markdown("### Classify")
    st.write(' ')

    if st.button('Flevo_l'):
        st.write('Classifying...')
        data, options, accuracy = models.classify('flevo_l')
        img = data_info.mask(data, options)
        st.image(img)
        st.write('## Test accuracy is: ', accuracy)
        
    if st.button('General'):
        st.write('Classifying...')
        data, options, accuracy = models.classify('general')
        img = data_info.mask(data, options)
        st.image(img)
        st.write('## Test accuracy is: ', accuracy)
        

