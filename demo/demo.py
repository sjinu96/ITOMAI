import pandas as pd
from PIL import Image
import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from utils import convert_rgb_to_label

label_key_dict = {8: 'boat', 14: 'bench', 96:'bush', 105: 'clouds',  110: 'dirt', 112: 'fence', 118: 'flower', 119: 'fog', 123: 'grass', 
124: 'gravel', 125: 'ground_other', 126: 'hill', 127: 'house', 128: 'leaves',
 134: 'mountain', 135: 'mud', 139: 'pavement', 141: 'plant_other', 147: 'river', 148: 'road', 
149: 'rock', 153: 'sand', 154: 'sea', 156: 'sky_other', 158: 'snow', 161: 'stone', 168: 'tree', 177: 'water_other',  181: 'wood'}
key_label_dict = {label_key_dict[k]:k for k in label_key_dict}


def main():
    st.title("Interactive Tools for Object Manipulation in Art Images")

    uploded_file = st.file_uploader("Upload the image you want to manipulate", type=["jpg", "jpeg", "png"])
    if uploded_file:
        device = "cuda" if torch.cuda.is_available() else "cpu"


        # Specify canvas parameters in application
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 75, 50)
        
        label = st.sidebar.selectbox("Choose a semantic label.", options = key_label_dict.keys())
        if label == "boat":
            stroke_color = st.sidebar.color_picker("boat", value = "#000000")
        elif label == "bench":
            stroke_color = st.sidebar.color_picker("bench", value = "#FF0600")
        elif label == "fence":
            stroke_color = st.sidebar.color_picker("fence", value = '#7D0200')
        elif label == "house":
            stroke_color = st.sidebar.color_picker("house", value = '#DFFF00')
        elif label == "pavement":
            stroke_color = st.sidebar.color_picker("pavement", value = '#7F7F7F')
        elif label == "road":
            stroke_color = st.sidebar.color_picker("road", value = '#AF4BD8')
        elif label == "bush":    
            stroke_color = st.sidebar.color_picker("bush", value = '#6AA450')
        elif label == "clouds":
            stroke_color = st.sidebar.color_picker("clouds", value = '#19F7FF')
        elif label == "dirt":
            stroke_color = st.sidebar.color_picker("dirt", value = '#CBB138')
        elif label == "flower":
            stroke_color = st.sidebar.color_picker("flower", value = '#D86D6D')
        elif label == "fog":    
            stroke_color = st.sidebar.color_picker("fog", value = '#3A4481')
        elif label == "grass":    
            stroke_color = st.sidebar.color_picker("grass", value = '#17AA17')
        elif label == "gravel":    
            stroke_color = st.sidebar.color_picker("gravel", value = '#751A23')
        elif label == "ground_other":    
            stroke_color = st.sidebar.color_picker("ground_other", value = '#C3C074')
        elif label == "hill":    
            stroke_color = st.sidebar.color_picker("hill", value = '#5FD87D')
        elif label == "leaves":    
            stroke_color = st.sidebar.color_picker("leaves", value = '#00B92F')
        elif label == "mountain":    
            stroke_color = st.sidebar.color_picker("mountain", value = '#2CEC91')
        elif label == "mud":    
            stroke_color = st.sidebar.color_picker("mud", value = '#965103')
        elif label == "plant_other":    
            stroke_color = st.sidebar.color_picker("plant_other", value = '#618141')
        elif label == "river":
            stroke_color = st.sidebar.color_picker("river", value = '#5F62B9')
        elif label == "rock":
            stroke_color = st.sidebar.color_picker("rock", value = '#5C4114')
        elif label == "sand":
            stroke_color = st.sidebar.color_picker("sand", value = '#E6C691')
        elif label == "sea":
            stroke_color = st.sidebar.color_picker("sea", value = '#362CF7')
        elif label == "sky_other":
            stroke_color = st.sidebar.color_picker("sky_other", value = '#3EE8F5')
        elif label == "snow":
            stroke_color = st.sidebar.color_picker("snow", value = '#D0D9E2')
        elif label == "tree":
            stroke_color = st.sidebar.color_picker("tree", value = '#1A691A')
        elif label == "water_other":
            stroke_color = st.sidebar.color_picker("water_other", value = '#4A7A81')
        elif label == "wood":
            stroke_color = st.sidebar.color_picker("wood", value = '#AF5E22')

        
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=Image.open(uploded_file),
            update_streamlit=True,
            width = 512,
            height = 512,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("Generate Image"):
            # 여기에 segmentation model 들어갈 것!!!
            origin_label_map = np.zeros((512, 512, 1))
            user_maked_label = canvas_result.image_data[:, :, :3]
            user_maked_label = convert_rgb_to_label(user_maked_label, origin_label_map)
            print(user_maked_label)
            


if __name__ == "__main__":
    main()
