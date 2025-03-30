import streamlit as st
import os
import sys
from dotenv import load_dotenv
import time
from datetime import datetime
import pandas as pd
import random
import google.generativeai as genai

# Add path to modules
sys.path.append('create')
sys.path.append('mannual')

# Import component modules
from route_plannar import generate_route_options
from itinerary_generator import generate_itinerary
from utils import load_lottie, display_lottie
from destination_info import display_destination_info, display_multi_destination_info
from booking_system import generate_flight_options, generate_train_options, generate_bus_options, generate_cab_options, generate_hotel_options
from payment_processor import display_payment_methods, process_payment, display_payment_summary

# Load environment variables
load_dotenv()

# Setup Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Cache for storing destination descriptions
if 'destination_descriptions' not in st.session_state:
    st.session_state.destination_descriptions = {}

def get_gemini_destination_description(location_name):
    """
    Get a unique description for a destination using Google Gemini API
    """
    # Check if we already have a cached description
    if location_name in st.session_state.destination_descriptions:
        return st.session_state.destination_descriptions[location_name]
    
    try:
        # Initialize the Gemini model
        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 1024,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config
        )
        
        # Craft a detailed prompt to get rich, unique descriptions
        prompt = f"""
        Create a rich, engaging, and unique description of {location_name} as a travel destination.
        Include:
        1. What makes it special or famous
        2. Key landmarks or attractions visitors should see
        3. Cultural or historical significance
        4. Local cuisine worth trying
        5. Best time to visit
        
        Keep the description between 100-150 words, making it informative yet concise for travelers.
        Focus on creating a vivid picture that captures the essence and unique character of {location_name}.
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        description = response.text.strip()
        
        # Cache the description
        st.session_state.destination_descriptions[location_name] = description
        
        return description
    except Exception as e:
        st.error(f"Error generating description for {location_name}: {e}")
        return f"{location_name} is a fascinating destination with unique attractions and cultural experiences. Visitors can explore local landmarks, sample regional cuisine, and immerse themselves in the local atmosphere."

# Modify this function to use Gemini for descriptions
def display_destination_with_gemini(location_name):
    """Display destination with Gemini-generated description"""
    from destination_info import fetch_destination_image
    
    # Fetch a relevant image URL for the location
    image_url = fetch_destination_image(location_name)
    
    # Display the image
    st.image(image_url, caption=location_name, use_container_width=True)
    
    # Generate description using Gemini
    description = get_gemini_destination_description(location_name)
    
    # Display the description
    st.write(description)

# Display multiple destinations with Gemini descriptions
def display_multi_destinations_with_gemini(location_names):
    """Display information about multiple destinations with Gemini descriptions"""
    # Calculate how many locations per row (3 max)
    cols_per_row = 3
    num_locations = len(location_names)
    
    # Display destinations in rows of 3 columns
    for i in range(0, num_locations, cols_per_row):
        # Create columns for this row
        cols = st.columns(min(cols_per_row, num_locations - i))
        
        # Fill each column with destination info
        for j, col in enumerate(cols):
            if i + j < num_locations:
                with col:
                    display_destination_with_gemini(location_names[i + j])