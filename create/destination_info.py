import streamlit as st
import requests
from bs4 import BeautifulSoup
import random

def fetch_destination_image(destination_name):
    """
    Fetch an image URL for a destination using Unsplash API
    """
    try:
        # Using Unsplash source for images
        return f"https://source.unsplash.com/800x600/?{destination_name.replace(' ', '%20')},travel,landmark"
    except Exception as e:
        # Fallback image if API fails
        return "https://images.unsplash.com/photo-1488646953014-85cb44e25828?q=80&w=1000"

def get_destination_description(location_name):
    """Generate a description for a destination (simplified mock function)"""
    descriptions = {
        "Bhubaneswar": "Bhubaneswar, the capital of Odisha, is known as the 'Temple City of India' with over 700 ancient temples. The city beautifully blends its rich historical heritage with modern urban planning.",
        "Delhi": "Delhi, India's capital territory, is a massive metropolitan area and the seat of India's government. The city is known for its historical sites, diverse culture, and bustling markets.",
        "Mathura": "Mathura is a sacred city in Uttar Pradesh, known as the birthplace of Lord Krishna. The city is filled with temples and is a major pilgrimage site along the banks of the Yamuna River.",
        "Agra": "Agra is home to the iconic Taj Mahal, a stunning white marble mausoleum that attracts millions of visitors. The city also features Agra Fort and Fatehpur Sikri, both UNESCO World Heritage sites.",
        "PrayagRaj": "PrayagRaj (formerly Allahabad) is known for the Triveni Sangam, the confluence of three sacred rivers: Ganga, Yamuna, and Saraswati. It hosts the Kumbh Mela, the world's largest religious gathering.",
        "Patna": "Patna, the capital of Bihar, is one of the oldest continuously inhabited places in the world. The city has a rich historical heritage dating back to the ancient kingdoms of Magadha.",
        "San Francisco": "San Francisco is known for its iconic Golden Gate Bridge, cable cars, colorful Victorian houses, and diverse neighborhoods. The city's position on the water provides stunning views.",
        "New York": "New York City comprises 5 boroughs where the Hudson River meets the Atlantic Ocean. Manhattan has skyscrapers like the Empire State Building and the vast Central Park."
    }
    
    # Return description if available, otherwise generate a generic one
    if location_name in descriptions:
        return descriptions[location_name]
    else:
        return f"{location_name} is a beautiful destination with unique attractions and cultural experiences. Visitors can explore local landmarks, sample regional cuisine, and immerse themselves in the local atmosphere."

def display_destination_info(location_name):
    """Display information and image about a destination"""
    # Generate a relevant image URL for the location
    image_url = f"https://source.unsplash.com/300x200/?{location_name},travel"
    
    # Display the image
    st.image(image_url, caption=location_name, use_container_width=True)  # Updated from use_column_width
    
    # Generate or fetch a brief description
    description = get_destination_description(location_name)
    
    # Display the description
    st.write(description)

def display_multi_destination_info(location_names):
    """Display information about multiple destinations in columns"""
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
                    display_destination_info(location_names[i + j])
                    