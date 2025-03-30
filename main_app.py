import streamlit as st
import os
import sys
from dotenv import load_dotenv
import time
from datetime import datetime
import pandas as pd
import random

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

# Check for API key
if os.getenv("GOOGLE_API_KEY") is None:
    st.error("GOOGLE_API_KEY is not set in the .env file")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'journey_legs' not in st.session_state:
    st.session_state.journey_legs = []

if 'booking_steps_completed' not in st.session_state:
    st.session_state.booking_steps_completed = {}

# Add this helper function here, before the main app code:
def handle_hotel_booking(location_idx, hotel_data, payment_method, price):
    """Process hotel booking and update session state"""
    # Store transaction in session state
    if "payments" not in st.session_state:
        st.session_state.payments = []
        
    # Add payment details
    transaction_id = f"TX-{random.randint(1000000, 9999999)}"
    st.session_state.payments.append({
        "transaction_id": transaction_id,
        "amount": price,
        "booking_type": f"Hotel: {hotel_data['name']} in {hotel_data['location']}",
        "payment_method": payment_method,
        "status": "Successful",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Mark this hotel as booked
    acc_name = f"Hotel in {hotel_data['location']}"
    st.session_state.booking_steps_completed[acc_name] = True
    
    # Store confirmation in session state
    st.session_state[f"hotel_{location_idx}_booked"] = True
    st.session_state[f"hotel_{location_idx}_confirmation"] = {
        "hotel_name": hotel_data['name'],
        "location": hotel_data['location'],
        "price": price,
        "transaction_id": transaction_id
    }
    
    return transaction_id

# App title and description
st.title("✈️ Smart Travel Planner")
st.markdown("""
This AI-powered tool helps you plan your perfect trip by:
1. Generating multiple route options between your start and destination
2. Creating a detailed day-by-day itinerary based on your selected route
3. Booking your transportation, accommodations, and processing payments
""")

# Sidebar for basic information
with st.sidebar:
    st.header("Trip Details")
    start_location = st.text_input("Starting Location", placeholder="e.g., Bhubaneswar")
    destination = st.text_input("Destination", placeholder="e.g., Delhi")
    budget = st.text_input("Budget (INR)", "₹200")
    duration = st.text_input("Duration (days)", placeholder="e.g., 7")
    places = st.text_input("Places to visit (comma-separated)", placeholder="e.g., Mathura, Agra, PrayagRaj, Patna")

    # Show loading animation in sidebar
    lottie_travel = load_lottie("https://assets1.lottiefiles.com/packages/lf20_UgZWvP.json")
    st_lottie_container = st.empty()
    
    # Display booking progress
    if 'booking_steps_completed' in st.session_state and st.session_state.booking_steps_completed:
        st.markdown("---")
        st.subheader("Booking Progress")
        for step, status in st.session_state.booking_steps_completed.items():
            icon = "✅" if status else "⬜"
            st.markdown(f"{icon} {step}")

# Main workflow
tab1, tab2, tab3 = st.tabs(["Route Planning", "Detailed Itinerary", "Booking & Reservations"])

with tab1:
    st.header("Step 1: Choose Your Route")
    st.markdown("First, let's explore different ways to reach your destination.")
    
    if start_location and destination:
        # Display destination information at the top of the tab
        st.subheader("About Your Destinations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"##### Starting Point: {start_location}")
            display_destination_info(start_location)
            
        with col2:
            st.markdown(f"##### Final Destination: {destination}")
            display_destination_info(destination)
    
    st.markdown("---")
    
    if st.button("Generate Route Options", key="generate_routes"):
        if not start_location or not destination:
            st.error("Please enter both starting location and destination.")
        else:
            with st.spinner("Generating route options..."):
                with st_lottie_container:
                    display_lottie(lottie_travel)
                
                # Call function to generate route options
                route_options = generate_route_options(
                    start_location, 
                    destination, 
                    budget, 
                    duration
                )
                
                # Store in session state for later use
                st.session_state.route_options = route_options
                
            # Display options
            st.success("Route options generated!")
        
    # Display route options if available
    if 'route_options' in st.session_state:
        route_options = st.session_state.route_options
        
        for option in route_options["route_options"]:
            with st.expander(f"Option {option['option_id']}: {option['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Cost:** {option['estimated_cost']}")
                    st.markdown(f"**Time:** {option['travel_time']}")
                    
                    st.markdown("**Pros:**")
                    for pro in option['pros']:
                        st.markdown(f"- {pro}")
                        
                with col2:
                    st.markdown("**Cons:**")
                    for con in option['cons']:
                        st.markdown(f"- {con}")
                
                st.markdown("**Transportation Details:**")
                st.info(option['transportation_details'])
                
                # Button to select this route
                if st.button(f"Select Option {option['option_id']}", key=f"select_{option['option_id']}"):
                    st.session_state.selected_route = option
                    
                    # Extract journey legs from the route
                    # Parse transportation details to identify journey legs
                    journey_legs = []
                    
                    if places:
                        # Create journey legs based on places to visit
                        place_list = [p.strip() for p in places.split(",")]
                        all_locations = [start_location] + place_list + [destination]
                        
                        # Create legs between consecutive locations
                        for i in range(len(all_locations)-1):
                            # Determine transportation type based on distance or route option
                            # This is a simplified logic - in real app, would be more sophisticated
                            from_loc = all_locations[i]
                            to_loc = all_locations[i+1]
                            
                            # Assign different transport types based on leg index
                            transport_types = ["flight", "train", "bus", "cab"]
                            transport_type = transport_types[i % len(transport_types)]
                            
                            journey_legs.append({
                                "from": from_loc,
                                "to": to_loc,
                                "type": transport_type
                            })
                    else:
                        # If no places specified, create direct route
                        journey_legs.append({
                            "from": start_location,
                            "to": destination,
                            "type": "flight" if "flight" in option['name'].lower() else "train"
                        })
                        
                    st.session_state.journey_legs = journey_legs
                    st.session_state.route_places = [start_location] + ([p.strip() for p in places.split(",")] if places else []) + [destination]
                    st.success(f"Option {option['option_id']} selected! Go to the Detailed Itinerary tab.")

with tab2:
    st.header("Step 2: View Your Detailed Itinerary")
    
    # Check if a route has been selected
    if 'selected_route' not in st.session_state:
        st.info("Please select a route option in the Route Planning tab first.")
    else:
        # Extract destinations from journey legs for display
        if 'journey_legs' in st.session_state and 'route_places' in st.session_state:
            st.subheader("Places on Your Journey")
            places_list = st.session_state.route_places
            
            # Display all destinations info
            with st.spinner("Generating unique descriptions for each destination..."):
                display_multi_destination_info(places_list)
            # display_multi_destination_info(places_list)
            
            st.markdown("---")
        
        if st.button("Generate Detailed Itinerary", key="generate_itinerary"):
            with st.spinner("Creating your personalized itinerary..."):
                with st_lottie_container:
                    display_lottie(lottie_travel)
                
                # Call function to generate itinerary
                itinerary = generate_itinerary(
                    st.session_state.selected_route,
                    start_location,
                    destination,
                    budget,
                    duration,
                    places
                )
                
                # Store in session state
                st.session_state.itinerary = itinerary
            
            st.success("Itinerary created!")
        
        # Display itinerary if available
        if 'itinerary' in st.session_state:
            itinerary = st.session_state.itinerary
            
            # Overview
            st.subheader("Trip Overview")
            st.write(itinerary["overview"])
            
            # Daily Plan
            st.subheader("Daily Plan")
            for day in itinerary["daily_plan"]:
                with st.expander(f"Day {day['day']} - {day['date']} - {day['location']}"):
                    st.markdown(f"**Accommodation:** {day['accommodation']}")
                    st.markdown(f"**Transportation:** {day['transportation_for_day']}")
                    
                    st.markdown("**Activities:**")
                    for activity in day['activities']:
                        st.markdown(f"- {activity}")
                    
                    st.markdown("**Meals:**")
                    for meal in day['meals']:
                        st.markdown(f"- {meal}")
            
            # Budget
            st.subheader("Budget Breakdown")
            budget_data = [[k.capitalize(), v] for k, v in itinerary["budget_breakdown"].items()]
            st.table(budget_data)
            
            # Packing
            st.subheader("Packing Suggestions")
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            items_per_col = len(itinerary["packing_suggestions"]) // 3 + 1
            
            for i, item in enumerate(itinerary["packing_suggestions"]):
                col_idx = i // items_per_col
                cols[col_idx].markdown(f"- {item}")
            
            # Download as PDF option
            st.download_button(
                "Download Itinerary as PDF",
                "Itinerary PDF content would go here",  # In a real app, you'd generate a PDF
                file_name="travel_itinerary.pdf",
                mime="application/pdf",
            )
            
            # Confirmation to proceed to booking
            st.subheader("Ready to Book Your Trip?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes, Proceed to Booking", type="primary"):
                    st.session_state.proceed_to_booking = True
                    st.success("Great! Let's book your trip. Go to the Booking & Reservations tab.")
            
            with col2:
                if st.button("No, Modify Routes", key="modify_routes"):
                    st.session_state.proceed_to_booking = False
                    st.info("You can go back to the Route Planning tab to explore more options.")

with tab3:
    st.header("Step 3: Book Your Trip")
    
    # Transport type icons dictionary
    transport_icons = {
        "flight": "✈️",
        "train": "🚆",
        "bus": "🚌",
        "cab": "🚕"
    }
    
    # Payment method icons dictionary
    payment_icons = {
        "Credit Card": "💳",
        "Debit Card": "💳",
        "Net Banking": "🏦",
        "UPI": "📱",
        "Wallet": "👛"
    }
    
    # Check if user has itinerary and has confirmed to proceed
    if 'itinerary' not in st.session_state:
        st.info("Please generate an itinerary in the Detailed Itinerary tab first.")
    elif not st.session_state.get('proceed_to_booking', False):
        st.info("Please confirm you want to proceed with booking in the Detailed Itinerary tab.")
    else:
        # Process each leg of the journey
        if 'journey_legs' in st.session_state:
            journey_legs = st.session_state.journey_legs
            
            # Display a step-by-step booking process
            for i, leg in enumerate(journey_legs):
                leg_id = f"leg_{i+1}"
                leg_type_icon = transport_icons.get(leg['type'].lower(), "🚀")
                leg_name = f"{leg['from']} to {leg['to']} via {leg['type'].title()}"
                
                # Check if this leg is already booked
                is_booked = st.session_state.booking_steps_completed.get(leg_name, False)
                status = "✅" if is_booked else "⏳ Pending"
                
                with st.expander(f"{leg_type_icon} Step {i+1}: {leg_name} - {status}", expanded=not is_booked):
                    if is_booked:
                        st.success(f"✅ Your {leg['type']} from {leg['from']} to {leg['to']} has been booked successfully.")
                        continue
                        
                    st.markdown(f"**From:** {leg['from']}  |  **To:** {leg['to']}  |  **Transport Type:** {leg_type_icon} {leg['type'].title()}")
                    
                    # Generate and display options based on transportation type
                    if leg['type'].lower() == 'flight':
                        # Add a flight icon/image
                        st.image("https://cdn-icons-png.flaticon.com/512/3125/3125713.png", width=60)
                        options = generate_flight_options(leg['from'], leg['to'], "tomorrow")
                        df = pd.DataFrame(options)
                        st.dataframe(
                            df[['airline', 'flight_number', 'departure', 'arrival', 'duration', 'price']], 
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Allow selection
                        selected_option = st.radio(
                            "Select a flight:",
                            [f"{opt['airline']} {opt['flight_number']} - ₹{opt['price']} - {opt['departure']} to {opt['arrival']}" for opt in options],
                            key=f"flight_select_{i}"
                        )
                        
                        # Get selected option details
                        selected_idx = [f"{opt['airline']} {opt['flight_number']} - ₹{opt['price']} - {opt['departure']} to {opt['arrival']}" for opt in options].index(selected_option)
                        price = options[selected_idx]['price']
                        
                    elif leg['type'].lower() == 'train':
                        # Add a train icon/image
                        st.image("https://cdn-icons-png.flaticon.com/512/3078/3078984.png", width=60)
                        options = generate_train_options(leg['from'], leg['to'], "tomorrow")
                        df = pd.DataFrame(options)
                        st.dataframe(
                            df[['train_name', 'train_number', 'departure', 'arrival', 'duration', 'price']], 
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Allow selection
                        selected_option = st.radio(
                            "Select a train:",
                            [f"{opt['train_name']} {opt['train_number']} - ₹{opt['price']} - {opt['departure']} to {opt['arrival']}" for opt in options],
                            key=f"train_select_{i}"
                        )
                        
                        # Get selected option details
                        selected_idx = [f"{opt['train_name']} {opt['train_number']} - ₹{opt['price']} - {opt['departure']} to {opt['arrival']}" for opt in options].index(selected_option)
                        price = options[selected_idx]['price']
                        
                    elif leg['type'].lower() == 'bus':
                        # Add a bus icon/image
                        st.image("https://cdn-icons-png.flaticon.com/512/9400/9400355.png", width=60)
                        options = generate_bus_options(leg['from'], leg['to'], "tomorrow")
                        df = pd.DataFrame(options)
                        st.dataframe(
                            df[['operator', 'bus_type', 'departure', 'arrival', 'duration', 'price']], 
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Allow selection
                        selected_option = st.radio(
                            "Select a bus:",
                            [f"{opt['operator']} {opt['bus_type']} - ₹{opt['price']} - {opt['departure']} to {opt['arrival']}" for opt in options],
                            key=f"bus_select_{i}"
                        )
                        
                        # Get selected option details
                        selected_idx = [f"{opt['operator']} {opt['bus_type']} - ₹{opt['price']} - {opt['departure']} to {opt['arrival']}" for opt in options].index(selected_option)
                        price = options[selected_idx]['price']
                        
                    elif leg['type'].lower() == 'cab':
                        # Add a cab icon/image
                        st.image("https://cdn-icons-png.flaticon.com/512/744/744465.png", width=60)
                        options = generate_cab_options(leg['from'], leg['to'])
                        df = pd.DataFrame(options)
                        st.dataframe(
                            df[['cab_type', 'price', 'operators']], 
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Allow selection
                        selected_option = st.radio(
                            "Select a cab type:",
                            [f"{opt['cab_type']} - ₹{opt['price']}" for opt in options],
                            key=f"cab_select_{i}"
                        )
                        
                        # Get selected option details
                        selected_idx = [f"{opt['cab_type']} - ₹{opt['price']}" for opt in options].index(selected_option)
                        price = options[selected_idx]['price']
                    
                    # Payment section with enhanced visuals
                    st.markdown("---")
                    st.subheader(f"💰 Payment for {leg['from']} to {leg['to']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        payment_method = display_payment_methods(key_suffix=f"leg_{i}")
                        # Display the payment icon
                        if payment_method in payment_icons:
                            st.markdown(f"**Selected Payment Method:** {payment_icons[payment_method]} {payment_method}")
                    
                    with col2:
                        st.info(f"**Amount:** ₹{price}")
                        
                        # Updated payment security image
                        st.image("https://cdn-icons-png.flaticon.com/512/2146/2146588.png", width=100, caption="Secure Payment")
                    
                    # Process payment
                    if st.button(f"💳 Complete Booking for {leg['from']} to {leg['to']}", key=f"pay_leg_{i}", type="primary"):
                        with st.spinner("Processing your booking..."):
                            time.sleep(2)  # Simulate processing time
                            st.session_state.booking_steps_completed[leg_name] = True
                            
                            if "payments" not in st.session_state:
                                st.session_state.payments = []
                                
                            # Add payment details to session state
                            st.session_state.payments.append({
                                "transaction_id": f"TX-{random.randint(1000000, 9999999)}",
                                "amount": price,
                                "booking_type": f"{leg['type']} from {leg['from']} to {leg['to']}",
                                "payment_method": payment_method,
                                "status": "Successful",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                        st.success(f"✅ Booking confirmed! Your {leg['type']} from {leg['from']} to {leg['to']} has been booked.")
            
            # Hotel booking section with enhanced visuals
            st.markdown("---")
            st.subheader("🏨 Hotel Bookings")

            # Hotel images dictionary for specific locations
            hotel_images = {
                "Bhubaneswar": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/0c/13/34/03/mayfair-convention.jpg?w=700&h=-1&s=1",
                "Delhi": "https://cdn.britannica.com/96/115096-050-5AFDAF5D/Bellagio-Hotel-Casino-Las-Vegas.jpg",
                "Agra": "https://imgcy.trivago.com/c_fill,d_dummy.jpeg,e_sharpen:60,f_auto,h_627,q_auto,w_1200/itemimages/66/54/66546_v5.jpeg",
                "PrayagRaj": "https://imgcy.trivago.com/c_fill,d_dummy.jpeg,e_sharpen:60,f_auto,h_627,q_auto,w_1200/itemimages/31/75/3175346_v4.jpeg",
                "Patna": "https://imgcy.trivago.com/c_fill,d_dummy.jpeg,e_sharpen:60,f_auto,h_627,q_auto,w_1200/itemimages/55/23/5523770.jpeg",
                "New York": "https://www.usatoday.com/gcdn/-mm-/05b227ad5b8ad4e9dcb53af4f31d7fbdb7fa901b/c=0-64-2119-1259/local/-/media/USATODAY/USATODAY/2014/08/13/1407953244000-177513283.jpg",
                "San Francisco": "https://cf.bstatic.com/xdata/images/hotel/max1024x768/164322215.jpg?k=fc3862975a4b55f867574c16f63e42bcb3dc5bd895895615fb0239fc42030b70&o=&hp=1",
            }

            # Extract accommodation locations from itinerary
            accommodation_locations = []
            for day in st.session_state.itinerary["daily_plan"]:
                if day['location'] and day['location'] not in accommodation_locations:
                    accommodation_locations.append(day['location'])

            # Display accommodations to book
            if accommodation_locations:
                for location_idx, location in enumerate(accommodation_locations):
                    acc_name = f"Hotel in {location}"
                    is_booked = st.session_state.booking_steps_completed.get(acc_name, False)
                    status = "✅" if is_booked else "⏳ Pending"
                    
                    with st.expander(f"🏨 Hotel Booking in {location} - {status}", expanded=not is_booked):
                        if is_booked:
                            st.success(f"✅ Your accommodation in {location} has been booked successfully.")
                            continue
                            
                        # Generate hotel options for the location
                        hotel_options = generate_hotel_options(location, "tomorrow", "day after")
                        
                        # Display hotel options with better images
                        for j, hotel in enumerate(hotel_options):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                # Use location-specific images if available, otherwise use a more specific hotel query
                                if location in hotel_images:
                                    image_url = hotel_images[location]
                                else:
                                    # More specific query for better hotel images
                                    image_url = f"https://source.unsplash.com/featured/300x200/?luxury,hotel,{location}"
                                st.image(image_url, caption=hotel['name'])
                            
                            with col2:
                                st.markdown(f"**{hotel['name']}**")
                                st.markdown(f"⭐ {'⭐' * (hotel['stars'] - 1)} ({hotel['rating']})")
                                st.markdown(f"**Price:** 💰 ₹{hotel['price_per_night']} per night")
                                st.markdown(f"**Amenities:** 🛌 {', '.join(hotel['amenities'])}")
                                
                                if st.button(f"🏨 Select {hotel['name']}", key=f"select_hotel_{location_idx}_{j}"):
                                    st.session_state.selected_hotel = hotel
                                    st.session_state.selected_hotel_location = location
                        
                        # Booking form if hotel selected
                        if "selected_hotel" in st.session_state and st.session_state.selected_hotel_location == location:
                            hotel = st.session_state.selected_hotel
                            st.markdown("---")
                            st.markdown(f"### 🏨 Booking: {hotel['name']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📅 Check-in Date:**")
                                checkin = st.date_input(f"Check-in", value=None, key=f"checkin_{location_idx}", label_visibility="collapsed")
                                st.markdown("**🛏️ Room Type:**")
                                room_type = st.selectbox("Room Type", ["Standard", "Deluxe", "Suite"], key=f"room_{location_idx}", label_visibility="collapsed")
                            with col2:
                                st.markdown("**📅 Check-out Date:**")
                                checkout = st.date_input(f"Check-out", value=None, key=f"checkout_{location_idx}", label_visibility="collapsed")
                                st.markdown("**👥 Number of Guests:**")
                                guests = st.selectbox("Guests", list(range(1, 5)), key=f"guests_{location_idx}", label_visibility="collapsed")
                                
                            # Calculate total price
                            nights = (checkout - checkin).days if checkin and checkout else 1
                            total_price = hotel['price_per_night'] * nights
                            
                            st.info(f"**Total for {nights} night(s):** 💰 ₹{total_price}")
                            
                            # Payment section
                            st.subheader(f"💳 Payment for {hotel['name']}")
                            
                            # Use a unique key for each hotel payment
                            payment_method = display_payment_methods(key_suffix=f"hotel_{location_idx}_{hotel['name'].replace(' ', '_')}")
                            
                            # Display payment icon
                            if payment_method in payment_icons:
                                st.markdown(f"**Selected Payment Method:** {payment_icons[payment_method]} {payment_method}")
                            
                            # Process payment
                            if st.button(f"💳 Complete Hotel Booking", key=f"pay_hotel_{location_idx}_{hotel['name'].replace(' ', '_')}", type="primary"):
                                with st.spinner("Processing your booking..."):
                                    time.sleep(2)  # Simulate processing time
                                    handle_hotel_booking(location_idx, hotel, payment_method, total_price)
                                    
                                st.success(f"✅ Hotel booking confirmed! Your stay at {hotel['name']} in {location} has been reserved.")
            
            # Payment summary section with enhanced visuals
            st.markdown("---")
            st.subheader("💰 Payment Summary")
            display_payment_summary()
            
            # Complete booking status
            total_steps = len(journey_legs) + len(accommodation_locations)
            completed_steps = sum(1 for step, status in st.session_state.booking_steps_completed.items() if status)
            
            progress_percentage = int((completed_steps / total_steps) * 100) if total_steps > 0 else 0
            st.progress(progress_percentage / 100)
            st.markdown(f"**Booking Progress:** {progress_percentage}% complete ({completed_steps}/{total_steps} steps)")
            
            if progress_percentage == 100:
                st.success("🎉 Congratulations! All bookings are complete. Your trip is fully booked and ready!")
                
                # Generate final itinerary button
                if st.button("📄 Generate Final Trip Itinerary"):
                    st.markdown("### 🧳 Your Complete Trip Itinerary")
                    st.markdown("Below is your finalized trip itinerary with all bookings confirmed:")
                    
                    # Display journey legs with booking details
                    for i, leg in enumerate(journey_legs):
                        leg_type_icon = transport_icons.get(leg['type'].lower(), "🚀")
                        st.markdown(f"**Leg {i+1}:** {leg_type_icon} {leg['from']} to {leg['to']} via {leg['type'].title()}")
                    
                    # Display accommodation bookings
                    for location in accommodation_locations:
                        st.markdown(f"**Stay:** 🏨 Hotel in {location}")
                    
                    # Display payment summary
                    if "payments" in st.session_state and st.session_state.payments:
                        total_amount = sum(payment["amount"] for payment in st.session_state.payments)
                        st.markdown(f"**Total Amount Paid:** 💰 ₹{total_amount:,.2f}")
                    
                    # Option to download
                    st.download_button(
                        "📥 Download Complete Itinerary",
                        "Complete itinerary content would go here",
                        file_name="complete_travel_itinerary.pdf",
                        mime="application/pdf",
                    )
            
            # Reset option
            st.markdown("---")
            if st.button("🔄 Start Over With New Route Planning"):
                for key in ['selected_route', 'itinerary', 'proceed_to_booking', 'journey_legs', 
                           'booking_steps_completed', 'payments', 'selected_hotel', 'selected_hotel_location']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Planning reset! You can now go back to the Route Planning tab.")