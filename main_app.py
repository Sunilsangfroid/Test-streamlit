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

# Add this function at the top of your file with other helper functions
def validate_payment_fields(payment_method, form_data):
    """
    Validate payment form fields based on payment method
    Returns (is_valid, error_message)
    """
    if payment_method in ["Credit Card", "Debit Card"]:
        card_num = form_data.get("card_number", "")
        card_num = card_num.replace(" ", "")  # Remove spaces
        
        # Validate card number
        if not card_num.isdigit() or len(card_num) != 16:
            return False, "Card number must be 16 digits"
            
        # Validate CVV
        cvv = form_data.get("cvv", "")
        if not cvv.isdigit() or len(cvv) != 3:
            return False, "CVV must be 3 digits"
            
        # Validate expiry date format
        exp_date = form_data.get("exp_date", "")
        if not "/" in exp_date:
            return False, "Expiry date must be in MM/YY format"
            
    elif payment_method == "UPI":
        upi_id = form_data.get("upi_id", "")
        if not "@" in upi_id or len(upi_id.split("@")) != 2:
            return False, "Please enter a valid UPI ID (username@provider)"
            
    elif payment_method == "Wallet":
        mobile = form_data.get("mobile", "")
        mobile = mobile.replace(" ", "")
        if not mobile.isdigit() or len(mobile) != 10:
            return False, "Mobile number must be 10 digits"
    
    return True, ""
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

# Add this right after creating the tabs but before the tab content
# Tab navigation based on session state
if st.session_state.get('nav_to_tab1', False):
    # Clear the navigation flag
    st.session_state['nav_to_tab1'] = False
    # This doesn't directly switch tabs but indicates to the user to go to tab1
    st.info("Your new trip planning is ready! Please click on the 'Route Planning' tab to begin.")

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
                        payment_method = st.selectbox(
                            "Select Payment Method",
                            ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"],
                            key=f"payment_method_{i}"
                        )
                        
                        # Form data to validate
                        form_data = {}
                        
                        # Display payment form based on selection with validation
                        if payment_method == "Credit Card" or payment_method == "Debit Card":
                            card_type = "Credit" if payment_method == "Credit Card" else "Debit"
                            card_number = st.text_input(
                                f"{card_type} Card Number", 
                                placeholder="XXXX XXXX XXXX XXXX", 
                                key=f"card_{i}",
                                help="Must be 16 digits"
                            )
                            form_data["card_number"] = card_number
                            
                            c1, c2 = st.columns(2)
                            with c1:
                                exp_date = st.text_input(
                                    "Expiry Date", 
                                    placeholder="MM/YY", 
                                    key=f"exp_{i}"
                                )
                                form_data["exp_date"] = exp_date
                            with c2:
                                cvv = st.text_input(
                                    "CVV", 
                                    type="password", 
                                    key=f"cvv_{i}",
                                    placeholder="XXX",
                                    help="Must be 3 digits"
                                )
                                form_data["cvv"] = cvv
                                
                            st.text_input("Name on Card", key=f"name_{i}")
                                
                        elif payment_method == "UPI":
                            upi_id = st.text_input(
                                "UPI ID", 
                                placeholder="username@upi", 
                                key=f"upi_{i}",
                                help="Enter in format: username@upi"
                            )
                            form_data["upi_id"] = upi_id
                                
                        elif payment_method == "Net Banking":
                            st.selectbox(
                                "Select Bank", 
                                ["SBI", "HDFC", "ICICI", "Axis", "PNB"], 
                                key=f"bank_{i}"
                            )
                            st.text_input("User ID", key=f"userid_{i}")
                            st.text_input("Password", type="password", key=f"pwd_{i}")
                                
                        elif payment_method == "Wallet":
                            col_code, col_num = st.columns([0.3, 0.7])
                            with col_code:
                                country_code = st.selectbox(
                                    "Code",
                                    ["+91", "+1", "+44", "+61", "+81"],
                                    index=0,
                                    key=f"country_code_{i}",
                                    label_visibility="collapsed"
                                )
                            with col_num:
                                mobile = st.text_input(
                                    "Mobile Number", 
                                    key=f"mobile_{i}",
                                    placeholder="10-digit number",
                                    help="Must be 10 digits"
                                )
                            form_data["mobile"] = mobile
                            
                            st.selectbox(
                                "Select Wallet", 
                                ["Paytm", "PhonePe", "Google Pay", "Amazon Pay"], 
                                key=f"wallet_type_{i}"
                            )
                        
                        # Display the payment icon
                        if payment_method in payment_icons:
                            st.markdown(f"**Selected Payment Method:** {payment_icons[payment_method]} {payment_method}")

                    with col2:
                        st.info(f"**Amount:** ₹{price}")
                        
                        # Updated payment security image
                        st.image("https://cdn-icons-png.flaticon.com/512/2146/2146588.png", width=100, caption="Secure Payment")

                    # Process payment with validation
                    if st.button(f"💳 Complete Booking for {leg['from']} to {leg['to']}", key=f"pay_leg_{i}", type="primary"):
                        # Validate form data
                        is_valid, error_message = validate_payment_fields(payment_method, form_data)
                        
                        if not is_valid:
                            st.error(f"🚫 {error_message}")
                        else:
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
            
            # Hotel booking section with fixed execution
            st.markdown("---")
            st.subheader("🏨 Hotel Bookings")

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
                    
                    # Hotel booking expander
                    with st.expander(f"🏨 Hotel Booking in {location} - {status}", expanded=not is_booked):
                        if is_booked:
                            st.success(f"✅ Your accommodation in {location} has been booked successfully.")
                            
                            # If confirmation details are available, show them
                            if f"hotel_{location_idx}_confirmation" in st.session_state:
                                conf = st.session_state[f"hotel_{location_idx}_confirmation"]
                                st.info(f"""
                                **Booking Details:**
                                - Hotel: {conf['hotel_name']}
                                - Location: {conf['location']}
                                - Amount Paid: ₹{conf['price']}
                                - Transaction ID: {conf['transaction_id']}
                                """)
                            continue
                        
                        # Generate hotel options for the location
                        hotel_options = generate_hotel_options(location, "tomorrow", "day after")
                        
                        # Display hotel options in a grid
                        st.write(f"**Select a hotel in {location}:**")
                        
                        # Create columns for hotel display
                        cols = st.columns(2)
                        
                        for j, hotel in enumerate(hotel_options):
                            col = cols[j % 2]
                            with col:
                                # Hotel card with border
                                with st.container():
                                    st.markdown("""
                                    <style>
                                    .hotel-card {
                                        border: 1px solid #ddd;
                                        border-radius: 10px;
                                        padding: 10px;
                                        margin-bottom: 15px;
                                    }
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown(f"<div class='hotel-card'>", unsafe_allow_html=True)
                                    st.markdown(f"### {hotel['name']}")
                                    st.markdown(f"⭐ {'⭐' * (hotel['stars'] - 1)} ({hotel['rating']})")
                                    st.markdown(f"**Price:** ₹{hotel['price_per_night']} per night")
                                    st.markdown(f"**Amenities:** {', '.join(hotel['amenities'][:3])}")
                                    
                                    # Simple one-click booking button
                                    if st.button(f"Book this hotel", key=f"quick_book_{location_idx}_{j}"):
                                        with st.spinner("Processing your booking..."):
                                            # Default to a 2-night stay
                                            total_price = hotel['price_per_night'] * 2
                                            
                                            # Process booking directly with minimal input
                                            transaction_id = handle_hotel_booking(
                                                location_idx,
                                                hotel,
                                                "Credit Card",  # Default payment method
                                                total_price
                                            )
                                            
                                            # Mark as booked in session state
                                            st.session_state.booking_steps_completed[acc_name] = True
                                            
                                            # Success notification without balloons
                                            st.success(f"✅ Hotel booking confirmed! Your stay at {hotel['name']} in {location} has been reserved.")
                                            
                                            # Use the modern rerun method
                                            st.rerun()
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)

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
                # Improved success message with green tick mark
                st.markdown("""
                <div style="background-color:#d4edda; padding:20px; border-radius:10px; border:1px solid #c3e6cb; margin:20px 0;">
                    <h3 style="color:#155724;"><span style="font-size:30px;">✅</span> All Bookings Complete!</h3>
                    <p>Congratulations! Your entire trip is now booked and ready. All transportation and accommodations have been confirmed.</p>
                </div>
                """, unsafe_allow_html=True)
                
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
                
                # Better navigation for new journey
                st.markdown("---")
                st.markdown("### Start a New Journey")
                if st.button("🔄 Plan Another Trip", type="primary"):
                    # Clear session state for all relevant keys
                    for key in ['selected_route', 'itinerary', 'proceed_to_booking', 'journey_legs', 
                               'booking_steps_completed', 'payments']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Clear any hotel specific keys
                    for key in list(st.session_state.keys()):
                        if key.startswith('hotel_'):
                            del st.session_state[key]
                    
                    # Add a flag to navigate back to tab1
                    st.session_state['nav_to_tab1'] = True
                    st.rerun()
            else:
                # Reset option for incomplete bookings
                st.markdown("---")
                if st.button("🔄 Start Over With New Route Planning"):
                    for key in ['selected_route', 'itinerary', 'proceed_to_booking', 'journey_legs', 
                               'booking_steps_completed', 'payments']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Clear hotel related keys
                    for key in list(st.session_state.keys()):
                        if key.startswith('hotel_'):
                            del st.session_state[key]
                        
                    st.success("Planning reset! You can now go back to the Route Planning tab.")
