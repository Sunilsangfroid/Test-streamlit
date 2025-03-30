import streamlit as st
import random
import pandas as pd
from datetime import datetime, timedelta
from payment_processor import display_payment_methods, process_payment

def generate_flight_options(from_loc, to_loc, date):
    """
    Generate flight options between two locations
    
    Args:
        from_loc: Departure location
        to_loc: Arrival location
        date: Departure date (string)
        
    Returns:
        List of flight options
    """
    # Mock flight data
    airlines = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoAir"]
    
    # Generate realistic flight times based on locations
    flight_times = {
        "short": {"min": 45, "max": 120},
        "medium": {"min": 120, "max": 240},
        "long": {"min": 240, "max": 480}
    }
    
    # Determine distance category (simplified)
    major_cities = {"Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"}
    if from_loc in major_cities and to_loc in major_cities:
        distance = "medium"
    elif from_loc in major_cities or to_loc in major_cities:
        distance = "medium"
    else:
        distance = "short"
        
    # For international destinations
    international = ["New York", "London", "Dubai", "Singapore", "Tokyo"]
    if from_loc in international or to_loc in international:
        distance = "long"
    
    # Generate options
    options = []
    num_options = random.randint(3, 6)
    
    # Base price factors
    price_factors = {
        "short": (2000, 5000),
        "medium": (5000, 10000),
        "long": (20000, 50000)
    }
    
    # Generate departure times spread throughout the day
    departure_hours = random.sample(range(6, 21), num_options)
    departure_hours.sort()
    
    for i in range(num_options):
        airline = random.choice(airlines)
        flight_number = f"{airline[:2]}{random.randint(100, 999)}"
        
        # Generate departure time
        hour = departure_hours[i]
        minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        departure = f"{hour:02d}:{minute:02d}"
        
        # Generate duration based on distance
        duration_mins = random.randint(
            flight_times[distance]["min"], 
            flight_times[distance]["max"]
        )
        
        # Calculate arrival time
        dep_hour, dep_min = map(int, departure.split(':'))
        total_mins = dep_hour * 60 + dep_min + duration_mins
        arr_hour = (total_mins // 60) % 24
        arr_min = total_mins % 60
        arrival = f"{arr_hour:02d}:{arr_min:02d}"
        
        # Format duration for display
        duration_str = f"{duration_mins // 60}h {duration_mins % 60}m"
        
        # Generate price
        base_min, base_max = price_factors[distance]
        price = random.randint(base_min, base_max)
        # Add some variation based on airline and time
        if airline in ["Vistara", "Air India"]:
            price += random.randint(1000, 3000)  # Premium airlines
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            price += random.randint(500, 1500)  # Peak hours
            
        options.append({
            "airline": airline,
            "flight_number": flight_number,
            "departure": departure,
            "arrival": arrival,
            "duration": duration_str,
            "price": price,
            "from": from_loc,
            "to": to_loc
        })
    
    return options

def generate_train_options(from_loc, to_loc, date):
    """
    Generate train options between two locations
    
    Args:
        from_loc: Departure location
        to_loc: Arrival location
        date: Departure date (string)
        
    Returns:
        List of train options
    """
    # Mock train data
    train_types = ["Rajdhani", "Shatabdi", "Duronto", "Superfast", "Express", "Passenger"]
    
    # Generate realistic train times based on locations
    train_times = {
        "short": {"min": 120, "max": 300},
        "medium": {"min": 300, "max": 720},
        "long": {"min": 720, "max": 1800}
    }
    
    # Determine distance category
    if from_loc == to_loc:
        return []  # No trains for same location
    
    major_cities = {"Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"}
    if from_loc in major_cities and to_loc in major_cities:
        distance = "medium"
    elif from_loc in major_cities or to_loc in major_cities:
        distance = "medium"
    else:
        distance = "short"
    
    # Generate options
    options = []
    num_options = random.randint(2, 5)
    
    # Base price factors
    price_factors = {
        "short": (300, 1000),
        "medium": (800, 2500),
        "long": (2000, 5000)
    }
    
    # Generate departure times spread throughout the day
    departure_hours = random.sample(range(0, 23), num_options)
    departure_hours.sort()
    
    for i in range(num_options):
        train_type_idx = random.randint(0, min(len(train_types)-1, 5))
        train_type = train_types[train_type_idx]
        train_name = f"{from_loc[:3]}-{to_loc[:3]} {train_type}"
        train_number = str(random.randint(10000, 99999))
        
        # Generate departure time
        hour = departure_hours[i]
        minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        departure = f"{hour:02d}:{minute:02d}"
        
        # Generate duration based on distance
        duration_mins = random.randint(
            train_times[distance]["min"], 
            train_times[distance]["max"]
        )
        
        # Calculate arrival time
        dep_hour, dep_min = map(int, departure.split(':'))
        total_mins = dep_hour * 60 + dep_min + duration_mins
        arr_hour = (total_mins // 60) % 24
        arr_min = total_mins % 60
        arrival = f"{arr_hour:02d}:{arr_min:02d}"
        
        # Format duration for display
        duration_str = f"{duration_mins // 60}h {duration_mins % 60}m"
        
        # Generate price
        base_min, base_max = price_factors[distance]
        price = random.randint(base_min, base_max)
        # Add some variation based on train type
        price_modifier = 5 - train_type_idx  # Higher price for premium trains
        price += price_modifier * 100
            
        options.append({
            "train_name": train_name,
            "train_number": train_number,
            "departure": departure,
            "arrival": arrival,
            "duration": duration_str,
            "price": price,
            "from": from_loc,
            "to": to_loc
        })
    
    return options

def generate_bus_options(from_loc, to_loc, date):
    """
    Generate bus options between two locations
    
    Args:
        from_loc: Departure location
        to_loc: Arrival location
        date: Departure date (string)
        
    Returns:
        List of bus options
    """
    # Mock bus data
    operators = ["RedBus", "Zing Bus", "Intercity", "Express Ways", "Travels"]
    bus_types = ["AC Sleeper", "AC Seater", "Non-AC Sleeper", "Non-AC Seater", "Deluxe", "Super Deluxe"]
    
    # Generate realistic bus times based on locations
    bus_times = {
        "short": {"min": 120, "max": 360},
        "medium": {"min": 360, "max": 720},
        "long": {"min": 720, "max": 1200}
    }
    
    # Determine distance category
    if from_loc == to_loc:
        return []  # No buses for same location
    
    major_cities = {"Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"}
    if from_loc in major_cities and to_loc in major_cities:
        distance = "medium"
    elif from_loc in major_cities or to_loc in major_cities:
        distance = "medium"
    else:
        distance = "short"
    
    # Generate options
    options = []
    num_options = random.randint(3, 8)
    
    # Base price factors
    price_factors = {
        "short": (250, 800),
        "medium": (700, 1800),
        "long": (1500, 3500)
    }
    
    # Generate departure times spread throughout the day and night
    departure_hours = random.sample(list(range(6, 24)) + list(range(0, 6)), min(num_options, 24))
    departure_hours.sort()
    
    for i in range(num_options):
        operator = random.choice(operators)
        bus_type = random.choice(bus_types)
        
        # Generate departure time
        hour = departure_hours[i % len(departure_hours)]
        minute = random.choice([0, 15, 30, 45])
        departure = f"{hour:02d}:{minute:02d}"
        
        # Generate duration based on distance
        duration_mins = random.randint(
            bus_times[distance]["min"], 
            bus_times[distance]["max"]
        )
        
        # Calculate arrival time
        dep_hour, dep_min = map(int, departure.split(':'))
        total_mins = dep_hour * 60 + dep_min + duration_mins
        arr_hour = (total_mins // 60) % 24
        arr_min = total_mins % 60
        arrival = f"{arr_hour:02d}:{arr_min:02d}"
        
        # Format duration for display
        duration_str = f"{duration_mins // 60}h {duration_mins % 60}m"
        
        # Generate price
        base_min, base_max = price_factors[distance]
        price = random.randint(base_min, base_max)
        
        # Adjust price based on bus type
        if "AC" in bus_type:
            price += random.randint(200, 500)
        if "Sleeper" in bus_type:
            price += random.randint(100, 300)
        if "Deluxe" in bus_type:
            price += random.randint(300, 700)
            
        options.append({
            "operator": operator,
            "bus_type": bus_type,
            "departure": departure,
            "arrival": arrival,
            "duration": duration_str,
            "price": price,
            "from": from_loc,
            "to": to_loc
        })
    
    return options

def generate_cab_options(from_loc, to_loc):
    """
    Generate cab options between two locations
    
    Args:
        from_loc: Departure location
        to_loc: Arrival location
        
    Returns:
        List of cab options
    """
    # Mock cab data
    cab_types = ["Economy", "Standard", "Prime", "SUV", "Luxury"]
    operators = ["Ola", "Uber", "Meru", "BlueSmart", "Local Taxi"]
    
    # Determine distance category
    if from_loc == to_loc:
        distance_km = random.randint(5, 30)  # City travel
    else:
        # Simplified distance estimation
        major_cities = {"Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad", 
                       "Patna", "Mathura", "Agra", "PrayagRaj", "Bhubaneswar"}
        
        if from_loc in major_cities and to_loc in major_cities:
            # Rough distance between major cities (simplified)
            city_distances = {
                ("Delhi", "Agra"): 230,
                ("Delhi", "Mathura"): 160,
                ("Mathura", "Agra"): 60,
                ("Agra", "PrayagRaj"): 420,
                ("PrayagRaj", "Patna"): 320,
                ("Patna", "Bhubaneswar"): 850,
                # Add more as needed
            }
            
            # Check if we have the distance for this city pair
            pair = (from_loc, to_loc)
            reverse_pair = (to_loc, from_loc)
            
            if pair in city_distances:
                distance_km = city_distances[pair]
            elif reverse_pair in city_distances:
                distance_km = city_distances[reverse_pair]
            else:
                # Default distance for cities without specific mapping
                distance_km = random.randint(200, 800)
        else:
            distance_km = random.randint(50, 200)
    
    # Generate options
    options = []
    
    for cab_type in cab_types:
        # Base rate per km based on cab type
        base_rates = {
            "Economy": 10,
            "Standard": 14,
            "Prime": 18,
            "SUV": 22,
            "Luxury": 35
        }
        
        base_rate = base_rates[cab_type]
        
        # Calculate price
        price = distance_km * base_rate
        
        # Add booking fee and taxes
        booking_fee = 50 if cab_type in ["Economy", "Standard"] else 100
        price += booking_fee
        
        # Add surge factor (randomly)
        if random.random() < 0.3:  # 30% chance of surge
            surge = random.uniform(1.1, 1.5)
            price *= surge
        
        # Round the price
        price = int(price)
        
        # Determine which operators offer this cab type
        available_operators = random.sample(operators, random.randint(1, len(operators)))
        
        options.append({
            "cab_type": cab_type,
            "price": price,
            "distance": f"{distance_km} km",
            "operators": ", ".join(available_operators),
            "from": from_loc,
            "to": to_loc
        })
    
    return options

def generate_hotel_options(location, checkin, checkout):
    """
    Generate hotel options for a location
    
    Args:
        location: Hotel location
        checkin: Check-in date
        checkout: Check-out date
        
    Returns:
        List of hotel options
    """
    # Mock hotel data
    hotel_types = {
        "budget": {
            "name_prefixes": ["Hotel", "Stay", "Rooms", "Lodge"],
            "name_suffixes": ["Inn", "Residency", "Stay", "Comforts"],
            "price_range": (1200, 3000),
            "stars": 2,
            "amenities": ["Wi-Fi", "AC", "TV", "Restaurant"]
        },
        "mid_range": {
            "name_prefixes": ["Hotel", "Taj", "The", "Royal"],
            "name_suffixes": ["Suites", "Paradise", "Comfort", "Grand"],
            "price_range": (3000, 7000),
            "stars": 3,
            "amenities": ["Wi-Fi", "AC", "TV", "Restaurant", "Room Service", "Parking", "Gym"]
        },
        "luxury": {
            "name_prefixes": ["The", "Grand", "Royal", "Luxury"],
            "name_suffixes": ["Resort", "Palace", "Majestic", "Monarch"],
            "price_range": (7000, 20000),
            "stars": 5,
            "amenities": ["Wi-Fi", "AC", "TV", "Multiple Restaurants", "24/7 Room Service", 
                         "Swimming Pool", "Spa", "Gym", "Concierge", "Airport Transfer"]
        }
    }
    
    # Adjust price ranges for metro cities
    metro_cities = {"Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"}
    
    # Generate options
    options = []
    num_options = random.randint(3, 7)
    
    # City-specific names to add local flavor
    city_specific = {
        "Delhi": ["Delhi", "Connaught", "Karol Bagh", "Paharganj"],
        "Mumbai": ["Mumbai", "Marine Drive", "Colaba", "Juhu"],
        "Bangalore": ["Bangalore", "MG Road", "Indiranagar", "Koramangala"],
        "Kolkata": ["Bengal", "Kolkata", "Park Street"],
        "Chennai": ["Chennai", "Mount Road", "T Nagar"],
        "Hyderabad": ["Hyderabad", "Cyberabad", "Banjara Hills"],
        "Agra": ["Taj", "Agra", "Mughal"],
        "Mathura": ["Krishna", "Mathura", "Vrindavan"],
        "PrayagRaj": ["Sangam", "Allahabad", "PrayagRaj"],
        "Patna": ["Patna", "Bihar", "Ganga"],
        "Bhubaneswar": ["Kalinga", "Odisha", "Temple City"]
    }
    
    local_names = city_specific.get(location, [location])
    
    # Select hotel types based on city (more luxury in metros)
    if location in metro_cities:
        type_distribution = ["budget"] * 2 + ["mid_range"] * 3 + ["luxury"] * 2
    else:
        type_distribution = ["budget"] * 3 + ["mid_range"] * 3 + ["luxury"] * 1
    
    for i in range(num_options):
        hotel_type = random.choice(type_distribution)
        type_data = hotel_types[hotel_type]
        
        # Generate hotel name
        prefix = random.choice(type_data["name_prefixes"])
        suffix = random.choice(type_data["name_suffixes"])
        local_term = random.choice(local_names) if random.random() > 0.5 else ""
        
        if local_term:
            hotel_name = f"{prefix} {local_term} {suffix}"
        else:
            hotel_name = f"{prefix} {suffix}"
        
        # Clean up the name
        hotel_name = hotel_name.strip()
        
        # Adjust price based on city
        min_price, max_price = type_data["price_range"]
        if location in metro_cities:
            min_price = int(min_price * 1.3)
            max_price = int(max_price * 1.3)
        
        price = random.randint(min_price, max_price)
        
        # Select amenities
        amenities = random.sample(type_data["amenities"], min(5, len(type_data["amenities"])))
        
        # Generate rating
        if type_data["stars"] == 2:
            rating = round(random.uniform(3.0, 4.2), 1)
        elif type_data["stars"] == 3:
            rating = round(random.uniform(3.5, 4.5), 1)
        else:
            rating = round(random.uniform(4.0, 4.9), 1)
        
        options.append({
            "name": hotel_name,
            "price_per_night": price,
            "stars": type_data["stars"],
            "rating": rating,
            "amenities": amenities,
            "location": location
        })
    
    return options

def display_booking_progress(itinerary_legs):
    """Display booking progress for multi-leg journey"""
    st.sidebar.subheader("Booking Progress")
    
    for i, leg in enumerate(itinerary_legs):
        is_complete = st.session_state.get(f"booking_complete_{leg['type']}_{i}", False)
        icon = "✅" if is_complete else "⬜"
        st.sidebar.markdown(f"{icon} **Leg {i+1}:** {leg['from']} to {leg['to']} via {leg['type']}")
    
    # Hotel booking status
    hotel_booked = st.session_state.get("booking_complete_hotel", False)
    icon = "✅" if hotel_booked else "⬜"
    st.sidebar.markdown(f"{icon} **Hotels**")

def handle_transportation_booking(journey_legs):
    """Handle booking for each transportation leg of the journey"""
    st.header("Transportation Booking")
    
    if not journey_legs:
        st.info("No journey legs defined. Please generate an itinerary first.")
        return
        
    for i, leg in enumerate(journey_legs):
        leg_id = f"{leg['type']}_{i}"
        is_complete = st.session_state.get(f"booking_complete_{leg_id}", False)
        
        with st.expander(f"Leg {i+1}: {leg['from']} to {leg['to']} via {leg['type'].title()}", expanded=not is_complete):
            if is_complete:
                st.success(f"✅ Booking completed for this leg")
                continue
                
            st.markdown(f"**From:** {leg['from']}  |  **To:** {leg['to']}  |  **Transport:** {leg['type'].title()}")
            
            # Generate options based on transportation type
            if leg['type'].lower() == 'flight':
                options = generate_flight_options(leg['from'], leg['to'], leg.get('date', 'tomorrow'))
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
                options = generate_train_options(leg['from'], leg['to'], leg.get('date', 'tomorrow'))
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
                options = generate_bus_options(leg['from'], leg['to'], leg.get('date', 'tomorrow'))
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
            
            else:
                st.warning(f"Booking for {leg['type']} is not supported yet.")
                continue
            
            # Payment section
            st.subheader(f"Payment for {leg['from']} to {leg['to']}")
            payment_method = display_payment_methods()
            
            # Process payment
            receipt = process_payment(price, f"{leg['type']} Booking", payment_method)
            
            if receipt:
                st.session_state[f"booking_complete_{leg_id}"] = True
                st.success(f"Booking confirmed for {leg['type']} from {leg['from']} to {leg['to']}!")
            else:
                st.error("Payment failed. Please try again.")