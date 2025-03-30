import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

def generate_itinerary(selected_route, start_location, destination, budget, duration, places):
    """
    Generate a detailed day-by-day itinerary based on the selected route.
    
    Args:
        selected_route: The route option chosen by the user
        start_location: The starting location
        destination: The final destination
        budget: The user's budget
        duration: The trip duration in days
        places: Places the user wants to visit (comma-separated string)
        
    Returns:
        A dictionary containing the detailed itinerary
    """
    # Template for generating itinerary
    template = """
    You are a professional travel planner with expertise in creating detailed travel itineraries.
    
    Create a detailed day-by-day itinerary for a trip from {start_location} to {destination} with the following details:
    
    - Budget: {budget}
    - Duration: {duration} days
    - Selected route: {selected_route}
    - Places the traveler wants to visit: {places}
    
    The itinerary should include:
    1. A brief overview of the trip
    2. Day-by-day plan with:
       - Date (start with tomorrow's date)
       - Location for the day
       - Accommodation details
       - Transportation details for each day
       - Activities for morning, afternoon, and evening
       - Meal suggestions (breakfast, lunch, dinner)
    3. Budget breakdown (accommodation, transportation, food, activities, miscellaneous)
    4. Packing suggestions based on the destinations and activities
    
    # JSON format should follow this structure:
    {{
      "overview": "Brief overview of the trip",
      "daily_plan": [
        {{
          "day": 1,
          "date": "YYYY-MM-DD",
          "location": "Location name",
          "accommodation": "Hotel name or type of accommodation",
          "transportation_for_day": "Transportation details for this day",
          "activities": ["Activity 1", "Activity 2", "Activity 3"],
          "meals": ["Breakfast suggestion", "Lunch suggestion", "Dinner suggestion"]
        }}
      ],
      "budget_breakdown": {{
        "accommodation": "Amount in INR",
        "transportation": "Amount in INR",
        "food": "Amount in INR",
        "activities": "Amount in INR",
        "miscellaneous": "Amount in INR"
      }},
      "packing_suggestions": ["Item 1", "Item 2", "Item 3"]
    }}
    
    Ensure that:
    - The itinerary is feasible given the duration and budget.
    - Activities are specific to the locations mentioned and include local attractions.
    - The budget breakdown is realistic and detailed.
    - The packing suggestions are tailored to the specific destinations and planned activities.
    
    Please provide the complete itinerary in the JSON format specified above.
    """
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the language model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    
    # Create a chain
    chain = LLMChain(prompt=prompt, llm=model)
    
    # Generate the itinerary
    itinerary = chain.invoke({
        "selected_route": selected_route["transportation_details"],
        "start_location": start_location,
        "destination": destination,
        "budget": budget,
        "duration": duration,
        "places": places
    })
    
    # Process the result
    try:
        # Extract the JSON part from the response text
        json_text = itinerary["text"]
        # Handle different possible formats from the model
        if "```json" in json_text:
            # If the model returned code-formatted JSON
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        
        result = json.loads(json_text)
        return result
    except Exception as e:
        # Create a fallback response if parsing fails
        print(f"Error parsing JSON: {e}")
        return {
            "overview": "An error occurred while generating the itinerary.",
            "daily_plan": [
                {
                    "day": 1,
                    "date": "2023-06-15",
                    "location": destination,
                    "accommodation": "Hotel recommendation unavailable",
                    "transportation_for_day": selected_route["transportation_details"],
                    "activities": ["Explore local attractions"],
                    "meals": ["Local restaurant"]
                }
            ],
            "budget_breakdown": {
                "accommodation": "Varies",
                "transportation": "Varies",
                "food": "Varies",
                "activities": "Varies",
                "miscellaneous": "Varies"
            },
            "packing_suggestions": ["Clothes appropriate for the weather", "Travel documents", "Personal care items"]
        }