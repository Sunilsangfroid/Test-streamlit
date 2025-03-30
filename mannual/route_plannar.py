import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

def generate_route_options(start_location, destination, budget, duration):
    """
    Generate multiple route options between a start location and destination.
    
    Args:
        start_location: The starting location
        destination: The final destination
        budget: The user's budget
        duration: The trip duration in days
        
    Returns:
        A dictionary containing route options with details
    """
    # Template for generating route options
    template = """
    You are a travel planning assistant with expertise in creating detailed route plans.
    
    Generate multiple route options from {start_location} to {destination} with the following constraints:
    - Budget: {budget}
    - Duration: {duration} days
    
    For each route option, provide:
    1. A name/title for the route
    2. Estimated cost
    3. Estimated travel time
    4. Pros of this route
    5. Cons of this route
    6. Detailed transportation information (modes of transport, transfers, etc.)
    
    Generate 3 distinct options that offer different trade-offs (e.g., cost vs. time, scenic vs. direct).
    
    Example output format:
    {{
      "route_options": [
        {{
          "option_id": 1,
          "name": "Direct Flight Route",
          "estimated_cost": "₹15,000",
          "travel_time": "3 hours",
          "pros": ["Fast", "Convenient", "No transfers"],
          "cons": ["More expensive", "Less scenic", "Limited luggage"],
          "transportation_details": "Direct flight from City A to City B, departing at 10:00 AM and arriving at 1:00 PM."
        }},
        {{
          "option_id": 2,
          "name": "Train and Bus Combination",
          "estimated_cost": "₹5,000",
          "travel_time": "8 hours",
          "pros": ["Affordable", "Scenic views", "More luggage allowed"],
          "cons": ["Longer travel time", "Multiple transfers", "Less comfortable"],
          "transportation_details": "Train from City A to City C (3 hours), then bus from City C to City B (5 hours)."
        }},
        {{
          "option_id": 3,
          "name": "Scenic Road Trip",
          "estimated_cost": "₹8,000",
          "travel_time": "10 hours",
          "pros": ["Flexible schedule", "Scenic detours possible", "Can stop anywhere"],
          "cons": ["Longest travel time", "Requires driving", "Parking costs"],
          "transportation_details": "Rent a car in City A, drive the scenic coastal route to City B with recommended stops at viewpoints."
        }}
      ]
    }}
    
    Ensure that:
    - Options are realistic and feasible
    - Cost estimates are reasonable for the locations and transportation modes
    - Each option offers a distinct experience
    - Transportation details are specific and actionable
    
    Please provide the complete route options in the JSON format specified above.
    """
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the language model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    
    # Create a chain
    chain = LLMChain(prompt=prompt, llm=model)
    
    # Generate route options
    route_options = chain.invoke({
        "start_location": start_location,
        "destination": destination,
        "budget": budget,
        "duration": duration
    })
    
    # Process the result
    try:
        # Extract the JSON part from the response text
        json_text = route_options["text"]
        
        # Handle different possible formats from the model
        if "```json" in json_text:
            # If the model returned code-formatted JSON
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            # If the model used generic code formatting
            json_text = json_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(json_text)
        return result
    except Exception as e:
        # Create a fallback response if parsing fails
        print(f"Error parsing JSON: {e}")
        return {
            "route_options": [
                {
                    "option_id": 1,
                    "name": f"Direct Route from {start_location} to {destination}",
                    "estimated_cost": budget,
                    "travel_time": "Varies",
                    "pros": ["Direct route"],
                    "cons": ["Information limited"],
                    "transportation_details": f"Travel directly from {start_location} to {destination}."
                }
            ]
        }
