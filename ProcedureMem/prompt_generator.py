## This file contains prompt templates for generating memory and events from trajectories.
def generate_workflow_from_trajectory_prompt(query, trajectory):
    template = "You are a helpful assistant."
    prompt = f'''
You are provided with a query and a trajectory taken to solve the query. The trajectory consists of multiple steps of thought, action and observation. 
Your task is to generate a workflow based on critical steps to help solve similar queries in the future.
A critical step is one that have a significant impact on fulfilling the query, the step action belongs to the set [FlightSearch, GoogleDistanceMatrix, AccommodationSearch, RestaurantSearch, AttractionSearch, CitySearch, NotebookWrite, Planner], and the action's outcome is successful and contributes positively to achieving the query.
Notice: Write the workflow as a natural, coherent paragraph (not as a bullet list or numbered steps). Use clear, concise language to describe what actions should be taken and in what general order.
-----EXAMPLE WORKFLOW----
To solve this query, begin by looking for available flights to the destination city and compare their arrival times. After choosing a suitable flight, check how far the airport is from the city center to plan your transportation. Then, search for accommodation options close to the main attractions or convenient transit. Once lodging is arranged, explore top-rated restaurants nearby and highlight a few must-visit attractions. Finally, organize all selected places and times into a daily plan to ensure a smooth and enjoyable trip.
-----EXAMPLE END----
Query:
{query}
Trajectory:
{trajectory}
Output the workflow without any explanation or context:
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]




def generate_events_from_trajectory_prompt(query, trajectory):
    template = "You are a helpful assistant."
    prompt = f'''
You are given a trajectory consisting of multiple steps. Each step contains thought, action and observation.
Your task is to convert each step into an Event, represented in JSON format, following this structure:
json
[
    {{
    "step": <step id>,
    "pre_state": "<the environment state before the action>",
    "action": "<the action taken>",
    "entity": "agent",
    "new_state": "<the environment state after the action>"
    }},
    ...
]
-----EXAMPLE 1----
Trajectory:
{{"step": 1, "thought": "Thought 1: To create a comprehensive 3-day travel itinerary from Oakland to Tucson, I need to gather information on flights, accommodations, dining options, attractions, and transportation. I'll start by searching for flights from Oakland to Tucson on March 15th, 2022.", "action": "Action 1: FlightSearch[Oakland, Tucson, 2022-03-15]", "observation": "Flight Number  Price DepTime ArrTime  ActualElapsedTime FlightDate OriginCityName DestCityName  Distance\n     F4002752    190   15:07   17:00 1 hours 53 minutes 2022-03-15        Oakland       Tucson     747.0", "state": "Successful"}}
JSON:
{{ "step": 1, "pre_state": "Initial state", "action": "FlightSearch[Oakland, Tucson, 2022-03-15]", "entity": "agent", "new_state": "Flight search completed. Found flights from Oakland to Tucson on 2022-03-15." }}
-----EXAMPLE 2----
Trajectory:
{{ "step": 2, "thought": "Thought 2: Thought 2: I have obtained the flight information from Oakland to Tucson. Now, I will store this information in the Notebook for future reference.", "action": "Action 2: NotebookWrite[Flights from Oakland to Tucson on 2022-03-15]", "observation": "The information has been recorded in Notebook, and its index is 0.", "state": "Successful" }}
JSON:
{{ "step": 2, "pre_state": "Flight search completed. Found flights from Oakland to Tucson on 2022-03-15.", "action": "NotebookWrite[Flights from Oakland to Tucson on 2022-03-15]", "entity": "agent", "new_state": "Flight information recorded in Notebook (index 0)." }}
-----EXAMPLE 3----
Trajectory:
{{"step": 11, "thought": "Thought 11: The transportation information using a taxi from San Diego International Airport to the city center was not valid. However, I have gathered sufficient information on flights, accommodations, dining options, and attractions. I will proceed to use the Planner tool to create a detailed travel plan for the solo traveler, considering the available data and the budget constraints.", "action": "Action 11: I'm sorry, but I cannot proceed with the Planner tool as the transportation information is incomplete. Let me try to gather more information on transportation options within San Diego. I'll estimate the distance, time, and cost for self-driving from the airport to a popular attraction in the city.", "observation": "Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query].", "state": "invalidAction"}}
JSON:
{{"step": 11, "pre_state": "Taxi information from San Diego International Airport to San Diego City Center was not valid.", "action": "Invalid Action", "entity": "agent", "new_state": "Attempted invalid action due to incomplete transportation information."}}
-----EXAMPLES END ----
Trajectory:
{trajectory}
Output the JSON without any explanation or context: 
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]

def generate_workflow_from_events_prompt(query, events):
    template = "You are a helpful assistant."
    prompt = f'''
You are provided with a query and a sequence of events, each consisting of the following elements: step, pre_state, action, entity, and new_state. 
Your task is to identify the critical events that have a significant impact on fulfilling the query.
A critical event is one where the action belongs to the set [FlightSearch, GoogleDistanceMatrix, AccommodationSearch, RestaurantSearch, AttractionSearch, CitySearch, NotebookWrite, Planner], and the action's outcome is successful and contributes positively to achieving the query.
Query:
{query}
Events:
{events}
Output the step IDs in json format without any explanation or context, your response should be a list of step IDs, e.g. [1, 2, 3].
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]