import json
import copy

from clients.climbing_data_client import ClimbingDataClient
from constants import ClimbStyle

SYSTEM_PROMPT = """
You are an AI climbing guide. Your task is to help people find information about climbing routes and areas, and plan which routes and areas to visit with their party. Don't offer general safety and climbing tips unless the user directly asks for it.
You have access to a climbing route database with the following fields:
- route_name: The name of the climbing route
- sector_name: The name of the climbing area
- description: Detailed description of the route
- location: Geographic coordinates (latitude/longitude)
- rating: User rating from 1-5
- style: The climbing style (boulder, sport, or trad)
- grade: The difficulty grade (e.g., V0, 5.10a)

When users ask about specific climbs, areas, or want recommendations, use the search_climbs function to find relevant information before responding. 
Keep it succinct and don't say anything about climbs you don't find in the database. (You may still provide general information about large areas you know about from training, however.)
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_climbs",
            "description": "Search for climbing routes based on route name",
            "parameters": {
                "type": "object",
                "properties": {
                    "route_name": {
                        "type": "string",
                        "description": "Name of a climbing route",
                    },
                    "sector_name": {
                        "type": "string",
                        "description": "Name of a sector (a climbing area with one or more climbs nearby each other)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the climbing route",
                    },
                    "location": {
                        "type": "object",
                        "properties": {
                            "lat": {
                                "type": "number",
                                "description": "Latitude of the center of a search region",
                            },
                            "lon": {
                                "type": "number",
                                "description": "Longitude of the center of a search region",
                            },
                        },
                        "additionalProperties": False,
                        "required": ["lat", "lon"],
                    },
                    "location_radius_miles": {
                        "type": "number",
                        "description": "The radius of the search region in miles",
                    },
                    "style": {
                        "type": "string",
                        "enum": [style for style in ClimbStyle],
                        "description": "The climbing style",
                    },
                    "rating_min": {
                        "type": "number",
                        "description": "The minimum rating to search for",
                    },
                    "grades": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "A list of climbing grades to search for",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }
]


def search_climbs(climbing_data_client, **kwargs):
    # TODO refactor this module to use an OpenAIClient that implements an abstract class LLMClient instead of hardcoding OpenAI specifics into the chat interface directly. The abstract class can still accept a ClimbingDataClient as an argument for completions.
    return climbing_data_client.search_climbs(**kwargs)


def call_function(function_name, climbing_data_client, **kwargs):
    if function_name == "search_climbs":
        return search_climbs(climbing_data_client, **kwargs)
    else:
        raise Exception("Unknown function name: " + function_name)


def get_completions_stream(
    openai_client, climbing_data_client: ClimbingDataClient, model: str, messages
):
    updated_messages = copy.deepcopy(messages)
    if len(updated_messages) == 0 or updated_messages[0]["role"] != "system":
        updated_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    rag_completion = openai_client.chat.completions.create(
        model=model,
        messages=updated_messages,
        tools=TOOLS,
    )
    tool_calls = rag_completion.choices[0].message.tool_calls
    print("tool calls", tool_calls)
    updated_messages.append(rag_completion.choices[0].message)

    if tool_calls:
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            result = call_function(name, climbing_data_client, **args)
            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )
        print(f"after rag messages: {updated_messages}")
    else:
        print(f"no tools called")
    return openai_client.chat.completions.create(
        model=model, messages=updated_messages, stream=True
    )
