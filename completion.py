import json


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


TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_climbs",
        "description": "Search for climbing routes based on route name",
        "parameters": {
            "type": "object",
            "properties": {
                "route_name": {
                    "type": "string",
                    "description": "Name of a climbing route"
                }
            },
            "required": [
                "route_name"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

def search_climbs(route_name):
    # TODO search ES index
    # TODO search over more fields than just the route name
    return {
        "route_name": route_name,
        "sector_name": "dummy sector",
        "description": "dummy description",
        "location": {"lat": 0, "lon": 0},
        "rating": 0,
        "style": "dummy style",
        "grade": "dummy grade"
    }

def call_function(function_name, **kwargs):
    if function_name == "search_climbs":
        return search_climbs(**kwargs)
    else:
        raise Exception("Unknown function name: " + function_name)

def get_completions_stream(openai_client, model, messages):
    rag_completion = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content": SYSTEM_PROMPT},
            *messages
        ],
        tools=TOOLS,
    )
    print(rag_completion.choices[0].message.tool_calls)
    after_rag_messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        *messages,
        rag_completion.choices[0].message,
    ]
    for tool_call in rag_completion.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        result = call_function(name, **args)
        after_rag_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })
    print(f"after rag messages: {after_rag_messages}")
    return openai_client.chat.completions.create(
        model=model,
        messages=after_rag_messages,
        stream=True
    )
