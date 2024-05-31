import json
import os
import httpx
import copy


async def validate_access_token(access_code: str):
    async with httpx.AsyncClient() as client:
        url = (
            "https://backend.buildpicoapps.com/screenshot_to_code/validate_access_token"
        )
        data = json.dumps(
            {
                "access_code": access_code,
                "secret": os.environ.get("PICO_BACKEND_SECRET"),
            }
        )
        headers = {"Content-Type": "application/json"}

        response = await client.post(url, content=data, headers=headers)
        response_data = response.json()

        if response_data["success"]:
            print("Access token is valid.")
            return True
        else:
            print(f"Access token validation failed: {response_data['failure_reason']}")
            return False


def truncate_data_strings(data):
    # Deep clone the data to avoid modifying the original object
    cloned_data = copy.deepcopy(data)

    if isinstance(cloned_data, dict):
        for key, value in cloned_data.items():
            # Recursively call the function if the value is a dictionary or a list
            if isinstance(value, (dict, list)):
                cloned_data[key] = truncate_data_strings(value)
            # Truncate the string if it starts with 'data:'
            elif isinstance(value, str) and value.startswith("data:"):
                cloned_data[key] = value[:20]
    elif isinstance(cloned_data, list):
        # Process each item in the list
        cloned_data = [truncate_data_strings(item) for item in cloned_data]

    return cloned_data
