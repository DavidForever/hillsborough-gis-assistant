import json
import re


def parse_claude_response(raw_response: str) -> dict:
    """
    Split Claude's response into:
    - text_answer: the plain-English explanation
    - highlight_tracts: list of GEOIDs to highlight
    - map_title: title for the Folium map
    - color_field: which field to choropleth
    """
    result = {
        "text_answer": raw_response,
        "highlight_tracts": [],
        "map_title": "Hillsborough County Analysis",
        "color_field": None
    }

    # Extract JSON block from Claude's response
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
        try:
            parsed = json.loads(json_str)
            result["highlight_tracts"] = parsed.get("highlight_tracts", [])
            result["map_title"] = parsed.get("map_title", result["map_title"])
            result["color_field"] = parsed.get("color_field")

            # Clean text answer (remove the JSON block)
            result["text_answer"] = raw_response[:json_match.start()].strip()
        except json.JSONDecodeError:
            pass

    return result