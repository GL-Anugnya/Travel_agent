import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import (
    get_culture_info,
    get_flight_options,
    get_hotel_options,
    get_itinerary,
    get_weather,
    get_working_gemini_model,
)

load_dotenv()

st.set_page_config(page_title="AI Travel Planning Agent", layout="wide")
st.title("AI Travel Planning Agent")
st.write(
    "Generate a complete travel plan with culture highlights, weather, flights, "
    "hotels, and a day-by-day itinerary."
)

MASTER_INSTRUCTION = """
You are a professional travel planning assistant.
When given a request:
1. Provide cultural & historical paragraph
2. Provide current weather & forecast
3. Suggest travel dates
4. Suggest 3 flight options
5. Suggest 3 hotel options
6. Provide day-by-day itinerary

Use tools when needed.

Return the final answer using exactly these markdown section headers:
## Cultural Overview
## Weather
## Flights
## Hotels
## Itinerary
""".strip()

SECTION_TITLES = [
    "Cultural Overview",
    "Weather",
    "Flights",
    "Hotels",
    "Itinerary",
]


def itinerary_tool_input_parser(raw_input: str) -> str:
    """Allow the agent to pass either 'city|days' or only a city."""
    value = raw_input.strip()
    if not value:
        return "Itinerary request failed: city name is required."

    if "|" in value:
        city_part, days_part = value.split("|", 1)
        city = city_part.strip()
        try:
            days = int(days_part.strip())
        except ValueError:
            return (
                "Itinerary request failed: invalid days. "
                "Use 'city|days', for example 'Tokyo|5'."
            )
        return get_itinerary(city, days)

    return get_itinerary(value, 5)


def extract_section(full_text: str, section_title: str) -> str:
    escaped_titles = [re.escape(title) for title in SECTION_TITLES]
    next_section_pattern = "|".join(escaped_titles)
    pattern = (
        rf"(?:^|\n)##\s*{re.escape(section_title)}\s*\n"
        rf"(.*?)(?=\n##\s*(?:{next_section_pattern})\s*\n|$)"
    )
    match = re.search(pattern, full_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def render_structured_plan(response_text: str) -> None:
    st.subheader("Travel Plan")
    for title in SECTION_TITLES:
        st.markdown(f"### {title}")
        section_text = extract_section(response_text, title)
        if section_text:
            st.write(section_text)
        else:
            st.write("Section not available in model response.")


google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.warning("GOOGLE_API_KEY not found. Add it to your .env file.")
    st.stop()

try:
    selected_model = get_working_gemini_model()
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
    )
except Exception as exc:
    st.error(f"Failed to initialize Gemini LLM: {exc}")
    st.stop()
st.caption(f"Using Gemini model: {selected_model}")

tools = [
    Tool(
        name="CultureInfo",
        func=get_culture_info,
        description="Get one paragraph about the historical and cultural significance of a city. Input: city name.",
    ),
    Tool(
        name="WeatherForecast",
        func=get_weather,
        description="Get current weather and 3-day forecast for a city. Input: city name.",
    ),
    Tool(
        name="FlightOptions",
        func=get_flight_options,
        description="Get 3 realistic economy flight options with approximate prices. Input: destination city name.",
    ),
    Tool(
        name="HotelOptions",
        func=get_hotel_options,
        description="Get 3 mid-range hotel options with location and price per night. Input: city name.",
    ),
    Tool(
        name="ItineraryPlanner",
        func=itinerary_tool_input_parser,
        description=(
            "Create a detailed day-by-day itinerary. "
            "Input format: city|days (example: Paris|4)."
        ),
    ),
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=MASTER_INSTRUCTION,
    debug=True,
)

city = st.text_input("Destination city")
days = st.number_input("Trip length (days)", min_value=1, max_value=30, value=5, step=1)
extra_preferences = st.text_area(
    "Travel preferences (optional)",
    placeholder="Budget, food preferences, pace, interests, etc.",
)

if st.button("Generate Plan"):
    if not city.strip():
        st.warning("Please enter a destination city.")
    else:
        user_request = (
            f"Destination city: {city.strip()}\n"
            f"Trip length: {int(days)} days\n"
            f"Preferences: {extra_preferences.strip() or 'Not specified'}\n"
            f"Use itinerary input format as {city.strip()}|{int(days)} when calling itinerary tool."
        )

        with st.spinner("Generating your travel plan..."):
            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": user_request}]}
                )
                messages = response.get("messages", [])
                final_text = ""
                for message in reversed(messages):
                    content = getattr(message, "content", "")
                    if isinstance(content, str) and content.strip():
                        final_text = content
                        break
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(item.get("text", ""))
                        joined = "\n".join(part for part in parts if part).strip()
                        if joined:
                            final_text = joined
                            break
                if not final_text:
                    final_text = str(response)
                render_structured_plan(final_text)
            except Exception as exc:
                st.error(f"Failed to generate plan: {exc}")

