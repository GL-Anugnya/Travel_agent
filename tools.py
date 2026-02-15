import os
from collections import Counter
from datetime import datetime
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-pro",
]


def _model_priority(model_name: str) -> int:
    name = model_name.lower()
    if "embedding" in name or "aqa" in name:
        return 999
    if "flash" in name and "2.5" in name:
        return 0
    if "flash" in name and "2.0" in name:
        return 1
    if "flash" in name and "1.5" in name:
        return 2
    if "pro" in name and "2.5" in name:
        return 3
    if "pro" in name and "1.5" in name:
        return 4
    if "gemini" in name:
        return 5
    return 999


def _list_generate_models(api_key: str) -> list[str]:
    """List available Gemini models that support generateContent."""
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        response = requests.get(url, params={"key": api_key}, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Unable to list Gemini models: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("Unable to parse Gemini model list response.") from exc

    models = []
    for item in payload.get("models", []):
        methods = item.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue
        raw_name = item.get("name", "")
        clean_name = raw_name.replace("models/", "").strip()
        if clean_name:
            models.append(clean_name)

    unique_models = sorted(set(models), key=_model_priority)
    return unique_models


@lru_cache(maxsize=1)
def get_working_gemini_model() -> str:
    """Return a compatible Gemini model for the current API key."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment.")

    preferred = os.getenv("GEMINI_MODEL", "").strip()
    listed_models = _list_generate_models(api_key)
    candidates = [preferred] if preferred else []
    candidates.extend(MODEL_CANDIDATES)
    candidates.extend(listed_models)

    seen = set()
    ordered_candidates = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            ordered_candidates.append(model)

    last_error = "No model candidates were available."
    for model in ordered_candidates:
        try:
            probe = ChatGoogleGenerativeAI(model=model, temperature=0)
            probe.invoke("Reply with OK.")
            return model
        except Exception as exc:
            last_error = str(exc)

    raise ValueError(f"No compatible Gemini model found. Last error: {last_error}")


def _get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini LLM client using env-based credentials."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment.")

    return ChatGoogleGenerativeAI(model=get_working_gemini_model(), temperature=0.3)


def _invoke_gemini(prompt: str) -> str:
    """Invoke Gemini and return normalized text output."""
    try:
        llm = _get_gemini_llm()
    except ValueError as exc:
        return f"Gemini request failed: {exc}"

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        return f"Gemini request failed: {exc}"

    content = getattr(response, "content", response)
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            else:
                chunks.append(str(item))
        text = "\n".join(chunk for chunk in chunks if chunk).strip()
    else:
        text = str(content).strip()

    return text or "Gemini returned an empty response."


def get_weather(city: str) -> str:
    """Fetch current weather and a 3-day forecast summary for a city."""
    city = city.strip()
    if not city:
        return "Weather lookup failed: city name is required."

    load_dotenv()
    api_key = os.getenv("OPENWEATHER_KEY")
    if not api_key:
        return "Weather lookup failed: OPENWEATHER_KEY is not set in environment."

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.RequestException as exc:
        return f"Weather lookup failed: network error ({exc})."

    if response.status_code != 200:
        try:
            error_payload = response.json()
            error_message = error_payload.get("message", response.text)
        except ValueError:
            error_message = response.text
        return (
            "Weather lookup failed: "
            f"OpenWeather API returned {response.status_code} ({error_message})."
        )

    try:
        payload = response.json()
    except ValueError:
        return "Weather lookup failed: invalid JSON received from OpenWeather API."

    entries = payload.get("list", [])
    if not entries:
        return "Weather lookup failed: no forecast data returned by OpenWeather API."

    city_name = payload.get("city", {}).get("name", city)

    current = entries[0]
    current_temp = current.get("main", {}).get("temp")
    current_desc = "Unknown"
    weather_list = current.get("weather", [])
    if weather_list and isinstance(weather_list, list):
        current_desc = weather_list[0].get("description", "Unknown").capitalize()

    if current_temp is None:
        current_temp_text = "N/A"
    else:
        current_temp_text = f"{current_temp:.1f}"

    current_date = None
    current_dt_txt = current.get("dt_txt")
    if current_dt_txt:
        try:
            current_date = datetime.strptime(current_dt_txt, "%Y-%m-%d %H:%M:%S").date()
        except ValueError:
            current_date = None

    by_day = {}
    for item in entries[1:]:
        dt_txt = item.get("dt_txt")
        if not dt_txt:
            continue
        try:
            item_date = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S").date()
        except ValueError:
            continue

        if current_date and item_date <= current_date:
            continue

        day_bucket = by_day.setdefault(item_date, {"temps": [], "descriptions": []})
        temp = item.get("main", {}).get("temp")
        if isinstance(temp, (int, float)):
            day_bucket["temps"].append(float(temp))

        item_weather = item.get("weather", [])
        if item_weather and isinstance(item_weather, list):
            desc = item_weather[0].get("description")
            if desc:
                day_bucket["descriptions"].append(desc.lower())

    next_days = sorted(by_day.keys())[:3]
    if not next_days:
        forecast_lines = ["- Forecast summary unavailable for next 3 days."]
    else:
        forecast_lines = []
        for day in next_days:
            bucket = by_day[day]
            temps = bucket["temps"]
            descriptions = bucket["descriptions"]

            if temps:
                low = min(temps)
                high = max(temps)
                temp_text = f"{low:.1f} to {high:.1f} deg C"
            else:
                temp_text = "temperature data unavailable"

            if descriptions:
                top_desc = Counter(descriptions).most_common(1)[0][0]
                desc_text = top_desc.capitalize()
            else:
                desc_text = "No description"

            day_label = day.strftime("%a, %b %d")
            forecast_lines.append(f"- {day_label}: {temp_text}, {desc_text}")

    return (
        f"Weather for {city_name}\n"
        f"Current: {current_temp_text} deg C, {current_desc}\n"
        "Next 3 days:\n"
        + "\n".join(forecast_lines)
    )


def get_culture_info(city: str) -> str:
    """Return one paragraph on the city's historical and cultural significance."""
    city = city.strip()
    if not city:
        return "Culture info request failed: city name is required."

    prompt = (
        f"Write exactly one well-structured paragraph about the historical and cultural "
        f"significance of {city}. Keep it factual, engaging, and concise (120-170 words)."
    )

    result = _invoke_gemini(prompt)
    return f"Cultural Snapshot: {city}\n{result}"


def get_flight_options(city: str) -> str:
    """Return 3 realistic economy flight options with approximate pricing."""
    city = city.strip()
    if not city:
        return "Flight options request failed: city name is required."

    prompt = (
        f"Generate 3 realistic economy flight options from major US hubs to {city}. "
        "Use this exact format for each option: "
        "Option X: Airline | Route | Duration | Stops | Approx Price (USD). "
        "Use plausible airlines, routes, and approximate round-trip prices. "
        "Do not include markdown tables."
    )

    result = _invoke_gemini(prompt)
    return f"Economy Flight Options: {city}\n{result}"


def get_hotel_options(city: str) -> str:
    """Return 3 mid-range hotel options with area and nightly pricing."""
    city = city.strip()
    if not city:
        return "Hotel options request failed: city name is required."

    prompt = (
        f"Generate 3 realistic mid-range hotel options in {city}. "
        "For each hotel, include: Hotel Name, area/location, price per night in USD, "
        "and one short reason it is a good choice. "
        "Keep output concise and cleanly formatted as numbered items."
    )

    result = _invoke_gemini(prompt)
    return f"Mid-Range Hotel Options: {city}\n{result}"


def get_itinerary(city: str, days: int) -> str:
    """Return a detailed day-by-day itinerary."""
    city = city.strip()
    if not city:
        return "Itinerary request failed: city name is required."

    if not isinstance(days, int) or days <= 0:
        return "Itinerary request failed: days must be a positive integer."

    prompt = (
        f"Create a detailed {days}-day travel itinerary for {city}. "
        "For each day, include: morning, afternoon, evening, food suggestions, and "
        "one practical local tip. Keep it realistic and traveler-friendly. "
        "Format with clear Day 1, Day 2, etc. headings."
    )

    result = _invoke_gemini(prompt)
    return f"{days}-Day Itinerary: {city}\n{result}"
