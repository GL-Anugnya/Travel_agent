# 🌍 AI Travel Planning Agent

An AI-powered Travel Planner built using **Streamlit + LangChain + Google Gemini + OpenWeather API**.

This agent combines LLM reasoning with real-time weather data to generate complete trip plans including cultural overview, weather forecast, flight suggestions, hotel options, and a structured day-by-day itinerary.

---

## 🚀 Features
- 🌦 Real-time weather & 5-day forecast (OpenWeather API)
- 🧠 Cultural & historical overview using Gemini
- ✈ AI-generated flight suggestions
- 🏨 AI-generated hotel recommendations
- 📅 Structured day-by-day itinerary
- 🔌 Tool-based architecture using LangChain Agent

---

## 🛠 Tech Stack
- Python
- Streamlit
- LangChain
- Google Gemini (gemini-1.5-flash)
- OpenWeather API

---

## ⚙️ Setup

Create a `.env` file in the project root:

GOOGLE_API_KEY=your_gemini_api_key  
OPENWEATHER_KEY=your_openweather_api_key  

Install dependencies:

pip install -r requirements.txt  

Run the application:

streamlit run app.py  

---

## 🧪 Sample Prompt

Plan a 3-day trip to Tokyo in May

---

Developed for Lab 12B – Agent Development using MCP.
