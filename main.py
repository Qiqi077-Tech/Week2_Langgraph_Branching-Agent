import os

import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from vectorstore import get_vector_store

load_dotenv()


def get_stock_price(ticker: str) -> str:
    """Get the last closing stock price for a given ticker symbol, e.g. AAPL."""
    stock = yf.Ticker(ticker)
    price = stock.fast_info.get("lastPrice")
    if price is None:
        return f"Could not find a price for ticker '{ticker}'."
    return f"The last price for {ticker.upper()} is ${price:.2f}"


def get_weather(city: str) -> str:
    """Get the current weather for a given city name, e.g. London."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "OPENWEATHER_API_KEY is not set."

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=10,
    )
    if response.status_code != 200:
        return f"Could not find weather for '{city}'."

    data = response.json()
    description = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"The weather in {city} is {description} at {temp}°C."


def search_apple_10k(query: str) -> str:
    """Search Apple's 2025 10-K annual report. Use for questions about Apple's
    financials (revenue, net income, segments), risk factors, or business."""
    docs = get_vector_store().similarity_search(query, k=4)
    results = []
    for doc in docs:
        label = f"PDF page {int(doc.metadata['page'])}"
        if "printed_page" in doc.metadata:
            label += f", printed page {int(doc.metadata['printed_page'])}"
        results.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(results)


model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

agent = create_agent(
    model=model,
    tools=[get_stock_price, get_weather, search_apple_10k],
    system_prompt=(
        "You are a helpful assistant with access to a stock price tool "
        "(`get_stock_price`), a weather tool (`get_weather`), and a tool that "
        "searches Apple's 2025 10-K report (`search_apple_10k`). Decide which "
        "tool to use based on the user's query, using several in sequence if "
        "the query needs them. When answering from the 10-K, cite only pages "
        "whose text you actually used, giving the printed page number (and "
        "the PDF page). Always give a clear, concise answer after using the "
        "tools."
    ),
)


def main() -> None:
    messages: list[dict] = []
    print("Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        reply = messages[-1].content
        if isinstance(reply, list):
            reply = "".join(
                block.get("text", "") for block in reply if isinstance(block, dict)
            )
        print(f"Agent: {reply}")


if __name__ == "__main__":
    main()


## HOMEWORK
## Use APPLE 10 K DOCUMENTS TO ANSWER THE FOLLOWING QUESTION: What is the total revenue for Apple in 2025?  - CHUNK BY PAGE AND STORE IN PINECONE VECTOR STORE
## 1. LOADING THE APPLE 10 K DOCUMENTS TO VECTOR STORE - PINE CONE (INSTALL PINECONE SKILLS + LANGCHAIN SKILLS)
## 2. CREATE A TOOL TO QUERY THE VECTOR STORE

### ADD LANGSMITH Instrumentation to the LangGraph and langchain agents to track the execution of the tasks.