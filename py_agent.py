OPENAI_API_KEY="x"
GEMINI_API_KEY="x"
DIFFY_KEY="x"



import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from google import genai
from google.genai import types

# 1. Instantiate the OpenAIProvider with the API key
# This explicitly tells the provider which key to use, overriding environment variables.
openai_provider = OpenAIProvider(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


start_system_prompt = """
You are an orchestration agent for experiements to test other agent designs.
You will be given a task to solve.
You will be given a list of models with their decritions.
You will be given a list of tools with their descriptions.
You will also have the ability to write the system prompt for the agent you create.
You will be fed information about the results of different agents you created and make new ones based on that feedback.
After every step you should log a descrition of what you did and why you did it using the log_steps tool.
"""

prompt_to_write_system_prompt = """
Based on the task, tools, available and past results (if any) write a system prompt for the agent you are creating.
"""

pick_models_tools_prompt = """
here is a list of the models and tools you can pick from:
model_dict = {
    'gpt-4o': openai_model_4o, # fast ouput model
    'o3-mini': openai_model_03, # slow thinking model most common
    'o4-mini': openai_model_o4mini, # slow thinking model newest
    'gpt-4-turbo': openai_model_4turbo, # fast ouput model lighter weight
    'o1-mini': openai_model_o1mini, # slow thinking model oldest
}




here is a list of tools to pick from:
tools_dict = {
    'duckduckgo_search': duckduckgo_search_tool,
    'ask_gemini': ask_gemini,
    'ask_gemini_reasoning_model': ask_gemini_reasoning_model,
    "ask_diffbot_api": ask_diffbot_api,
    "get_stock_data_for_year": get_stock_data_for_year,
    "get_current_stock_info": get_current_stock_info,
    "get_historical_stock_data": get_historical_stock_data,
}



you need to come up with a combination which you want to test.
Your ouput needs to be in a very strict format. no extra text is allowed.

First pick a model then the tools.

Your output should be in the following format. note there can only be one model but multiple(even all tools):

MODEL_PICKED: gpt-4o

TOOLS_PICKED: tool1 tool2 tool3

"""

async def call_gemini_api(prompt: str, system_prompt: str, model_name: str = "gemini-2.5-flash"):
    conversation = [{'role': "system", 'content': system_prompt}]
    conversation.append({"role": 'user', "content": prompt})
    input = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(input, generation_config={"temperature": 0.1})
        response = response.text.split() #array
        #response.text.strip() # single: strip is important to prevent extra spaces
        return response
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

# 2. Instantiate the OpenAIModel, passing your custom provider
# Use the model name you were trying before, 'o3-mini' or 'gpt-4o-mini' etc.
# If 'o3-mini' isn't recognized or available, try 'gpt-3.5-turbo' for wider compatibility.
openai_model_4o = OpenAIModel('gpt-4o', provider=openai_provider)
openai_model_03 = OpenAIModel('o3-mini', provider=openai_provider)
openai_model_o4mini = OpenAIModel('o4-mini', provider=openai_provider)
openai_model_4turbo = OpenAIModel('gpt-4-turbo', provider=openai_provider)
openai_model_o1mini = OpenAIModel('o1-mini', provider=openai_provider)

model_dict = {
    'gpt-4o': openai_model_4o,
    'o3-mini': openai_model_03,
    'o4-mini': openai_model_o4mini,
    'gpt-4-turbo': openai_model_4turbo,
    'o1-mini': openai_model_o1mini,
}

#tools_dict = {
#    'duckduckgo_search': duckduckgo_search_tool,
#    'ask_gemini': ask_gemini,
#    'ask_gemini_reasoning_model': ask_gemini_reasoning_model,
#    "ask_diffbot_api": ask_diffbot_api,
#    "get_stock_data_for_year": get_stock_data_for_year,
#    "get_current_stock_info": get_current_stock_info,
#    "get_historical_stock_data": get_historical_stock_data,
#}




class AgentState(BaseModel):
    agent_steps: List[str] = Field(default_factory=list, description="steps taken by the agent during execution.")
    notes: Optional[str] = Field(None, description="General notes for the agent.")


# 3. Initialize the Agent with your configured OpenAIModel
agent = Agent(
    model=model_dict['gpt-4o'], # Pass the instantiated model object
    tools=[duckduckgo_search_tool()],
    system_prompt=start_system_prompt,
)

@agent.tool
async def get_stock_data_for_year(
    ctx: RunContext[AgentState],
    ticker: str,
    year: int,
    interval: str = "1d"
) -> str:
    try:
        # Calculate start and end dates for the given year
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)

        # Convert dates to string format required by yfinance
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        ticker_obj = yf.Ticker(ticker.upper())
        hist_data = ticker_obj.history(start=start_date_str, end=end_date_str, interval=interval)

        if hist_data.empty:
            return f"No historical data found for {ticker.upper()} for the year {year} with interval '{interval}'. This could be due to invalid ticker, or no market activity in that year/interval."

        # Prepare a summary of the data
        # Handle cases where there might be only one data point for the year
        if len(hist_data) == 1:
            first_record = hist_data.iloc[0]
            summary_parts = [
                f"Historical Data for {ticker.upper()} for the year {year} ({interval} interval):",
                f"  Only 1 data point found:",
                f"    Date: {hist_data.index[0].strftime('%Y-%m-%d')}",
                f"    Open: ${first_record['Open']:.2f}, High: ${first_record['High']:.2f}, Low: ${first_record['Low']:.2f}, Close: ${first_record['Close']:.2f}, Volume: {first_record['Volume']:,}"
            ]
        else:
            first_record = hist_data.iloc[0]
            last_record = hist_data.iloc[-1]
            num_records = len(hist_data)

            summary_parts = [
                f"Historical Data for {ticker.upper()} for the year {year} ({interval} interval):",
                f"  Total data points: {num_records}",
                f"  First Record ({hist_data.index[0].strftime('%Y-%m-%d')}):",
                f"    Open: ${first_record['Open']:.2f}, High: ${first_record['High']:.2f}, Low: ${first_record['Low']:.2f}, Close: ${first_record['Close']:.2f}, Volume: {first_record['Volume']:,}",
                f"  Last Record ({hist_data.index[-1].strftime('%Y-%m-%d')}):",
                f"    Open: ${last_record['Open']:.2f}, High: ${last_record['High']:.2f}, Low: ${last_record['Low']:.2f}, Close: ${last_record['Close']:.2f}, Volume: {last_record['Volume']:,}"
            ]

        return "\n".join(summary_parts)

    except ValueError as ve:
        return f"Error: Invalid year or interval format. Details: {ve}"
    except Exception as e:
        return f"An unexpected error occurred while fetching data for {ticker.upper()}: {e}"




@agent.tool
async def get_current_stock_info(ctx: RunContext[AgentState], ticker: str) -> str:
    """Retrieves the current stock price, change, and volume for a given ticker symbol from Yahoo Finance.

    Args:
        ticker: The stock ticker symbol (e.g., "NVDA", "GOOGL").
    Returns:
        A string containing the latest stock information or an error message.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info # This fetches a lot of data, including current price, volume etc.

        if not info or 'regularMarketPrice' not in info:
            return f"Could not retrieve current data for {ticker.upper()}. It might be an invalid ticker or data is unavailable."

        price = info.get('regularMarketPrice')
        change = info.get('regularMarketChange')
        change_percent = info.get('regularMarketChangePercent')
        volume = info.get('regularMarketVolume')

        result_str = (
            f"Current Info for {ticker.upper()}:\n"
            f"  Price: ${price:,.2f}\n"
            f"  Change: ${change:,.2f} ({change_percent:,.2f}%)\n"
            f"  Volume: {volume:,}"
        )
        ctx.deps.calculation_history.append(f"Stock Info: {ticker.upper()} - {price}")
        return result_str
    except Exception as e:
        return f"Error fetching current stock info for {ticker.upper()}: {e}. Please ensure it's a valid ticker."

@agent.tool
async def get_historical_stock_data(
    ctx: RunContext[AgentState],
    ticker: str,
    period: str = "1mo", # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval: str = "1d" # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
) -> str:
    """Retrieves historical stock data for a given ticker, period, and interval from Yahoo Finance.
    Returns a summary of the data (Open, High, Low, Close for the period).

    Args:
        ticker: The stock ticker symbol (e.g., "AAPL", "MSFT").
        period: The period for historical data (e.g., "1mo" for one month, "1y" for one year).
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        interval: The data interval (e.g., "1d" for daily, "1wk" for weekly).
                  Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
                  Note: 1m interval is only available for the last 7 days.
    Returns:
        A string summarizing the historical data or an error message.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period, interval=interval)

        if hist.empty:
            return f"No historical data found for {ticker.upper()} with period {period} and interval {interval}. Check ticker, period, and interval validity."

        first_day = hist.iloc[0]
        last_day = hist.iloc[-1]

        summary_str = (
            f"Historical Data for {ticker.upper()} ({period}, {interval} interval):\n"
            f"  Period Start ({hist.index[0].strftime('%Y-%m-%d')}): Open ${first_day['Open']:.2f}, High ${first_day['High']:.2f}, Low ${first_day['Low']:.2f}, Close ${first_day['Close']:.2f}\n"
            f"  Period End ({hist.index[-1].strftime('%Y-%m-%d')}): Open ${last_day['Open']:.2f}, High ${last_day['High']:.2f}, Low ${last_day['Low']:.2f}, Close ${last_day['Close']:.2f}"
        )
        ctx.deps.calculation_history.append(f"Historical Stock Data: {ticker.upper()} - {period}")
        return summary_str
    except Exception as e:
        return f"Error fetching historical stock data for {ticker.upper()}: {e}. Ensure ticker, period, and interval are valid."


@agent.tool
async def log_steps(ctx: RunContext[AgentState], input: str) -> str:
    ctx.deps.agent_steps.append(input)


@agent.tool
async def ask_diffbot_api(ctx: RunContext[AgentState], question: str) -> str:
    # call diffbot api
    client = OpenAI(
        base_url="https://llm.diffbot.com/rag/v1",
        api_key=DIFFY_KEY
    )
    completion = client.chat.completions.create(
        model="diffbot-small-xl",
        temperature=0,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return completion.choices[0].message.content



@agent.tool
async def ask_gemini(ctx: RunContext[AgentState], question: str) -> str:
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


@agent.tool
async def ask_gemini_reasoning_model(ctx: RunContext[AgentState], question: str) -> str:
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=question,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


async def main():
    print("Running agent asynchronously...")
    initial_state = AgentState(agent_steps=[], notes="")
    # Call the asynchronous run method using await
    result = await agent.run(
        'give me a full research report on nvidia stock performance.',
        deps=initial_state
    )
    print("Asynchronous Agent Output:")
    print(result.output)


# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())
