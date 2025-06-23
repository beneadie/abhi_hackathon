OPENAI_API_KEY="x"
GEMINI_API_KEY="x"
DIFFY_KEY="x"


from fpdf import FPDF
import yfinance as yf
import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
#from google import genai
import google.generativeai as genai
from openai import OpenAI
from google.genai import types
from datetime import datetime
# 1. Instantiate the OpenAIProvider with the API key
# This explicitly tells the provider which key to use, overriding environment variables.
openai_provider = OpenAIProvider(api_key=OPENAI_API_KEY)
#gemini_client = genai.Client(api_key=GEMINI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)


start_system_prompt = """
You are an orchestration agent for experiements to test other agent designs.
The purpose of the agents will always be a form of research on a topic.
Note that the end goal is to solve the task as best as possible.
The agent you create will be a research agent that will actually do the research to solve the task.
You will be given a task to solve.
You will be given a list of models with their decritions.
You will be given a list of tools with their descriptions.
You will get to pick from any temperature between 0.0 and 1.0. (note it must be a float).
The temperature chosen will be used for the agent and for you when writing the system prompt and agent prompt.
You will also have the opportunity to write the system prompt for the agent you create.
You will be fed information about the results of different agents you created and make new ones based on that feedback.
After every step you should log a descrition of what you did and why you did it using the log_steps tool.
"""

prompt_to_write_system_prompt = """
Based on the task, model, temperature, tools available and past results (if any), write a system prompt for the agent you are creating. This will just be to define the agent. The instructions will be defined later.
"""

prompt_to_write_agent_prompt = """
Based on the task, model, temperature, tools available and past results (if any), and previosuly stated system prompt, write a prompt for the agent to initatie its task. This will be the first prompt the agent sees."""

pick_models_tools_prompt = """
here is a list of the models and tools you can pick from. You will pick a combination of these as well as a temperature for the model.:
model_dict = {
    'gpt-4o': openai_model_4o, # fast ouput model
    'o3-mini': openai_model_03, # slow thinking model most common
    'o4-mini': openai_model_o4mini, # slow thinking model newest
    'gpt-4-turbo': openai_model_4turbo, # fast ouput model lighter weight
    'o1-mini': openai_model_o1mini, # slow thinking model oldest
}


here is a list of tools to pick from:
tools_dict = {
    'duckduckgo_search': duckduckgo_search_tool, #internet seach
    'ask_gemini': ask_gemini, # ask for help from the google gemini LLM
    'ask_gemini_reasoning_model': ask_gemini_reasoning_model,  # ask for help from the google gemini reasoning LLM
    "ask_diffbot_fact_api": ask_diffbot_fact_api, # ask diffbot, a fact checking company, for more reliable and up to date RAG LLM help
    "get_stock_data_for_year_yahoo_finance": get_stock_data_for_year_yahoo_finance, # gets stock data for a specific year from yahoo finance
    "get_current_stock_info_yahoo_finance": get_current_stock_info_yahoo_finance,  # gets current stock data from yahoo finance
    "get_historical_stock_data_yahoo_finance": get_historical_stock_data_yahoo_finance,  # gets current stock data from yahoo finance at specific dates and time periods
}



you need to come up with a combination which you want to test.
It should be to evaluate performance variations by models and tools.
Try to iteratively improve it.
Your ouput needs to be in a very strict format. no extra text is allowed.

First pick a model then the tools.

Your output should be in the following format. note there can only be one model but multiple(even all tools):

MODEL_TEMPERATURE: 0.5

MODEL_PICKED: gpt-4o

TOOLS_PICKED: tool1 tool2 tool3

"""

def setup_gemini(model_name: str = "gemini-2.5-flash",
                temperature: float = 0.7,
                max_tokens: int = None,
                system_instruction: str = "You are a helpful AI assistant."):  # Set to None to let the model decide
    if max_tokens != None:
        return genai.GenerativeModel(
                        model_name=model_name,
                        generation_config={
                            "temperature": min(1.0, temperature),
                            "max_output_tokens": max_tokens,
                        },
                        safety_settings={
                            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                        },
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(
                            include_thoughts=False
                            )
                        ),
                        system_instruction=system_instruction
                    )
    else:
        return genai.GenerativeModel(
                        model_name=model_name,
                        generation_config={
                            "temperature": min(1.0, temperature),
                        },
                        safety_settings={
                            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                        },
                        system_instruction=system_instruction
                    )



async def parse_model_and_tools_output(output_string: str):
    model_name = ""
    tools_list = []
    temperature = 0.1

    lines = output_string.strip().split('\n')

    for line in lines:

        if line.startswith("MODEL_TEMPERATURE:"):
            # Extract the part after "MODEL_TEMPERATURE: " and strip any whitespace
            temperature = float(line.replace("MODEL_TEMPERATURE:", "").strip())
        if line.startswith("MODEL_PICKED:"):
            # Extract the part after "MODEL_PICKED: " and strip any whitespace
            model_name = line.replace("MODEL_PICKED:", "").strip()
        elif line.startswith("TOOLS_PICKED:"):
            # Extract the part after "TOOLS_PICKED: "
            tools_string = line.replace("TOOLS_PICKED:", "").strip()
            # Split the tools string by spaces to get individual tool names
            # .split() without arguments handles multiple spaces and leading/trailing spaces
            tools_list = tools_string.split()

    return model_name, tools_list, temperature


async def call_gemini_api(input: str, conversation_history: list = [], model_name: str = "gemini-2.5-flash", system_prompt = "You are a helpful AI assistant.", temperature: float = 0.1):
    try:
        print(f"\n--- Starting API Call to {model_name} ---")
        model = setup_gemini(model_name=model_name, temperature=temperature, system_instruction=system_prompt)

        # Convert conversation history to the format expected by Gemini
        gemini_history = []
        for msg in conversation_history:
            gemini_history.append({
                'role': msg['role'],
                'parts': msg['parts'] if isinstance(msg['parts'], list) else [str(msg['parts'])]
            })

        chat = model.start_chat(history=gemini_history)
        response = await chat.send_message_async(input)

        if not response or not hasattr(response, 'text'):
            raise ValueError(f"Unexpected response format: {response}")

        #print(f"API Response: {response.text[:200]}..." if len(response.text) > 200 else f"API Response: {response.text}")
        return response.text

    except Exception as e:
        error_msg = f"Error in call_gemini_api: {str(e)}"
        print(error_msg)
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response content: {e.response.text}")
        raise Exception(error_msg) from e

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


class AgentState(BaseModel):
    agent_steps: List[str] = Field(default_factory=list, description="steps taken by the agent during execution.")
    notes: Optional[str] = Field(None, description="General notes for the agent.")
    calculation_history: List[str] = Field(default_factory=list, description="A history of financial calculations made.")

#temp_agent_for_tools = Agent(
#    model=OpenAIModel('gpt-3.5-turbo', provider=openai_provider), # A temporary model
#    system_prompt="Temporary system prompt for tool registration",
#    tools=[] # No tools needed for this temporary agent
#)
import yfinance as yf

async def get_stock_data_for_year_yahoo_finance(
    ctx: RunContext[AgentState],
    ticker: str,
    year: int,
    interval: str = "1d"
) -> str:
    print("Stock data for year tool called")
    print("ticker: ", ticker)
    print("year: ", year)
    print("interval: ", interval)
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
        print("Error: Invalid year or interval format. Details: {ve}")
        return f"Error: Invalid year or interval format. Details: {ve}"
    except Exception as e:
        print("An unexpected error occurred while fetching data from Yfinance: {e}")
        return f"An unexpected error occurred while fetching data for {ticker.upper()}: {e}"





async def get_current_stock_info_yahoo_finance(ctx: RunContext[AgentState], ticker: str) -> str:
    """Retrieves the current stock price, change, and volume for a given ticker symbol from Yahoo Finance.

    Args:
        ticker: The stock ticker symbol (e.g., "NVDA", "GOOGL").
    Returns:
        A string containing the latest stock information or an error message.
    """
    print("Current stock info tool called")
    print("ticker: ", ticker)
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
        print("Error fetching current stock info for {ticker.upper()}: {e}. Please ensure it's a valid ticker.")
        return f"Error fetching current stock info for {ticker.upper()}: {e}. Please ensure it's a valid ticker."


async def get_historical_stock_data_yahoo_finance(
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
    print("Historical stock data tool called")
    print("ticker: ", ticker)
    print("period: ", period)
    print("interval: ", interval)
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
        print(f"Error fetching historical stock data for {ticker.upper()}: {e}. Ensure ticker, period, and interval are valid.")
        return f"Error fetching historical stock data for {ticker.upper()}: {e}. Ensure ticker, period, and interval are valid."


#@temp_agent_for_tools.tool
#async def log_steps(ctx: RunContext[AgentState], input: str) -> str:
#    ctx.deps.agent_steps.append(input)



async def ask_diffbot_fact_api(ctx: RunContext[AgentState], question: str) -> str:
    """Calls the Diffbot API which is an LLM reinforced with a factchecking RAG database that is up to date.
    It can struggle with larger tasks so try to ask it relatively concise questions.

    Args:
        ctx: The context of the agent.
        question: The question to ask the Diffbot API.

    Returns:
        str: The response from the Diffbot API.
    """
    print("Diffbot tool called")
    print("Question for diffbot: ", question)
    # call diffbot api
    client = OpenAI(
        base_url="https://llm.diffbot.com/rag/v1",
        api_key=DIFFY_KEY
    )
    try:
        completion = client.chat.completions.create(
            model="diffbot-small-xl",
            temperature=0,
            messages=[
                {"role": "user", "content": question}
        ]
    )
    except Exception as e:
        return f"Error calling Diffbot API: {e}"
    return completion.choices[0].message.content




async def ask_gemini(ctx: RunContext[AgentState], question: str) -> str:
    """Calls the Gemini API which is an LLM that might offer an alternative perspective."""
    print("Gemini flash tool called")
    print("Question for Gemini: ", question)
    try:
        return await call_gemini_api(input=question)
    except Exception as e:
        return f"Error calling Gemini API: {e}"



async def ask_gemini_reasoning_model(ctx: RunContext[AgentState], question: str) -> str:
    """Calls the Gemini reasoning model API which is a special model that can do train of thought reasoning to think carefully about a problem."""
    print("Gemini reasoning tool called")
    print("Question for Gemini: ", question)
    try:
        response = await call_gemini_api(input=question, model_name="gemini-2.5-pro")
        return response
    except Exception as e:
        return f"Error calling Gemini API: {e}"

async def async_input(prompt: str = "") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


from time import time

def convert_txt_to_pdf(txt_filepath, pdf_filepath):
    """
    Converts a given text file to a PDF file.

    Args:
        txt_filepath (str): The path to the input .txt file.
        pdf_filepath (str): The path for the output .pdf file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10) # You can adjust font and size

    try:
        with open(txt_filepath, "r", encoding="utf-8") as f:
            for line in f:
                pdf.multi_cell(0, 5, txt=line.strip(), align='L') # 0 for max width, 5 for line height
                # pdf.cell(200, 10, txt=line.strip(), ln=1, align='L') # Alternative for single lines
    except Exception as e:
        print(f"Error reading text file: {e}")
        return False

    try:
        pdf.output(pdf_filepath)
        print(f"Successfully converted '{txt_filepath}' to '{pdf_filepath}'")
        return True
    except Exception as e:
        print(f"Error writing PDF file: {e}")
        return False



# Assuming these are defined elsewhere in your file
# from your_module import (
#     async_input, call_gemini_api, parse_model_and_tools_output,
#     duckduckgo_search_tool, ask_gemini, ask_gemini_reasoning_model,
#     ask_diffbot_fact_api, get_stock_data_for_year, get_current_stock_info,
#     get_historical_stock_data, Agent, AgentState, model_dict,
#     start_system_prompt, pick_models_tools_prompt,
#     prompt_to_write_system_prompt, prompt_to_write_agent_prompt
# )

async def main():
    # Define max_iterations and a way to track improvement
    MAX_ITERATIONS = 5  # Or any number you deem appropriate
    iteration_count = 0
    best_result = None
    best_parameters = {}
    best_quality_score = -1 # Initialize with a very low score

    # Initialize conversation for the meta-agent (the one configuring the research agent)
    meta_agent_conversation_history = [
        {'role': 'user', 'parts': [start_system_prompt]},
        {'role': 'model', 'parts': ['Understood.']}
    ]

    user_input_topic = await async_input("Enter topic you want to generate a research report about: ")
    meta_agent_conversation_history.append({
        'role': 'user',
        'parts': [f'The topic which you will be focusing on configuring agents to research is --- "{user_input_topic}"']
    })

    all_runs_history = []

    while iteration_count < MAX_ITERATIONS:
        print(f"\n--- Starting Iteration {iteration_count + 1} ---")
        try:
            # --- Agent Configuration Phase ---
            # These API calls define the research agent's parameters
            unprocessed_model_and_tools_output = await call_gemini_api(
                input=pick_models_tools_prompt,
                conversation_history=meta_agent_conversation_history,
                temperature=0.5 # this temperature always same for now
            )
            if not unprocessed_model_and_tools_output:
                raise ValueError("Failed to get model and tools output")

            model_name, tools_list, temperature = await parse_model_and_tools_output(unprocessed_model_and_tools_output)

            meta_agent_conversation_history.append({
                'role': 'model',
                'parts': [f'The models and tools which you will be using are --- "{unprocessed_model_and_tools_output}"']
            })

            agent_system_prompt_output = await call_gemini_api(
                input=prompt_to_write_system_prompt,
                conversation_history=meta_agent_conversation_history,
                model_name="gemini-2.5-pro", # Using 2.5-pro for its reasoning capabilities
                temperature=temperature
            )
            if not agent_system_prompt_output:
                raise ValueError("Failed to get system prompt output")

            meta_agent_conversation_history.append({
                'role': 'model',
                'parts': [f'The system prompt for the agent you will be creating is --- "{agent_system_prompt_output}"']
            })

            agent_initiate_prompt = await call_gemini_api(
                input=prompt_to_write_agent_prompt,
                conversation_history=meta_agent_conversation_history,
                model_name="gemini-2.5-pro", # Using 2.5-pro for its reasoning capabilities
                temperature=temperature
            )

            tools_dict = {
                'duckduckgo_search': duckduckgo_search_tool(),
                'ask_gemini': ask_gemini,
                'ask_gemini_reasoning_model': ask_gemini_reasoning_model,
                "ask_diffbot_fact_api": ask_diffbot_fact_api, # Corrected tool name
                "get_stock_data_for_year_yahoo_finance": get_stock_data_for_year_yahoo_finance,
                "get_current_stock_info_yahoo_finance": get_current_stock_info_yahoo_finance,
                "get_historical_stock_data_yahoo_finance": get_historical_stock_data_yahoo_finance,
            }

            chosen_tools = []
            for tool in tools_list:
                if tool in tools_dict:
                    chosen_tools.append(tools_dict[tool])
                else:
                    print(f"Warning: Tool '{tool}' not found in tools_dict. Skipping.")


            # --- Agent Execution Phase ---
            print("Running agent asynchronously...")
            initial_state = AgentState(agent_steps=[], notes="")

            start_time = time()

            print(f"\n\n\n\n\nagent_system_prompt_output: {agent_system_prompt_output}")

            print("Agent initializing")
            agent = Agent(
                model=model_dict[model_name],
                tools=chosen_tools,
                system_prompt=agent_system_prompt_output,
                temperature=temperature
            )

            print(f"\n\n\n\n\nagent_initiate_prompt: {agent_initiate_prompt}")

            result = await agent.run(
                agent_initiate_prompt,
                deps=initial_state
            )
            run_time = time() - start_time
            print("\n\n\n\n\n\n\nAsynchronous Agent Output:")
            print(result.output)
            print(f"\n\n\n\n\n\nTotal time taken: {run_time:.2f} seconds")

            # --- Automated Evaluation and Feedback Phase by Gemini 2.5 Pro ---
            print("\n\n\n\n\n\n\n--- Automated Evaluation Phase ---")

            # 1. Capture current parameters and results
            current_parameters = {
                "model_name": model_name,
                "tools_list": tools_list,
                "agent_system_prompt": agent_system_prompt_output,
                "agent_initiate_prompt": agent_initiate_prompt
            }
            current_run_data = {
                "output": result.output,
                "time_taken": run_time,
                "parameters": current_parameters
            }

            # 2. Automated evaluation prompt
            # Replace your existing evaluation_prompt
            evaluation_prompt = (
                f"You are an expert AI evaluator specialized in optimizing research agents. "
                """here is a list of the models and tools the model can pick from:
                    model_dict = {
                        'gpt-4o': openai_model_4o, # fast ouput model
                        'o3-mini': openai_model_03, # slow thinking model most common
                        'o4-mini': openai_model_o4mini, # slow thinking model newest
                        'gpt-4-turbo': openai_model_4turbo, # fast ouput model lighter weight
                        'o1-mini': openai_model_o1mini, # slow thinking model oldest
                    }


                    here is a list of tools the model can pick from:
                    tools_dict = {
                        'duckduckgo_search': duckduckgo_search_tool, #internet seach
                        'ask_gemini': ask_gemini, # ask for help from the google gemini LLM
                        'ask_gemini_reasoning_model': ask_gemini_reasoning_model,  # ask for help from the google gemini reasoning LLM
                        "ask_diffbot_fact_api": ask_diffbot_fact_api, # ask diffbot, a fact checking company, for more reliable and up to date RAG LLM help
                        "get_stock_data_for_year": get_stock_data_for_year, # gets stock data for a specific year from yahoo finance
                        "get_current_stock_info": get_current_stock_info,  # gets current stock data from yahoo finance
                        "get_historical_stock_data": get_historical_stock_data,  # gets current stock data from yahoo finance at specific dates and time periods
                    }"""
                f"Your task is to analyze the performance of a research agent across multiple iterations. "
                f"The overarching goal is to generate a comprehensive and high-quality research report "
                f"on the topic: \"{user_input_topic}\".\n\n"
                f"Here is the report from the *current* iteration:\n"
                f"```\n{result.output}\n```\n\n"
                f"**CURRENT ITERATION PARAMETERS:**\n"
                f"Model: {model_name}\n"
                f"Tools: {', '.join(tools_list)}\n"
                f"System Prompt: \"\"\"{agent_system_prompt_output}\"\"\"\n"
                f"Initiate Prompt: \"\"\"{agent_initiate_prompt}\"\"\"\n\n"
                # --- Add historical context ---
                + (f"**PAST ITERATIONS FOR COMPARISON:**\n"
                f"Review these previous attempts to understand patterns and evolutionary paths.\n\n"
                if all_runs_history else "")
                + "\n".join([
                    f"--- Iteration {i+1} ---\n"
                    f"Report Output (truncated):\n```\n{run['output'][:500]}...\n```\n" # Truncate for brevity
                    f"Parameters:\n"
                    f"  Model: {run['parameters']['model_name']}\n"
                    f"  Tools: {', '.join(run['parameters']['tools_list'])}\n"
                    f"  System Prompt (truncated): \"\"\"{run['parameters']['agent_system_prompt']}\"\"\"\n"
                    f"  Initiate Prompt (truncated): \"\"\"{run['parameters']['agent_initiate_prompt']}\"\"\"\n"
                    f"  Evaluator's Previous Critique (Summary):\n{run['evaluator_critique_summary']}\n" # Summarize previous critique
                    for i, run in enumerate(all_runs_history)
                ])
                + "\n\n" if all_runs_history else ""
                f"\n\n\n--- End Historical Context ---\n\n\n"

                f"**DETAILED EVALUATION FOR CURRENT ITERATION:**\n"
                f"1. **Strengths:** What aspects of the current report are particularly good? What did this combination of parameters achieve well?\n"
                f"2. **Weaknesses/Areas for Improvement:** Where does the current report fall short? Be specific. Are there factual errors, lack of depth, poor structure, or misinterpretation of the topic?\n"
                f"3. **Tool Utilization Analysis:** How effectively were the chosen tools (e.g., duckduckgo_search, ask_gemini) used? Was a tool overused, underused, or used inappropriately? Did the prompt effectively guide tool use?\n"
                f"4. **Prompt Effectiveness Analysis:** Evaluate the system prompt and initiate prompt. Were they clear, comprehensive, and well-aligned with the task? How could they be improved to steer the agent more effectively?\n"
                f"5. **Model Suitability Analysis:** Was the chosen model appropriate for the task and the complexity of the information? Should a different model be considered (e.g., more reasoning, more creative)?\n\n"
                f"**COMPARATIVE ANALYSIS & STRATEGIC RECOMMENDATIONS (The 'Art'):**\n"
                f"Considering *all* past iterations and the current one, acknowledge that quality is not always a simple quantifiable score, but an art of finding the right combination of strengths. Identify patterns in what worked and what didn't.\n"
                f"Based on this comparative analysis, provide strategic recommendations for the *next* iteration. Focus on the interplay between prompts, models, and tools. Do not just suggest, but *explain the reasoning* behind your recommendations.\n"
                f"Your recommendations should be highly actionable and directly translatable into changes for the research agent's configuration.\n\n"
                f"**Format your response as follows:**\n\n"
                f"**CRITIQUE_CURRENT_ITERATION:**\n"
                f"**Strengths:** [Detailed points]\n"
                f"**Weaknesses:** [Detailed points]\n"
                f"**Tool Utilization:** [Detailed analysis]\n"
                f"**Prompt Effectiveness:** [Detailed analysis]\n"
                f"**Model Suitability:** [Detailed analysis]\n\n"
                f"**COMPARATIVE_ANALYSIS_AND_NEXT_STEPS:**\n"
                f"[Your detailed comparative analysis and strategic recommendations for the meta-agent for the *next* iteration. This should be prose, explaining *why* certain changes are suggested, considering the history. For example: \"Previous iteration X showed strength in Y when using Z tool, but lacked depth. For the next iteration, I recommend modifying the system prompt to emphasize... and perhaps shifting to model A for better reasoning capabilities on complex topics like this.\"]"
            )

            # Call Gemini 2.5 Pro for automated evaluation
            evaluation_response = await call_gemini_api(
                input=evaluation_prompt,
                conversation_history=[], # New conversation for evaluation to keep it focused
                model_name="gemini-2.5-pro",
                temperature=temperature # Higher temperature for more creative/diverse feedback
            )

            if not evaluation_response:
                print("Automated evaluation failed to produce a response. Skipping this iteration's feedback.")
                iteration_count += 1
                continue

            print("\n--- Gemini 2.5 Pro's Evaluation and Suggestions ---")
            print(evaluation_response)

            # 3. Use the entire evaluation response as feedback
            feedback_for_meta_agent = evaluation_response

            # 4. Store results without quality score comparison
            best_result = current_run_data
            best_parameters = current_parameters

            # Log results and parameters to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"run_log_{timestamp}.txt"
            with open(log_filename, "w") as f:
                f.write(f"Run Log - {timestamp}\n")
                f.write("-" * 30 + "\n\n")
                f.write("Model Parameters:\n")
                for key, value in current_parameters.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\nSystem Prompts:\n")
                f.write("  Meta Agent Conversation History:\n")
                for entry in meta_agent_conversation_history:
                    f.write(f"    Role: {entry['role']}\n")
                    # Ensure 'parts' is a list and contains at least one element for safe access
                    part_content = entry['parts'][0] if entry['parts'] else "N/A"
                    f.write(f"\n\n    Parts: {str(part_content)}\n\n\n\n\n\n\n\n\n\n\n\n\n") # Truncate for brevity, convert to string
                f.write(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nAgent System Prompt: {agent_system_prompt_output}\n")
                f.write(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nAgent Initiate Prompt: {agent_initiate_prompt}\n")
                f.write(f"\n\n\n\n\n\n\n\n\n\n\nResults:\n")
                f.write(f"  Output:\n{result.output}\n\n\n\n\n\n\n")
                f.write(f"  Time Taken: {run_time:.2f} seconds\n")
                f.write(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nEvaluation Response:\n{evaluation_response}\n\n\n\n\n\n\n\n\n\n")
                log_filename_pdf = f"run_log_{timestamp}.pdf"
                convert_txt_to_pdf(log_filename, log_filename_pdf)


            print(f"Logged current iteration's data to {log_filename}")

            # 5. Use automated feedback to update 'meta_agent_conversation_history'
            # This is where the meta-agent learns for the next iteration.
            # We append the previous agent's output AND the evaluator's critique/suggestions.
            meta_agent_conversation_history.append({
                'role': 'user',
                'parts': [f'The agent in the previous iteration (configuration: {current_parameters}) produced the following report for the topic "{user_input_topic}":\n\n'
                          f'```\n{result.output}\n```\n\n'
                          f'An expert AI evaluator (Gemini 2.5 Pro) provided the following evaluation and suggestions:\n\n'
                          f'{evaluation_response}\n\n'
                          f'Based on this feedback, please generate *improved* parameters for the research agent for the next iteration. '
                          f'Focus on refining the system prompt, initial prompt, and carefully selecting models and tools to address the points raised by the evaluator.']
            })
            # The next API calls (pick_models_tools_prompt, prompt_to_write_system_prompt, etc.)
            # will use this updated conversation history, allowing the meta-agent to iterate based on LLM feedback.

            iteration_count += 1
            await asyncio.sleep(15)

        except Exception as e:
            print(f"An error occurred during iteration {iteration_count + 1}: {e}")
            print("Attempting to continue to the next iteration despite the error...")
            iteration_count += 1 # Still increment to avoid infinite loop on persistent error
            # You might want to add a more sophisticated error handling, e.g.,
            # if consecutive errors occur, then break.

    print("\n--- Iteration Loop Finished ---")
    if best_result:
        print("\nBest Agent Configuration Found:")
        print(f"Output:\n{best_result['output']}")
        print(f"Parameters:\n{best_parameters}")
    else:
        print("No successful runs completed.")

# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())
