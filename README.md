# Iterative AI Research Agent

This project implements a sophisticated, self-improving AI research agent. It uses a "meta-agent" to iteratively configure, run, and evaluate a "research agent" to generate high-quality reports on a given topic. The system leverages a qualitative feedback loop, where an evaluator LLM analyzes the research agent's output and provides strategic recommendations for the next iteration.

## Key Features

- **Iterative Improvement:** The agent automatically refines its approach over multiple iterations to improve the quality of its research reports.
- **Dynamic Configuration:** A meta-agent dynamically selects the best model (from OpenAI and Google), tools, and temperature for the task at hand.
- **Automated Evaluation:** After each run, Google's Gemini 2.5 Pro model acts as an expert evaluator, providing a detailed critique and actionable feedback.
- **Qualitative Feedback Loop:** The system moves beyond simple numeric scores, using rich, prose-based feedback to guide the meta-agent's decisions for the next iteration.
- **Extensive Toolset:** The research agent can use a variety of tools, including:
    - `duckduckgo_search` for general web searches.
    - `yfinance` for fetching current, historical, and yearly stock data.
    - `Diffbot` for fact-checked, RAG-enhanced answers.
    - `Gemini Flash & Pro` for auxiliary LLM assistance and reasoning.
- **Detailed Logging:** Every iteration's parameters, prompts, output, and evaluation are logged to a timestamped `.txt` file and automatically converted to a `.pdf` in the `results/` directory for easy review.

## How It Works

The architecture consists of three main components:

1.  **The Meta-Agent (Controller):** This is a high-level LLM responsible for strategy. Based on the user's topic and the feedback from previous iterations, it decides on the optimal configuration for the research agent. This includes:
    -   Choosing a model (e.g., `gpt-4o`, `gemini-2.5-pro`).
    -   Selecting a set of tools.
    -   Setting the model's temperature.
    -   Writing the system and initial prompts for the research agent.

2.  **The Research Agent (Worker):** This agent executes the research task based on the configuration provided by the meta-agent. It uses its assigned model, prompts, and tools to generate a research report.

3.  **The Evaluator (Critic):** After the research agent completes its task, a powerful model (Gemini 2.5 Pro) analyzes the generated report. It assesses its strengths, weaknesses, tool usage, and prompt effectiveness, then provides strategic advice for the next iteration. This feedback is then passed back to the Meta-Agent, closing the loop.

This cycle repeats for a set number of iterations, with the goal of progressively enhancing the final report's quality.

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/beneadie/abhi_hackathon
cd abhi_hackathon
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv .venv

# Activate it
# On Windows
.\.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

The project requires API keys for OpenAI, Google Gemini, and Diffbot.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your API keys to this file in the following format:

```env
OPENAI_API_KEY="your_openai_api_key_here"
GEMINI_API_KEY="your_google_gemini_api_key_here"
DIFFBOT_API_KEY="your_diffbot_api_key_here"
```

## How to Run

With the setup complete, you can run the agent with a single command:

```bash
python py_agent.py
```

The script will then prompt you to enter the research topic you want the agent to investigate.

```
Enter topic you want to generate a research report about: [Your topic here]
```

The agent will start its iterative process. You can monitor the progress in your console and find the detailed logs and final PDF reports in the `results/` directory, which will be created automatically if it doesn't exist.
