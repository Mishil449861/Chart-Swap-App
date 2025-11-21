# Chart-Swap-App

This project converts natural-language instructions or OCR-extracted text from images into high-quality charts using a secure, multi-agent architecture.
It integrates LLM-powered intent detection, OCR data extraction, automated matplotlib code generation, debugging, safe execution, and chart enhancement.

The project includes:

agents.py → Multi-agent orchestration system

chart_swap_app.py → Complete Streamlit app for uploading an image and swapping/modifying the chart

/tmp isolated code execution

Automatic debugging and regeneration

OCR-to-chart conversion

High-resolution image output

IMPORTANT WARNING

**You MUST supply your own API key and JSON credentials.**

**Add your Gemini API key as an environment variable:**

**export GEMINI_API_KEY="your_key_here"**

**If using OCR or Google Cloud Vision, you must also add your own:**

**Service account**

**JSON credential file**

**Correct path exported as:**

**export GOOGLE_APPLICATION_CREDENTIALS="path/to/your_service_account.json"**

DO NOT commit your keys or JSON files to GitHub.

Use .gitignore to keep them private.

**Features**

✓ Natural-language chart transformation

✓ OCR data extraction

✓ Automatic chart code generation

✓ Debugging + safe execution

✓ LLM-driven intent extraction

✓ High-resolution image enhancement

✓ Sandboxed execution (prevents malicious code)

✓ Interactive Streamlit interface

**Architecture Overview**

The system uses a pipeline of specialized agents:

**1. SafetyAgent**

Ensures all generated Python code is safe before running.

Blocks:

File writes

System calls

Shell commands

Infinite loops

Long sleep timers

**2. IntentAgent**

Extracts user intent from natural language.

Examples:

“Increase 2021 value by 10”

“Highlight Q3”

“Sort bars descending”

**3. DataExtractorAgent**

Parses messy OCR text into clean structured data:

Labels

Values

Units

Categories

**4. PlanningAgent**

Decides next steps:

Extract data

Regenerate code

Run the debugger

Execute Python

Enhance final chart

**5. CodeGeneratorAgent**

Uses an LLM to create clean matplotlib code.

Outputs pure Python, not markdown.

**6. DebugAgent**

When execution errors occur:

Reads traceback

Analyzes intent

Regenerates fixed code

**7. ExecutionAgent**

Runs code in a secure subprocess

Adds a plt.savefig() if missing

Timeouts protect against hangs

Returns the generated image

**8. ChartQualityAgent**

Resaves the image at higher DPI for clarity.

**9. AgentStack**

The orchestrator tying all agents together.

**chart_swap_app.py Overview**

chart_swap_app.py is the front-end interface built in Streamlit.

**Users can:**

Upload an image of a chart

OCR text is extracted

Intent is collected from the user

Agents generate new chart code

The chart is executed + displayed

Users download the modified chart

**Core components:**

File uploader

Image display

OCR processing

AgentStack execution

Error logs + regenerated charts

**Example Workflow**

Upload chart image


The app extracts data

You give instructions:

“Convert this bar chart into a line chart and highlight 2020.”

Agents parse intent → extract data → generate code → debug → execute

A high-quality updated chart is returned
