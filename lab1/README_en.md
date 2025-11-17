# SciResearch Agent

A sophisticated research assistant agent that helps find and analyze scientific information from arXiv and Crossref databases, with additional GitHub repository search capabilities.

## Overview

The SciResearch Agent is built using LangGraph and implements a multi-node pipeline for comprehensive scientific research:

- **ReAct Agent Node**: Performs reasoning and uses research tools
- **Summary Node**: Generates research summaries
- **GitHub Node**: Searches for relevant GitHub repositories
- **Writer Node**: Compiles final research reports

## Features

- **Multi-source Research**: Searches arXiv and Crossref databases
- **GitHub Integration**: Finds relevant code repositories
- **Structured Output**: Uses defined data formats for consistent results
- **Tool-based Approach**: Modular tools for different research tasks
- **Visual Pipeline**: Clear graph representation of the workflow

## Pipeline Architecture

![Agent Pipeline](SciResearch_agent.png)

The pipeline follows this flow:
1. **START** → **ReActAgentNode** (initial research and reasoning)
2. **ReActAgentNode** → **SummaryNode** (generate research summary)
3. **ReActAgentNode** → **GuthubNode** (search GitHub repositories)
4. **SummaryNode** → **WriterNode** (compile summary into report)
5. **GuthubNode** → **WriterNode** (include repository findings)
6. **WriterNode** → **END** (final report output)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/blabla9111/NLP_labs.git
```

2. Install dependencies (ensure you have LangChain, LangGraph, and other required packages):
```bash
pip install -r requirements.txt
cd NLP_labs/lab1
```

3. Configure API keys in `config.py`:
```python
BASE_URL = "your_deepseek_api_base"
API_KEY = "your_deepseek_api_key"
MODEL_NAME = "deepseek-chat"
GITHUB_API_TOKEN = "your_github_token"
```

## Usage

Run the main script:

```bash
python main.py
```

The agent will process research queries and generate comprehensive reports including:
- Scientific paper summaries from arXiv and Crossref
- Relevant GitHub repositories
- Structured research findings

## Tools

- **arXiv Search**: Find relevant scientific papers
- **Crossref API**: Access publication metadata
- **Topic Search**: Identify research topics
- **New Query Tool**: Handle follow-up questions
- **GitHub Search**: Find code implementations

## Data Formats

The system uses structured data formats defined in `input_output_formats.py`

## Dependencies

- LangChain
- LangGraph
- DeepSeek Chat Models
- GitHub API wrapper
- arXiv and Crossref API clients