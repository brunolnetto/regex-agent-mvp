# Regex Agent MVP

A modular, LLM-powered agent for designing and validating regular expressions with minimal user input. The workflow is orchestrated using [LangGraph](https://github.com/langchain-ai/langgraph) and visualized in ASCII, Mermaid, and PNG formats.

## 🚀 Achievements

- **LLM-Powered Regex Design:**
  - Uses OpenAI models to generate regex patterns and examples from natural language descriptions.
- **Self-Validating Workflow:**
  - Automatically generates positive/negative examples and validates regexes, refining as needed.
- **Modular Specialist Agents:**
  - Each step (generation, example creation, validation, refinement) is a separate agent for easy extension.
- **LangGraph Orchestration:**
  - Workflow is defined as a graph, supporting loops and conditional logic.
- **Visualization:**
  - Prints the workflow as ASCII art, Mermaid code, and exports a PNG image (using Mermaid.Ink API).
- **Minimal User Input:**
  - User only needs to describe the desired regex in plain language.
- **Configurable LLM Model:**
  - Model name is set via the `MODEL_NAME` environment variable in `.env`.

## 🛠️ Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, modern Python dependency management and installation.

1. **Install [uv](https://github.com/astral-sh/uv):**
   ```bash
   pip install uv
   # or
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Create a `.env` file with your OpenAI API key and (optionally) model name:
     ```env
     OPENAI_API_KEY=sk-...
     MODEL_NAME=gpt-3.5-turbo
     ```

## 🏃 Usage

```bash
python main.py
```
- Enter a description of the regex you want (e.g., "Write a pattern to validate e-mail").
- The agent will:
  1. Generate a regex pattern using an LLM
  2. Generate positive/negative examples
  3. Validate and refine the regex as needed
  4. Print the final state
  5. Visualize the workflow in ASCII, Mermaid, and PNG (`workflow.png`)

## 📊 Visualization Example

- **ASCII Art:**
  - Printed in the terminal
- **Mermaid Code:**
  - Printed in the terminal (can be pasted into Mermaid Live Editor)
- **PNG Image:**
  - Saved as `workflow.png` in the project directory

## 📦 Dependencies
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Pydantic](https://docs.pydantic.dev/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [uv](https://github.com/astral-sh/uv)

## 📝 License
MIT
