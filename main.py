from typing import Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# --- State Model ---
class RegexState(BaseModel):
    description: str
    pattern: Optional[str] = None
    examples_positive: Optional[List[str]] = None
    examples_negative: Optional[List[str]] = None
    validation_passed: Optional[bool] = None
    explanation: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

# --- LLM Utilities ---
def call_openai(prompt: str, system: Optional[str] = None, model: Optional[str] = None) -> str:
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model or MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""

# --- Specialist Agent Stubs (LLM-powered) ---
def generate_regex_agent(state: RegexState) -> RegexState:
    print("[Agent] Generating regex pattern from description...")
    prompt = f"Write a Python regular expression pattern (do not include slashes or quotes) that matches the following description: {state.description}\nJust output the regex pattern only."
    try:
        regex = call_openai(prompt)
        if isinstance(regex, str) and regex:
            regex = regex.strip().splitlines()[0].strip('`"')
        else:
            regex = r".*"
        state.pattern = regex
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        state.pattern = r".*"
    return state

def generate_examples_agent(state: RegexState) -> RegexState:
    print("[Agent] Generating positive/negative examples...")
    prompt = (
        f"Given the regex pattern: {state.pattern}\n"
        f"and the description: {state.description}\n"
        "Generate 3 positive example strings that should match, and 3 negative example strings that should not match. "
        "Return them as JSON with keys 'positive' and 'negative'."
    )
    try:
        import json
        examples_str = call_openai(prompt)
        start = examples_str.find('{')
        end = examples_str.rfind('}') + 1
        if start != -1 and end != -1:
            examples_json = examples_str[start:end]
            examples = json.loads(examples_json)
            state.examples_positive = examples.get('positive', [])
            state.examples_negative = examples.get('negative', [])
        else:
            lines = [l.strip() for l in examples_str.splitlines() if l.strip()]
            state.examples_positive = lines[:3]
            state.examples_negative = lines[3:6]
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        state.examples_positive = ["example1", "example2"]
        state.examples_negative = ["not_a_match", "123"]
    return state

def validate_regex_agent(state: RegexState) -> RegexState:
    print("[Agent] Validating regex against examples...")
    try:
        pattern = re.compile(state.pattern or "")
        pos = all(pattern.fullmatch(ex) for ex in (state.examples_positive or []))
        neg = all(not pattern.fullmatch(ex) for ex in (state.examples_negative or []))
        state.validation_passed = bool(pos and neg)
    except Exception as e:
        print(f"[Validation ERROR] {e}")
        state.validation_passed = False
    return state

def refine_agent(state: RegexState) -> RegexState:
    print("[Agent] Refining regex or examples...")
    state.retries += 1
    prompt = (
        f"The following regex pattern failed validation:\n{state.pattern}\n"
        f"Description: {state.description}\n"
        f"Positive examples: {state.examples_positive}\n"
        f"Negative examples: {state.examples_negative}\n"
        "Suggest a corrected regex pattern (Python syntax, no slashes or quotes). Output only the pattern."
    )
    try:
        regex = call_openai(prompt)
        if isinstance(regex, str) and regex:
            regex = regex.strip().splitlines()[0].strip('`"')
        else:
            regex = r". +"
        state.pattern = regex
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        state.pattern = r". +"
    return state

# --- Langgraph Workflow ---
def build_workflow():
    graph = StateGraph(RegexState)
    graph.add_node("generate_regex", generate_regex_agent)
    graph.add_node("generate_examples", generate_examples_agent)
    graph.add_node("validate_regex", validate_regex_agent)
    graph.add_node("refine", refine_agent)

    graph.add_edge(START, "generate_regex")
    graph.add_edge("generate_regex", "generate_examples")
    graph.add_edge("generate_examples", "validate_regex")

    def validation_router(state: RegexState):
        if state.validation_passed:
            return END
        if state.retries >= state.max_retries:
            return END
        return "refine"

    graph.add_conditional_edges(
        "validate_regex",
        validation_router,
        {"refine": "refine", END: END}
    )
    graph.add_edge("refine", "generate_regex")
    return graph

def main():
    print("=== Regex Agent MVP ===")
    description = input("Describe the regex you want to generate: ")
    state = RegexState(description=description)
    workflow_graph = build_workflow()
    workflow = workflow_graph.compile()
    # Print ASCII diagram of the workflow
    try:
        workflow.get_graph().print_ascii()
    except Exception:
        print("[Could not print ASCII diagram of the workflow]")
    # Print Mermaid code
    try:
        mermaid_code = workflow.get_graph().draw_mermaid()
        print("\nWorkflow Graph (Mermaid):")
        print(mermaid_code)
    except Exception:
        print("[Could not generate Mermaid diagram]")
    # Save PNG image using Mermaid.Ink
    try:
        png_bytes = workflow.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png_bytes)
        print("[Workflow graph image saved as workflow.png]")
    except Exception:
        print("[Could not generate workflow PNG image]")
    result = workflow.invoke(state)
    final_state = RegexState(**result)
    print("\nFinal State:")
    print(final_state.model_dump_json(indent=2))

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("[Warning] Please set your OPENAI_API_KEY environment variable.")
    main()
