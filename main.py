from typing import Optional, List, Dict, Union
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich.prompt import Prompt
import csv
import json
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from datetime import datetime
import time
import argparse

# Load environment variables from .env
load_dotenv()
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

console = Console()

class RegexExamples(BaseModel):
    positive: List[str]
    negative: List[str]
    
    @classmethod
    def validate(cls, value):
        # Ensure both lists exist, are lists, and all elements are non-empty strings
        if not isinstance(value, dict):
            raise ValueError("Expected a dict with 'positive' and 'negative' keys.")
        for key in ("positive", "negative"):
            if key not in value:
                raise ValueError(f"Missing key: {key}")
            if not isinstance(value[key], list):
                raise ValueError(f"{key} must be a list.")
            for v in value[key]:
                if not isinstance(v, str) or not v.strip():
                    raise ValueError(f"All {key} examples must be non-empty strings.")
        return value

class RegexPattern(BaseModel):
    pattern: str
    explanation: str = ""
    source: str = "llm"  # "catalog", "llm", "user"
    is_valid: bool = False
    confidence: Optional[float] = None
    created_at: datetime = datetime.now()
    flags: Optional[int] = None
    examples: Optional[RegexExamples] = None
    false_negatives: Optional[List[str]] = None
    false_positives: Optional[List[str]] = None
    attempt_history: Optional[List[dict]] = None

class RegexRefinement(BaseModel):
    pattern: str
    explanation: str = ""

class PatternTask(BaseModel):
    type: str
    description: str
    catalog: Optional[dict] = None

class ClarifyDecomposeOutput(BaseModel):
    pattern_tasks: Optional[List[PatternTask]] = None
    clarification: Optional[str] = None

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
    # Multi-pattern fields
    pattern_tasks: Optional[List[Dict]] = None
    current_task_index: int = 0
    results: Optional[List[Dict]] = None
    clarification_needed: bool = False
    clarification_prompt: Optional[str] = None
    # Advanced validation fields
    false_positives: Optional[List[str]] = None  # Negatives that matched
    false_negatives: Optional[List[str]] = None  # Positives that did not match
    catalog_failed: bool = False
    catalog_failure_explanation: Optional[str] = None
    # Per-pattern attempt history
    attempt_history: Optional[List[dict]] = None
    # DRY: store RegexPattern
    regex_pattern: Optional[RegexPattern] = None
    # Propagate non-interactive mode
    non_interactive: bool = False
    verbose: bool = True

    def load_from_pattern(self, pattern: RegexPattern):
        self.pattern = pattern.pattern or ""
        self.explanation = pattern.explanation or ""
        if pattern.examples:
            self.examples_positive = pattern.examples.positive
            self.examples_negative = pattern.examples.negative
        else:
            self.examples_positive = []
            self.examples_negative = []
        self.false_negatives = pattern.false_negatives if pattern.false_negatives is not None else []
        self.false_positives = pattern.false_positives if pattern.false_positives is not None else []
        self.validation_passed = pattern.is_valid if pattern.is_valid is not None else False
        self.regex_pattern = pattern

    def update_pattern_from_state(self):
        if self.regex_pattern:
            if self.pattern is not None:
                self.regex_pattern.pattern = self.pattern
            if self.explanation is not None:
                self.regex_pattern.explanation = self.explanation
            # Always update examples as a RegexExamples object
            self.regex_pattern.examples = RegexExamples(
                positive=self.examples_positive or [],
                negative=self.examples_negative or []
            )
            if self.false_negatives is not None:
                self.regex_pattern.false_negatives = self.false_negatives
            if self.false_positives is not None:
                self.regex_pattern.false_positives = self.false_positives
            if self.validation_passed is not None:
                self.regex_pattern.is_valid = self.validation_passed

CATALOG_PATH = "pattern_catalog.json"

def load_pattern_catalog(path=CATALOG_PATH):
    if not os.path.exists(path):
        # If the file does not exist, create an empty catalog
        with open(path, "w") as f:
            json.dump({}, f, indent=2)
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_pattern_catalog(catalog, path=CATALOG_PATH):
    with open(path, "w") as f:
        json.dump(catalog, f, indent=2)

class RegexAgent:
    def __init__(self, non_interactive: bool = False, verbose: bool = True):
        self.console = console
        self.state = None
        self.PATTERN_CATALOG = load_pattern_catalog()
        self.non_interactive = non_interactive
        self.verbose = verbose

    def add_pattern_to_catalog(self, pattern_type, pattern_dict):
        key = str(pattern_type)
        self.PATTERN_CATALOG[key] = pattern_dict
        save_pattern_catalog(self.PATTERN_CATALOG)
        self.console.print(f"[green]Pattern '{key}' added to catalog and saved to {CATALOG_PATH}.[/green]")

    def clarify_and_decompose_agent(self, state: RegexState) -> RegexState:
        if getattr(state, 'verbose', True):
            self.console.print("[bold cyan][Agent][/bold cyan] Clarifying and decomposing user request...")
        prompt = (
            "You are a regex pattern decomposition assistant.\n"
            "Given a user request, output a JSON object with either a 'pattern_tasks' key (a list of pattern types and descriptions), or a 'clarification' key (a string question if the request is ambiguous).\n"
            "Examples:\n"
            "Request: 'Extract emails and phone numbers'\n"
            "Output: {\"pattern_tasks\": [{\"type\": \"email\", \"description\": \"Match email addresses\"}, {\"type\": \"phone\", \"description\": \"Match phone numbers\"}]}\n"
            "Request: 'Generate an e-mail pattern'\n"
            "Output: {\"pattern_tasks\": [{\"type\": \"email\", \"description\": \"Match email addresses\"}]}\n"
            "Request: 'Find all numbers'\n"
            "Output: {\"pattern_tasks\": [{\"type\": \"number\", \"description\": \"Match numbers (digits)\"}]}\n"
            "Request: 'Extract all patterns'\n"
            "Output: {\"clarification\": 'What kind of patterns do you want to extract? (e.g., emails, phone numbers, dates, etc.)'}\n"
            "Request: 'Extract IPv4 addresses'\n"
            "Output: {\"pattern_tasks\": [{\"type\": \"ipv4\", \"description\": \"Match IPv4 addresses\"}]}\n"
            "Request: 'Find all hexadecimal numbers'\n"
            "Output: {\"pattern_tasks\": [{\"type\": \"hex\", \"description\": \"Match hexadecimal numbers\"}]}\n"
            "Request: 'Extract UUIDs'\n"
            "Output: {\"pattern_tasks\": [{\"type\": \"uuid\", \"description\": \"Match UUIDs\"}]}\n"
            "Request: 'Find all URLs'\n"
            "Output: {\"pattern_tasks\": [{\"type\": \"url\", \"description\": \"Match URLs\"}]}\n"
            "---\n"
            f"USER REQUEST: {state.description}\n"
            "OUTPUT:"
            "\nOutput only the JSON object for the USER REQUEST."
        )
        agent = Agent(MODEL_NAME, output_type=ClarifyDecomposeOutput)
        result = agent.run_sync(prompt)
        output = result.output
        if output.pattern_tasks:
            # Supplement/override with catalog if type is known
            for t in output.pattern_tasks:
                ttype = t.type.lower()
                if ttype in self.PATTERN_CATALOG:
                    t.catalog = self.PATTERN_CATALOG[ttype]
            state.pattern_tasks = [t.model_dump() for t in output.pattern_tasks]
            state.clarification_needed = False
            state.clarification_prompt = None
        elif output.clarification:
            state.clarification_needed = True
            state.clarification_prompt = output.clarification
        else:
            state.pattern_tasks = [
                {"type": "unknown", "description": state.description}
            ]
            state.clarification_needed = False
            state.clarification_prompt = None
        state.results = []
        return state

    def clarification_interrupt(self, state: RegexState) -> RegexState:
        if self.non_interactive:
            # In non-interactive mode, skip clarification and use the current description
            self.console.print("[yellow]Non-interactive mode: skipping clarification and using current description.[/yellow]")
            state.clarification_needed = False
            state.clarification_prompt = None
            return state
        if getattr(state, 'verbose', True):
            self.console.print(Panel(state.clarification_prompt or "Clarification required.", title="[bold yellow]User Clarification Needed"))
        user_input = Prompt.ask("Your clarification")
        state.description = user_input
        state.clarification_needed = False
        state.clarification_prompt = None
        return state

    def user_confirmation(self, state: RegexState) -> RegexState:
        if getattr(state, 'verbose', True):
            self.console.print(Panel("Please confirm the LLM's interpretation of your request.", title="[bold magenta]User Confirmation"))
        if not state.pattern_tasks:
            self.console.print("[red]No pattern tasks found.[/red]")
            state.clarification_needed = True
            state.clarification_prompt = "Could you clarify your request?"
            return state
        table = Table(title="LLM Interpretation")
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Pattern Type", style="green")
        table.add_column("Description", style="white")
        for i, task in enumerate(state.pattern_tasks):
            table.add_row(str(i+1), task.get('type', 'unknown'), task.get('description', ''))
        if getattr(state, 'verbose', True):
            self.console.print(table)
        if self.non_interactive:
            state.clarification_needed = False
            state.clarification_prompt = None
            return state
        answer = Prompt.ask("Is this correct? (Y/n)", default="Y").strip().lower()
        if answer in ("n", "no"):
            state.clarification_needed = True
            state.clarification_prompt = "Could you clarify or rephrase your request?"
        else:
            state.clarification_needed = False
            state.clarification_prompt = None
        return state

    @staticmethod
    def generate_regex_agent(state: RegexState) -> RegexState:
        if getattr(state, 'verbose', True):
            console.print("[bold cyan][Agent][/bold cyan] Generating regex pattern from description...")
        prompt = (
            f"Write a Python regex pattern (no slashes or quotes) for: {state.description}\n"
            "Output as JSON: {\"pattern\": \"...\", \"explanation\": \"...\"}"
        )
        agent = Agent(MODEL_NAME, output_type=RegexPattern)
        result = agent.run_sync(prompt)
        regex_pattern = RegexPattern(
            pattern=result.output.pattern,
            explanation=getattr(result.output, 'explanation', ''),
            source="llm",
            created_at=datetime.now(),
        )
        state.load_from_pattern(regex_pattern)
        return state

    @staticmethod
    def generate_examples_agent(state: RegexState, history_context: str = "") -> RegexState:
        non_interactive = getattr(state, 'non_interactive', False)
        if getattr(state, 'verbose', True):
            console.print("[bold cyan][Agent][/bold cyan] Generating positive/negative examples...")
        prompt = (
            f"Given the regex pattern: {state.pattern}\n"
            f"and the description: {state.description}\n"
            + (f"Recent failed attempts (pattern, false negatives, false positives):\n{history_context}\n" if history_context else "")
            + "Generate exactly 3 positive and 3 negative example strings. Output ONLY a JSON object with two fields: 'positive' (list of 3 strings that should match) and 'negative' (list of 3 strings that should not match). Do not include any explanation or extra text. Example: {\"positive\": [\"123\", \"456\", \"789\"], \"negative\": [\"abc\", \"12a34\", \"one23\"]}"
        )
        agent = Agent(MODEL_NAME, output_type=RegexExamples, output_retries=5)
        try:
            result = agent.run_sync(prompt)
            output = result.output
            if isinstance(output, RegexExamples):
                state.examples_positive = output.positive
                state.examples_negative = output.negative
            elif isinstance(output, dict) and 'positive' in output and 'negative' in output:
                state.examples_positive = output['positive']
                state.examples_negative = output['negative']
            else:
                raise ValueError("LLM output did not match expected schema.")
        except UnexpectedModelBehavior as e:
            if getattr(state, 'verbose', True):
                console.print(f"[red][LLM OUTPUT ERROR][/red] Could not parse LLM output for examples: {e.message}")
            if getattr(e, 'body', None):
                if getattr(state, 'verbose', True):
                    console.print(f"[yellow]Raw LLM output:[/yellow] {e.body}")
            if non_interactive:
                if getattr(state, 'verbose', True):
                    console.print("[yellow]Non-interactive mode: skipping manual example entry.[/yellow]")
                state.examples_positive = []
                state.examples_negative = []
            else:
                if getattr(state, 'verbose', True):
                    console.print("[bold yellow]LLM failed to generate valid examples. Please provide them manually.[/bold yellow]")
                pos = Prompt.ask("Enter 3 positive examples (comma-separated)", default="").strip()
                neg = Prompt.ask("Enter 3 negative examples (comma-separated)", default="").strip()
                state.examples_positive = [s.strip() for s in pos.split(",") if s.strip()][:3]
                state.examples_negative = [s.strip() for s in neg.split(",") if s.strip()][:3]
        except Exception as e:
            if getattr(state, 'verbose', True):
                console.print(f"[red][LLM OUTPUT ERROR][/red] Unexpected error: {e}")
            if non_interactive:
                if getattr(state, 'verbose', True):
                    console.print("[yellow]Non-interactive mode: skipping manual example entry.[/yellow]")
                state.examples_positive = []
                state.examples_negative = []
            else:
                if getattr(state, 'verbose', True):
                    console.print("[bold yellow]LLM failed to generate valid examples. Please provide them manually.[/bold yellow]")
                pos = Prompt.ask("Enter 3 positive examples (comma-separated)", default="").strip()
                neg = Prompt.ask("Enter 3 negative examples (comma-separated)", default="").strip()
                state.examples_positive = [s.strip() for s in pos.split(",") if s.strip()][:3]
                state.examples_negative = [s.strip() for s in neg.split(",") if s.strip()][:3]
        # Always update pattern examples after generation
        if state.regex_pattern:
            state.regex_pattern.examples = RegexExamples(
                positive=state.examples_positive or [],
                negative=state.examples_negative or []
            )
        return state

    @staticmethod
    def validate_regex_agent(state: RegexState) -> RegexState:
        if getattr(state, 'verbose', True):
            console.print("[bold cyan][Agent][/bold cyan] Validating regex against examples...")
        try:
            flags = 0
            if hasattr(state, 'regex_pattern') and state.regex_pattern and getattr(state.regex_pattern, 'flags', None):
                flags = state.regex_pattern.flags or 0
            pattern = re.compile(state.pattern or "", flags)
            false_neg = [ex for ex in (state.examples_positive or []) if not pattern.fullmatch(ex)]
            false_pos = [ex for ex in (state.examples_negative or []) if pattern.fullmatch(ex)]
            state.false_negatives = false_neg
            state.false_positives = false_pos
            state.validation_passed = not (false_neg or false_pos)
            if state.validation_passed:
                state.explanation = "All positive examples matched, all negative examples rejected."
            else:
                expl = []
                if false_neg:
                    expl.append(f"False negatives: {false_neg}")
                if false_pos:
                    expl.append(f"False positives: {false_pos}")
                state.explanation = "; ".join(expl)
        except Exception as e:
            if getattr(state, 'verbose', True):
                console.print(f"[red][Validation ERROR][/red] {e}")
            state.validation_passed = False
            state.false_negatives = state.examples_positive or []
            state.false_positives = []
            state.explanation = f"Validation error: {e}"
        state.update_pattern_from_state()
        # Update attempt history
        if state.regex_pattern:
            if not hasattr(state.regex_pattern, 'attempt_history') or state.regex_pattern.attempt_history is None:
                state.regex_pattern.attempt_history = []
            state.regex_pattern.attempt_history.append({
                'pattern': state.pattern,
                'explanation': state.explanation,
                'examples_positive': list(state.examples_positive or []),
                'examples_negative': list(state.examples_negative or []),
                'false_negatives': list(state.false_negatives or []),
                'false_positives': list(state.false_positives or []),
                'validation_passed': state.validation_passed,
            })
        return state

    @staticmethod
    def feedback_agent(state: RegexState) -> RegexState:
        # If called as instance method, get non_interactive from self; else, default to False
        non_interactive = getattr(state, 'non_interactive', False)
        if getattr(state, 'verbose', True):
            console.print("[bold yellow][Agent][/bold yellow] Feedback on failed validation:")
        if getattr(state, 'catalog_failed', False):
            if getattr(state, 'verbose', True):
                console.print(f"[yellow]The standard (catalog) pattern failed. Now using LLM support to refine the pattern.[/yellow]")
            if getattr(state, 'catalog_failure_explanation', None):
                if getattr(state, 'verbose', True):
                    console.print(f"[yellow]{state.catalog_failure_explanation}[/yellow]")
        if state.false_negatives and getattr(state, 'verbose', True):
            console.print(f"[red]False negatives (should match but did not):[/red] {state.false_negatives}")
        if state.false_positives and getattr(state, 'verbose', True):
            console.print(f"[red]False positives (should not match but did):[/red] {state.false_positives}")
        # If max retries nearly reached, ask user for more examples or clarification
        if state.retries + 1 >= state.max_retries:
            # Show attempt history (per-pattern)
            if hasattr(state, 'attempt_history') and state.attempt_history:
                if getattr(state, 'verbose', True):
                    console.print("[bold magenta]Attempt History:[/bold magenta]")
                table = Table(show_lines=True)
                table.add_column("#", style="cyan", justify="right")
                table.add_column("Pattern", style="magenta")
                table.add_column("Explanation", style="white")
                table.add_column("False Negatives", style="red")
                table.add_column("False Positives", style="red")
                table.add_column("Valid?", style="bold")
                for i, att in enumerate(state.attempt_history):
                    table.add_row(
                        str(i+1),
                        att.get('pattern', '') or '',
                        att.get('explanation', '') or '',
                        ", ".join(att.get('false_negatives', []) or []),
                        ", ".join(att.get('false_positives', []) or []),
                        "✔" if att.get('validation_passed') else "✘"
                    )
                if getattr(state, 'verbose', True):
                    console.print(table)
            # Build history context for LLM (last 3 attempts)
            history_lines = []
            for att in (state.attempt_history or [])[-3:]:
                history_lines.append(
                    f"Pattern: {att.get('pattern', '')}, "
                    f"False negatives: {att.get('false_negatives', [])}, "
                    f"False positives: {att.get('false_positives', [])}"
                )
            history_context = "\n".join(history_lines)
            if non_interactive:
                if getattr(state, 'verbose', True):
                    console.print(f"[yellow]Non-interactive mode: auto-improving by generating new examples and clarification (up to 3 times).[/yellow]")
                # Try up to 3 times to improve description and examples
                for attempt in range(3):
                    # 1. Rephrase/clarify description using LLM
                    improved_description = RegexAgent.clarify_description_llm(state, history_context=history_context)
                    if improved_description and improved_description != state.description:
                        if getattr(state, 'verbose', True):
                            console.print(f"[yellow]Description improved by LLM (attempt {attempt+1}): {improved_description}[/yellow]")
                        state.description = improved_description
                    else:
                        # If no improvement, break early
                        if attempt > 0:
                            break
                    # 2. Regenerate examples with history context
                    RegexAgent.generate_examples_agent(state, history_context=history_context)
                return state
            if getattr(state, 'verbose', True):
                console.print("[bold yellow]Max retries nearly reached. You can add more examples or clarify the intent.")
            add_more = Prompt.ask("Would you like to add more examples or clarify? (y/N)", default="N").strip().lower()
            if add_more in ("y", "yes"):
                pos = Prompt.ask("Add positive examples (comma-separated, or leave blank)", default="").strip()
                neg = Prompt.ask("Add negative examples (comma-separated, or leave blank)", default="").strip()
                if pos:
                    state.examples_positive = (state.examples_positive or []) + [s.strip() for s in pos.split(",") if s.strip()]
                if neg:
                    state.examples_negative = (state.examples_negative or []) + [s.strip() for s in neg.split(",") if s.strip()]
                clar = Prompt.ask("Clarify the intent (or leave blank to keep current description)", default="").strip()
                if clar:
                    state.description = clar
        return state

    @staticmethod
    def clarify_description_llm(state: RegexState, history_context: str = "") -> str:
        """Use the LLM to clarify or rephrase the description based on failures and history."""
        prompt = (
            f"The following regex pattern failed validation multiple times:\n"
            f"Pattern: {state.pattern}\n"
            f"Description: {state.description}\n"
            + (f"Recent failed attempts (pattern, false negatives, false positives):\n{history_context}\n" if history_context else "")
            + "Please rephrase or clarify the description to help the LLM generate a better regex. Output ONLY the improved description as a string, no explanation."
        )
        agent = Agent(MODEL_NAME, output_type=str)
        try:
            result = agent.run_sync(prompt)
            if isinstance(result.output, str) and result.output.strip():
                return result.output.strip()
        except Exception as e:
            if getattr(state, 'verbose', True):
                console.print(f"[yellow][LLM clarification error: {e}][/yellow]")
        return state.description

    @staticmethod
    def refine_agent(state: RegexState) -> RegexState:
        if getattr(state, 'verbose', True):
            console.print("[bold cyan][Agent][/bold cyan] Refining regex or examples...")
        state.retries += 1

        # Build attempt history context for LLM
        history_lines = []
        if hasattr(state, 'regex_pattern') and state.regex_pattern and state.regex_pattern.attempt_history:
            for i, att in enumerate(state.regex_pattern.attempt_history[-3:]):  # last 3 attempts
                history_lines.append(
                    f"Attempt {i+1}: Pattern: {att.get('pattern', '')}, "
                    f"False negatives: {att.get('false_negatives', [])}, "
                    f"False positives: {att.get('false_positives', [])}"
                )
        history_context = "\n".join(history_lines)

        prompt = (
            (f"The following standard (catalog) regex pattern failed validation and is being refined with LLM support.\n"
             f"Catalog failure explanation: {str(getattr(state, 'catalog_failure_explanation', '') or '')}\n") if getattr(state, 'catalog_failed', False) else ""
            f"The following regex pattern failed validation:\n{str(state.pattern or '')}\n"
            f"Description: {str(state.description or '')}\n"
            f"Positive examples: {str(state.examples_positive or [])}\n"
            f"Negative examples: {str(state.examples_negative or [])}\n"
            f"Recent attempts and failures:\n{str(history_context or '')}\n"
            "Suggest a corrected regex pattern. Output as JSON: {\"pattern\": \"...\", \"explanation\": \"...\"}"
        )
        agent = Agent(MODEL_NAME, output_type=RegexPattern)
        result = agent.run_sync(prompt)
        if state.regex_pattern:
            state.regex_pattern.pattern = result.output.pattern or ""
            state.regex_pattern.explanation = getattr(result.output, 'explanation', '') or ""
            state.regex_pattern.source = "llm"
            state.regex_pattern.created_at = datetime.now()
            state.load_from_pattern(state.regex_pattern)
        # Update attempt history
        if state.regex_pattern:
            if not hasattr(state.regex_pattern, 'attempt_history') or state.regex_pattern.attempt_history is None:
                state.regex_pattern.attempt_history = []
            state.regex_pattern.attempt_history.append({
                'pattern': state.pattern,
                'explanation': state.explanation,
                'examples_positive': list(state.examples_positive or []),
                'examples_negative': list(state.examples_negative or []),
                'false_negatives': list(state.false_negatives or []),
                'false_positives': list(state.false_positives or []),
                'validation_passed': state.validation_passed,
            })
        return state

    @staticmethod
    def build_single_pattern_workflow():
        graph = StateGraph(RegexState)
        graph.add_node("generate_regex", RegexAgent.generate_regex_agent)
        graph.add_node("generate_examples", RegexAgent.generate_examples_agent)
        graph.add_node("validate_regex", RegexAgent.validate_regex_agent)
        graph.add_node("feedback", RegexAgent.feedback_agent)
        graph.add_node("refine", RegexAgent.refine_agent)

        graph.add_edge(START, "generate_regex")
        graph.add_edge("generate_regex", "generate_examples")
        graph.add_edge("generate_examples", "validate_regex")

        def validation_router(state: RegexState):
            if state.validation_passed:
                return END
            if state.retries >= state.max_retries:
                return END
            return "feedback"

        graph.add_conditional_edges(
            "validate_regex",
            validation_router,
            {"feedback": "feedback", END: END}
        )
        graph.add_edge("feedback", "refine")
        graph.add_edge("refine", "generate_regex")
        return graph

    def build_clarification_workflow(self):
        graph = StateGraph(RegexState)
        graph.add_node("clarify_and_decompose", self.clarify_and_decompose_agent)
        graph.add_node("clarification_interrupt", self.clarification_interrupt)
        graph.add_node("user_confirmation", self.user_confirmation)
        graph.add_edge(START, "clarify_and_decompose")
        def clarify_router(state: RegexState):
            if state.clarification_needed:
                return "clarification_interrupt"
            return "user_confirmation"
        graph.add_conditional_edges(
            "clarify_and_decompose",
            clarify_router,
            {"clarification_interrupt": "clarification_interrupt", "user_confirmation": "user_confirmation"}
        )
        def confirm_router(state: RegexState):
            if state.clarification_needed:
                return "clarification_interrupt"
            return END
        graph.add_conditional_edges(
            "user_confirmation",
            confirm_router,
            {"clarification_interrupt": "clarification_interrupt", END: END}
        )
        graph.add_edge("clarification_interrupt", "clarify_and_decompose")
        return graph

    def display_workflow_diagram(self, workflow_graph, name, print_ascii=True, save_png=True):
        workflow = workflow_graph.compile()
        if print_ascii:
            try:
                workflow.get_graph().print_ascii()
            except Exception:
                self.console.print(f"[yellow][Could not print ASCII diagram of the {name} workflow][/yellow]")
        if save_png:
            try:
                os.makedirs("images", exist_ok=True)
                png_bytes = workflow.get_graph().draw_mermaid_png()
                with open(f"images/{name}_workflow.png", "wb") as f:
                    f.write(png_bytes)
                self.console.print(f"[green][{name.capitalize()} workflow image saved as images/{name}_workflow.png][/green]")
            except Exception:
                self.console.print(f"[yellow][Could not generate {name} workflow PNG image][/yellow]")

    def generate_markdown_report(self, filename: str = None):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if filename is None or not isinstance(filename, str):
            filename = f"results/run_{now}/report.md"
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        lines = []
        description = getattr(self.state, 'description', '') if self.state else ''
        results = getattr(self.state, 'results', []) if self.state and getattr(self.state, 'results', None) is not None else []
        lines.append(f"# Regex Agent MVP Report\n")
        lines.append(f"**Run Timestamp:** {now} UTC  ")
        lines.append(f"**User Prompt:** {description}  ")
        lines.append(f"**Model:** {MODEL_NAME}\n\n---\n")
        # Summary Table
        lines.append("## Summary Table\n")
        lines.append("| # | Type | Description | Regex | Valid? | # False Neg | # False Pos |")
        lines.append("|---|------|-------------|-------|--------|-------------|-------------|")
        for i, res in enumerate(results):
            regex = res.get('pattern', '')
            valid = '✔️' if res.get('is_valid') else '✘'
            false_neg = res.get('false_negatives', []) or []
            false_pos = res.get('false_positives', []) or []
            lines.append(f"| {i+1} | {res.get('type', 'unknown')} | {res.get('description', '')} | `{regex}` | {valid} | {len(false_neg)} | {len(false_pos)} |")
        lines.append("\n---\n")
        # Per-pattern details
        for i, res in enumerate(results):
            lines.append(f"## Pattern {i+1}: {res.get('type', 'unknown').capitalize()}\n")
            lines.append(f"- **Description:** {res.get('description', '')}")
            lines.append(f"- **Regex:** `{res.get('pattern', '')}`")
            lines.append(f"- **Explanation:** {res.get('explanation', '')}")
            lines.append(f"- **Source:** {res.get('source', '')}")
            lines.append(f"- **Valid:** {'✔️' if res.get('is_valid') else '✘'}\n")
            # Examples
            examples = res.get('examples', {})
            pos = (examples.get('positive', []) if isinstance(examples, dict) else [])
            neg = (examples.get('negative', []) if isinstance(examples, dict) else [])
            lines.append(f"**Examples:**\n- Positive: {', '.join(pos) if pos else '_None_'}\n- Negative: {', '.join(neg) if neg else '_None_'}\n")
            # False negatives/positives
            fn = res.get('false_negatives', []) or []
            fp = res.get('false_positives', []) or []
            lines.append(f"**False Negatives:**  ")
            lines.append(f"{', '.join(fn) if fn else '_None_'}\n")
            lines.append(f"**False Positives:**  ")
            lines.append(f"{', '.join(fp) if fp else '_None_'}\n")
            # Attempt history
            ah = res.get('attempt_history', []) or []
            if ah:
                lines.append(f"**Attempt History:**\n")
                lines.append("| # | Pattern | Explanation | False Negatives | False Positives | Valid? |\n|---|---------|-------------|-----------------|-----------------|--------|")
                for j, att in enumerate(ah):
                    lines.append(f"| {j+1} | `{att.get('pattern', '')}` | {att.get('explanation', '')} | {', '.join(att.get('false_negatives', []) or [])} | {', '.join(att.get('false_positives', []) or [])} | {'✔' if att.get('validation_passed') else '✘'} |")
                lines.append("")
            # LLM-created assets (images, diagrams)
            # If you save images/diagrams, reference them here
            # Example: lines.append(f"![Workflow Diagram](../images/single_pattern_workflow.png)")
            lines.append("---\n")
        # Save file
        with open(filename, "w") as f:
            f.write("\n".join(lines))
        self.console.print(f"[green]Markdown report saved as {filename}[/green]")

    def run(self):
        # Display diagrams for the clarification and single-pattern workflows before any user input
        if self.verbose:
            self.console.print("[bold underline cyan]Clarification/Decomposition Workflow[/bold underline cyan]")
        clarification_graph = self.build_clarification_workflow()
        self.display_workflow_diagram(clarification_graph, "clarification", print_ascii=True, save_png=True)

        if self.verbose:
            self.console.print("[bold underline cyan]Single-Pattern Workflow[/bold underline cyan]")
        workflow_graph = self.build_single_pattern_workflow()
        self.display_workflow_diagram(workflow_graph, "single_pattern", print_ascii=True, save_png=True)

        if self.verbose:
            self.console.print(Panel("Regex Agent MVP", title="[bold green]Welcome"))
        description = Prompt.ask("Describe the regex you want to generate")
        self.state = RegexState(description=description, non_interactive=self.non_interactive, verbose=self.verbose)

        # Use LangGraph workflow for clarification/confirmation
        clarification_workflow = clarification_graph.compile()
        self.state = RegexState(**clarification_workflow.invoke(self.state))

        if not self.state.pattern_tasks:
            self.console.print("[red][Error] No pattern tasks found. Exiting.[/red]")
            return

        results_with_idx = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_pattern, task, idx, len(self.state.pattern_tasks)): idx
                for idx, task in enumerate(self.state.pattern_tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results_with_idx.append((idx, result))

        # After processing, sort results by idx and keep only the result dicts
        results_with_idx.sort(key=lambda x: x[0])
        self.state.results = [r for idx, r in results_with_idx]

        # Prepare timestamped run directory
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = f"results/run_{now}"
        os.makedirs(run_dir, exist_ok=True)
        json_path = os.path.join(run_dir, "results.json")
        csv_path = os.path.join(run_dir, "results.csv")
        md_path = os.path.join(run_dir, "report.md")

        # Present all results
        table = Table(title="All Results", show_lines=True)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Pattern Type", style="green")
        table.add_column("Description", style="white")
        table.add_column("Regex", style="magenta")
        table.add_column("Explanation", style="white")
        table.add_column("Source", style="yellow")
        table.add_column("Valid?", style="bold")
        table.add_column("False Negatives", style="red")
        table.add_column("False Positives", style="red")
        for i, res in enumerate(self.state.results):
            table.add_row(
                str(i+1),
                res.get('type', 'unknown'),
                res.get('description', ''),
                Syntax(res.get('pattern', '') or '', "python", theme="ansi_dark").highlight(res.get('pattern', '') or ''),
                res.get('explanation', ''),
                res.get('source', ''),
                "[green]✔[/green]" if res.get('is_valid') else "[red]✘[/red]",
                ", ".join(res.get('false_negatives', []) or []),
                ", ".join(res.get('false_positives', []) or []),
            )
        if self.verbose:
            self.console.print(table)
        # In non-verbose mode, print a concise summary table with prompt, regex, and status
        if not self.verbose:
            summary_table = Table(title="Summary: Prompt, Regex, Status", show_lines=True)
            summary_table.add_column("Prompt", style="white")
            summary_table.add_column("Regex", style="magenta")
            summary_table.add_column("Status", style="bold")
            prompt_str = getattr(self.state, 'description', '')
            for res in self.state.results:
                summary_table.add_row(
                    prompt_str,
                    res.get('pattern', ''),
                    "[green]✔[/green]" if res.get('is_valid') else "[red]✘[/red]"
                )
            self.console.print(summary_table)
        # Export results to CSV and JSON
        try:
            with open(json_path, "w") as f:
                json.dump(self.state.results, f, indent=2, default=str)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["#", "Pattern Type", "Description", "Regex", "Explanation", "Source", "Valid?", "False Negatives", "False Positives"])
                for i, res in enumerate(self.state.results):
                    writer.writerow([
                        i+1,
                        res.get('type', 'unknown'),
                        res.get('description', ''),
                        res.get('pattern', ''),
                        res.get('explanation', ''),
                        res.get('source', ''),
                        "✔" if res.get('is_valid') else "✘",
                        "; ".join(res.get('false_negatives', []) or []),
                        "; ".join(res.get('false_positives', []) or []),
                    ])
            if self.verbose:
                self.console.print(f"[green]Results exported to {json_path} and {csv_path}[/green]")
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow][Could not export results: {e}][/yellow]")
        # Generate Markdown report
        self.generate_markdown_report(md_path)
        self.console.print("[bold green]Done![bold green]")
        # Print summary log
        self.console.print("\n[bold cyan]Run summary:[/bold cyan]")
        self.console.print(f"[bold]Run directory:[/bold] {run_dir}")
        self.console.print(f"[bold]Results JSON:[/bold] {json_path}")
        self.console.print(f"[bold]Results CSV:[/bold] {csv_path}")
        self.console.print(f"[bold]Markdown report:[/bold] {md_path}\n")

    def process_pattern(self, task: Dict, idx: int, total: int) -> Dict:
        if self.verbose:
            self.console.print(Panel(f"Processing pattern {idx+1}/{total}: [bold]{task['description']}[/bold]", title="[blue]Pattern Task"))
        sub_state = RegexState(
            description=task['description'],
            max_retries=3,
            non_interactive=self.non_interactive,
            verbose=self.verbose
        )
        sub_state.attempt_history = []
        ttype = task.get('type', '').lower()
        catalog_failed = False
        catalog_failure_explanation = None
        if 'catalog' in task and ttype in self.PATTERN_CATALOG:
            cat = self.PATTERN_CATALOG[ttype]
            regex_pattern = RegexPattern(
                pattern=cat['pattern'] or "",
                explanation=f"Catalog pattern for {ttype}",
                source="catalog",
                created_at=datetime.now(),
                examples=RegexExamples(
                    positive=cat['examples_positive'],
                    negative=cat['examples_negative']
                ),
            )
            if regex_pattern is not None:
                sub_state.load_from_pattern(regex_pattern)
            sub_state.validation_passed = None
            sub_state.retries = 0
            sub_state = self.validate_regex_agent(sub_state)
            if not sub_state.validation_passed:
                catalog_failed = True
                catalog_failure_explanation = f"[CATALOG ERROR] Pattern failed on its own examples. {sub_state.explanation}"
                sub_state.explanation = catalog_failure_explanation
                if sub_state.regex_pattern:
                    sub_state.regex_pattern.explanation = catalog_failure_explanation
        sub_state.catalog_failed = catalog_failed
        sub_state.catalog_failure_explanation = catalog_failure_explanation
        workflow_graph = self.build_single_pattern_workflow()
        workflow = workflow_graph.compile()
        try:
            result = workflow.invoke(sub_state)
            regex_pattern = getattr(result, 'regex_pattern', None)
            if regex_pattern is None:
                regex_pattern = RegexPattern(
                    pattern=result.get("pattern") or "",
                    explanation=result.get("explanation", "") or "",
                    source="llm",
                    is_valid=bool(result.get("validation_passed", False)),
                    created_at=datetime.now(),
                    examples=RegexExamples(
                        positive=result.get("examples_positive") or [],
                        negative=result.get("examples_negative") or []
                    ),
                    false_negatives=result.get("false_negatives") or [],
                    false_positives=result.get("false_positives") or [],
                    attempt_history=result.get("attempt_history", []),
                )
            return regex_pattern.model_dump()
        except UnexpectedModelBehavior as e:
            if self.verbose:
                self.console.print(f"[red][LLM ERROR][/red] {e}")
            return {
                'type': task.get('type', 'unknown'),
                'description': task.get('description', ''),
                'pattern': '',
                'explanation': f'LLM error: {e}',
                'source': 'llm',
                'is_valid': False,
                'false_negatives': [],
                'false_positives': [],
                'examples': {},
                'attempt_history': [],
                'error': str(e),
            }
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red][Pattern Processing ERROR][/red] {e}")
            return {
                'type': task.get('type', 'unknown'),
                'description': task.get('description', ''),
                'pattern': '',
                'explanation': f'Error: {e}',
                'source': 'llm',
                'is_valid': False,
                'false_negatives': [],
                'false_positives': [],
                'examples': {},
                'attempt_history': [],
                'error': str(e),
            }

    def run_with_prompt(self, prompt: str):
        if not isinstance(prompt, str) or not prompt:
            self.console.print("[red][Error] No prompt provided to run_with_prompt.[/red]")
            return
        self.state = RegexState(description=prompt, non_interactive=self.non_interactive, verbose=self.verbose)
        clarification_graph = self.build_clarification_workflow()
        clarification_workflow = clarification_graph.compile()
        self.state = RegexState(**clarification_workflow.invoke(self.state))
        if not self.state.pattern_tasks:
            self.console.print("[red][Error] No pattern tasks found. Exiting.[/red]")
            return
        results_with_idx = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_pattern, task, idx, len(self.state.pattern_tasks)): idx
                for idx, task in enumerate(self.state.pattern_tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results_with_idx.append((idx, result))
        results_with_idx.sort(key=lambda x: x[0])
        self.state.results = [r for idx, r in results_with_idx]
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = f"results/run_{now}"
        os.makedirs(run_dir, exist_ok=True)
        json_path = os.path.join(run_dir, "results.json")
        csv_path = os.path.join(run_dir, "results.csv")
        md_path = os.path.join(run_dir, "report.md")
        # Output as in run()
        table = Table(title="All Results", show_lines=True)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Pattern Type", style="green")
        table.add_column("Description", style="white")
        table.add_column("Regex", style="magenta")
        table.add_column("Explanation", style="white")
        table.add_column("Source", style="yellow")
        table.add_column("Valid?", style="bold")
        table.add_column("False Negatives", style="red")
        table.add_column("False Positives", style="red")
        for i, res in enumerate(self.state.results):
            table.add_row(
                str(i+1),
                res.get('type', 'unknown'),
                res.get('description', ''),
                Syntax(res.get('pattern', '') or '', "python", theme="ansi_dark").highlight(res.get('pattern', '') or ''),
                res.get('explanation', ''),
                res.get('source', ''),
                "[green]✔[/green]" if res.get('is_valid') else "[red]✘[/red]",
                ", ".join(res.get('false_negatives', []) or []),
                ", ".join(res.get('false_positives', []) or []),
            )
        if self.verbose:
            self.console.print(table)
        # In non-verbose mode, print a concise summary table with prompt, regex, and status
        if not self.verbose:
            summary_table = Table(title="Summary: Prompt, Regex, Status", show_lines=True)
            summary_table.add_column("Prompt", style="white")
            summary_table.add_column("Regex", style="magenta")
            summary_table.add_column("Status", style="bold")
            prompt_str = getattr(self.state, 'description', '')
            for res in self.state.results:
                summary_table.add_row(
                    prompt_str,
                    res.get('pattern', ''),
                    "[green]✔[/green]" if res.get('is_valid') else "[red]✘[/red]"
                )
            self.console.print(summary_table)
        try:
            with open(json_path, "w") as f:
                json.dump(self.state.results, f, indent=2, default=str)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["#", "Pattern Type", "Description", "Regex", "Explanation", "Source", "Valid?", "False Negatives", "False Positives"])
                for i, res in enumerate(self.state.results):
                    writer.writerow([
                        i+1,
                        res.get('type', 'unknown'),
                        res.get('description', ''),
                        res.get('pattern', ''),
                        res.get('explanation', ''),
                        res.get('source', ''),
                        "✔" if res.get('is_valid') else "✘",
                        "; ".join(res.get('false_negatives', []) or []),
                        "; ".join(res.get('false_positives', []) or []),
                    ])
            if self.verbose:
                self.console.print(f"[green]Results exported to {json_path} and {csv_path}[/green]")
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow][Could not export results: {e}][/yellow]")
        self.generate_markdown_report(md_path)
        self.console.print("[bold green]Done![bold green]")
        self.console.print("\n[bold cyan]Run summary:[/bold cyan]")
        self.console.print(f"[bold]Run directory:[/bold] {run_dir}")
        self.console.print(f"[bold]Results JSON:[/bold] {json_path}")
        self.console.print(f"[bold]Results CSV:[/bold] {csv_path}")
        self.console.print(f"[bold]Markdown report:[/bold] {md_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regex Agent MVP")
    parser.add_argument('--prompt', type=str, help='Regex prompt to process')
    parser.add_argument('--prompt-file', type=str, help='File with one prompt per line')
    parser.add_argument('--non-interactive', action='store_true', help='Run without user interaction')
    parser.add_argument('--verbose', action='store_true', help='Show all intermediate output (default: True)')
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[yellow][Warning] Please set your OPENAI_API_KEY environment variable.[/yellow]")

    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        agent = RegexAgent(non_interactive=args.non_interactive, verbose=args.verbose)
        agent.run()
    else:
        for prompt in prompts:
            agent = RegexAgent(non_interactive=True, verbose=args.verbose)
            agent.run_with_prompt(prompt)
