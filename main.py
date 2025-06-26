from typing import Optional, List, Dict
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

# Load environment variables from .env
load_dotenv()
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

console = Console()

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

class RegexAgent:
    def __init__(self):
        self.console = console
        self.state = None

    @staticmethod
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

    def clarify_and_decompose_agent(self, state: RegexState) -> RegexState:
        self.console.print("[bold cyan][Agent][/bold cyan] Clarifying and decomposing user request...")
        prompt = (
            "You are a regex pattern decomposition assistant.\n"
            "Given a user request, output a JSON list of pattern types and descriptions, or a clarification question if the request is ambiguous.\n"
            "Examples:\n"
            "Request: 'Extract emails and phone numbers'\n"
            "Output: [{\"type\": \"email\", \"description\": \"Match email addresses\"}, {\"type\": \"phone\", \"description\": \"Match phone numbers\"}]\n"
            "Request: 'Generate an e-mail pattern'\n"
            "Output: [{\"type\": \"email\", \"description\": \"Match email addresses\"}]\n"
            "Request: 'Find all numbers'\n"
            "Output: [{\"type\": \"number\", \"description\": \"Match numbers (digits)\"}]\n"
            "Request: 'Extract all patterns'\n"
            "Output: Clarification question: 'What kind of patterns do you want to extract? (e.g., emails, phone numbers, dates, etc.)'\n"
            "Request: 'Extract IPv4 addresses'\n"
            "Output: [{\"type\": \"ipv4\", \"description\": \"Match IPv4 addresses\"}]\n"
            "Request: 'Find all hexadecimal numbers'\n"
            "Output: [{\"type\": \"hex\", \"description\": \"Match hexadecimal numbers\"}]\n"
            "Request: 'Extract UUIDs'\n"
            "Output: [{\"type\": \"uuid\", \"description\": \"Match UUIDs\"}]\n"
            "Request: 'Find all URLs'\n"
            "Output: [{\"type\": \"url\", \"description\": \"Match URLs\"}]\n"
            "---\n"
            f"USER REQUEST: {state.description}\n"
            "OUTPUT:"
            "\nOutput only the JSON list or clarification question for the USER REQUEST."
        )
        import json
        try:
            response = self.call_openai(prompt, model=MODEL_NAME)
            if response.strip().startswith('['):
                start = response.find('[')
                end = response.rfind(']') + 1
                tasks = json.loads(response[start:end])
                # Supplement/override with catalog if type is known
                for t in tasks:
                    ttype = t.get('type', '').lower()
                    if ttype in self.PATTERN_CATALOG:
                        t['catalog'] = self.PATTERN_CATALOG[ttype]
                state.pattern_tasks = tasks
                state.clarification_needed = False
                state.clarification_prompt = None
            else:
                state.clarification_needed = True
                state.clarification_prompt = response.strip()
        except Exception as e:
            self.console.print(f"[red][LLM ERROR][/red] {e}")
            state.pattern_tasks = [{"type": "unknown", "description": state.description}]
            state.clarification_needed = False
            state.clarification_prompt = None
        state.results = []
        return state

    def clarification_interrupt(self, state: RegexState) -> RegexState:
        self.console.print(Panel(state.clarification_prompt or "Clarification required.", title="[bold yellow]User Clarification Needed"))
        user_input = Prompt.ask("Your clarification")
        state.description = user_input
        state.clarification_needed = False
        state.clarification_prompt = None
        return state

    def user_confirmation(self, state: RegexState) -> RegexState:
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
        self.console.print(table)
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
        console.print("[bold cyan][Agent][/bold cyan] Generating regex pattern from description...")
        prompt = f"Write a Python regular expression pattern (do not include slashes or quotes) that matches the following description: {state.description}\nJust output the regex pattern only."
        try:
            regex = RegexAgent.call_openai(prompt)
            if isinstance(regex, str) and regex:
                regex = regex.strip().splitlines()[0].strip('`"')
            else:
                regex = r".*"
            state.pattern = regex
        except Exception as e:
            console.print(f"[red][LLM ERROR][/red] {e}")
            state.pattern = r".*"
        return state

    @staticmethod
    def generate_examples_agent(state: RegexState) -> RegexState:
        console.print("[bold cyan][Agent][/bold cyan] Generating positive/negative examples...")
        prompt = (
            f"Given the regex pattern: {state.pattern}\n"
            f"and the description: {state.description}\n"
            "Generate 3 positive example strings that should match, and 3 negative example strings that should not match. "
            "Return them as JSON with keys 'positive' and 'negative'."
        )
        try:
            import json
            examples_str = RegexAgent.call_openai(prompt)
            start = examples_str.find('{')
            end = examples_str.rfind('}') + 1
            if start != -1 and end != -1:
                examples_json = examples_str[start:end]
                examples = json.loads(examples_json)
                state.examples_positive = examples.get('positive', [])
                state.examples_negative = examples.get('negative', [])
            else:
                lines = [line.strip() for line in examples_str.splitlines() if line.strip()]
                state.examples_positive = lines[:3]
                state.examples_negative = lines[3:6]
        except Exception as e:
            console.print(f"[red][LLM ERROR][/red] {e}")
            state.examples_positive = ["example1", "example2"]
            state.examples_negative = ["not_a_match", "123"]
        return state

    @staticmethod
    def validate_regex_agent(state: RegexState) -> RegexState:
        console.print("[bold cyan][Agent][/bold cyan] Validating regex against examples...")
        try:
            pattern = re.compile(state.pattern or "")
            false_neg = [ex for ex in (state.examples_positive or []) if not pattern.fullmatch(ex)]
            false_pos = [ex for ex in (state.examples_negative or []) if pattern.fullmatch(ex)]
            state.false_negatives = false_neg
            state.false_positives = false_pos
            state.validation_passed = not (false_neg or false_pos)
            # Add explanation for feedback
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
            console.print(f"[red][Validation ERROR][/red] {e}")
            state.validation_passed = False
            state.false_negatives = state.examples_positive or []
            state.false_positives = []
            state.explanation = f"Validation error: {e}"
        return state

    @staticmethod
    def feedback_agent(state: RegexState) -> RegexState:
        console.print("[bold yellow][Agent][/bold yellow] Feedback on failed validation:")
        if state.false_negatives:
            console.print(f"[red]False negatives (should match but did not):[/red] {state.false_negatives}")
        if state.false_positives:
            console.print(f"[red]False positives (should not match but did):[/red] {state.false_positives}")
        # If max retries nearly reached, ask user for more examples or clarification
        if state.retries + 1 >= state.max_retries:
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
    def refine_agent(state: RegexState) -> RegexState:
        console.print("[bold cyan][Agent][/bold cyan] Refining regex or examples...")
        state.retries += 1
        prompt = (
            f"The following regex pattern failed validation:\n{state.pattern}\n"
            f"Description: {state.description}\n"
            f"Positive examples: {state.examples_positive}\n"
            f"Negative examples: {state.examples_negative}\n"
            "Suggest a corrected regex pattern (Python syntax, no slashes or quotes). Output only the pattern."
        )
        try:
            regex = RegexAgent.call_openai(prompt)
            if isinstance(regex, str) and regex:
                regex = regex.strip().splitlines()[0].strip('`"')
            else:
                regex = r". +"
            state.pattern = regex
        except Exception as e:
            console.print(f"[red][LLM ERROR][/red] {e}")
            state.pattern = r". +"
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

    # Expanded pattern catalog for known types
    PATTERN_CATALOG = {
        "email": {
            "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "examples_positive": ["john.doe@example.com", "jane_smith123@example.co.uk", "test_email123@test.org"],
            "examples_negative": ["notanemail", "invalid.email@com", "missing@dotcom"]
        },
        "phone": {
            "pattern": r"\d{3}-\d{3}-\d{4}",
            "examples_positive": ["123-456-7890", "555-555-5555", "999-888-7777"],
            "examples_negative": ["12-345-6789", "123-45-6789", "123-456-789"]
        },
        "date": {
            "pattern": r"\d{4}-\d{2}-\d{2}",
            "examples_positive": ["2023-01-01", "1999-12-31", "2020-02-29"],
            "examples_negative": ["01-01-2023", "2023/01/01", "20230101"]
        },
        "number": {
            "pattern": r"\d+",
            "examples_positive": ["123", "4567", "890"], "examples_negative": ["abc", "12a34", "one23"]
        },
        "ipv4": {
            "pattern": r"(\d{1,3}\.){3}\d{1,3}",
            "examples_positive": ["192.168.1.1", "8.8.8.8", "127.0.0.1"], "examples_negative": ["999.999.999.999", "abc.def.ghi.jkl", "1234.5.6.7"]
        },
        "hex": {
            "pattern": r"0x[0-9a-fA-F]+", "examples_positive": ["0x1A3F", "0xabc123", "0xDEADBEEF"], "examples_negative": ["1234", "xyz", "0x"]
        },
        "uuid": {
            "pattern": r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "examples_positive": ["123e4567-e89b-12d3-a456-426614174000"], "examples_negative": ["notauuid", "1234-5678"]
        },
        "url": {
            "pattern": r"https?://[\w.-]+(?:\.[\w\.-]+)+[/#?]?.*",
            "examples_positive": ["https://example.com", "http://test.org/page"], "examples_negative": ["not a url", "ftp://example.com"]
        }
    }

    def run(self):
        # Display diagrams for the clarification and single-pattern workflows before any user input
        self.console.print("[bold underline cyan]Clarification/Decomposition Workflow[/bold underline cyan]")
        clarification_graph = self.build_clarification_workflow()
        self.display_workflow_diagram(clarification_graph, "clarification", print_ascii=True, save_png=True)

        self.console.print("[bold underline cyan]Single-Pattern Workflow[/bold underline cyan]")
        workflow_graph = self.build_single_pattern_workflow()
        self.display_workflow_diagram(workflow_graph, "single_pattern", print_ascii=True, save_png=True)

        self.console.print(Panel("Regex Agent MVP", title="[bold green]Welcome"))
        description = Prompt.ask("Describe the regex you want to generate")
        self.state = RegexState(description=description)

        # Use LangGraph workflow for clarification/confirmation
        clarification_workflow = clarification_graph.compile()
        self.state = RegexState(**clarification_workflow.invoke(self.state))

        if not self.state.pattern_tasks:
            self.console.print("[red][Error] No pattern tasks found. Exiting.[/red]")
            return

        results_with_idx = []
        with Progress() as progress:
            task_id = progress.add_task("[cyan]Processing patterns...", total=len(self.state.pattern_tasks))
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_pattern, task, idx, len(self.state.pattern_tasks)): idx
                    for idx, task in enumerate(self.state.pattern_tasks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    results_with_idx.append((idx, result))
                    progress.advance(task_id)

        # After processing, sort results by idx and keep only the result dicts
        results_with_idx.sort(key=lambda x: x[0])
        self.state.results = [r for idx, r in results_with_idx]

        # Present all results
        table = Table(title="All Results", show_lines=True)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Pattern Type", style="green")
        table.add_column("Description", style="white")
        table.add_column("Regex", style="magenta")
        table.add_column("Positive Examples", style="yellow")
        table.add_column("Negative Examples", style="yellow")
        table.add_column("Valid?", style="bold")
        for i, res in enumerate(self.state.results):
            table.add_row(
                str(i+1),
                res['type'],
                res['description'],
                Syntax(res['pattern'] or '', "python", theme="ansi_dark").highlight(res['pattern'] or ''),
                "\n".join(res['examples_positive'] or []),
                "\n".join(res['examples_negative'] or []),
                "[green]✔[/green]" if res['validation_passed'] else "[red]✘[/red]"
            )
        self.console.print(table)
        # Export results to CSV and JSON
        try:
            os.makedirs("results", exist_ok=True)
            with open("results/results.json", "w") as f:
                json.dump(self.state.results, f, indent=2)
            with open("results/results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["#", "Pattern Type", "Description", "Regex", "Positive Examples", "Negative Examples", "Valid?"])
                for i, res in enumerate(self.state.results):
                    writer.writerow([
                        i+1,
                        res['type'],
                        res['description'],
                        res['pattern'],
                        "; ".join(res['examples_positive'] or []),
                        "; ".join(res['examples_negative'] or []),
                        "✔" if res['validation_passed'] else "✘"
                    ])
            self.console.print("[green]Results exported to results/results.json and results/results.csv[/green]")
        except Exception as e:
            self.console.print(f"[yellow][Could not export results: {e}][/yellow]")
        self.console.print("[bold green]Done![/bold green]")

    def process_pattern(self, task: Dict, idx: int, total: int) -> Dict:
        self.console.print(Panel(f"Processing pattern {idx+1}/{total}: [bold]{task['description']}[/bold]", title="[blue]Pattern Task"))
        sub_state = RegexState(
            description=task['description'],
            max_retries=3
        )
        ttype = task.get('type', '').lower()
        # Use catalog if available, but always run the workflow
        if 'catalog' in task and ttype in self.PATTERN_CATALOG:
            cat = self.PATTERN_CATALOG[ttype]
            sub_state.pattern = cat['pattern']
            sub_state.examples_positive = cat['examples_positive']
            sub_state.examples_negative = cat['examples_negative']
            sub_state.validation_passed = None
            sub_state.retries = 0
            # Validate using the catalog examples first
            sub_state = self.validate_regex_agent(sub_state)
            if not sub_state.validation_passed:
                # If the hard-coded catalog pattern fails its own examples, report and skip retries
                return {
                    "type": task.get("type", "unknown"),
                    "description": task.get("description", ""),
                    "pattern": sub_state.pattern,
                    "examples_positive": sub_state.examples_positive,
                    "examples_negative": sub_state.examples_negative,
                    "validation_passed": False,
                    "explanation": f"[CATALOG ERROR] Pattern failed on its own examples. {sub_state.explanation}",
                    "false_negatives": sub_state.false_negatives,
                    "false_positives": sub_state.false_positives
                }
            # If catalog pattern passes, use as initial state but run the full workflow with user/LLM examples
            # (user/LLM examples will be generated in the workflow)
        # Always run the full workflow for all patterns
        workflow_graph = self.build_single_pattern_workflow()
        workflow = workflow_graph.compile()
        result = workflow.invoke(sub_state)
        return {
            "type": task.get("type", "unknown"),
            "description": task.get("description", ""),
            "pattern": result.get("pattern"),
            "examples_positive": result.get("examples_positive"),
            "examples_negative": result.get("examples_negative"),
            "validation_passed": result.get("validation_passed"),
            "explanation": result.get("explanation", ""),
            "false_negatives": result.get("false_negatives"),
            "false_positives": result.get("false_positives")
        }

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[yellow][Warning] Please set your OPENAI_API_KEY environment variable.[/yellow]")
    agent = RegexAgent()
    agent.run()
