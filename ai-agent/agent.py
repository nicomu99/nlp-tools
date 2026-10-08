"""An AI agent from scratch.

Use the agent:
    python agent.py ask "How many vacation days does Priya have left?"
    python agent.py chat
"""

import sys
import json
from datetime import date
from pathlib import Path

from llm import call_model, has_api_key

ROOT = Path(__file__).resolve().parent
HANDBOOK_DIR = ROOT / "data" / "handbook"
EMPLOYEES_FILE = ROOT / "data" / "employees.json"

VERBOSE = True  # print every tool call as it happens

# Toy company database
EMPLOYEES: list[dict] = []
BOOKINGS: list[dict] = []


def reset_state() -> None:
    """Reload employees from disk and forget all bookings."""
    EMPLOYEES[:] = json.loads(EMPLOYEES_FILE.read_text(encoding="utf-8"))
    BOOKINGS.clear()


reset_state()


def find_employee(name: str) -> dict | None:
    """Case-insensitive match on the full name or the first name."""
    wanted = name.strip().lower()
    for person in EMPLOYEES:
        full = person["name"].lower()
        if wanted == full or wanted == full.split()[0]:
            return person
    return None


# ---------------------------------------------------------------------------
# Step 1: Tools
# ---------------------------------------------------------------------------


def list_handbook() -> str:
    """Return the names of all handbook pages (the .md files in HANDBOOK_DIR),
    sorted, one per line:
        equipment.md
        expenses.md
        ...
    """
    return "\n".join([path.name for path in HANDBOOK_DIR.glob("*.md")])


def read_handbook(page: str) -> str:
    """Return the text of one handbook page."""
    if ".md" not in page:
        page = f"{page}.md"
    file_path = HANDBOOK_DIR / page
    if not file_path.exists():
        available_pages = ", ".join(path.name for path in HANDBOOK_DIR.glob("*.md"))
        raise ValueError(f"Page with name {page} does not exist. Available pages: {available_pages}")
    return file_path.read_text(encoding="UTF-8")


def get_employee(name: str) -> str:
    """Returns an employee record as a JSON string (json.dumps).

    If nobody matches, raises a ValueError with a message that lists the known employees' names.
    """
    employee = find_employee(name)
    if employee is None:
        available_employees = ", ".join(e["name"] for e in EMPLOYEES)
        raise ValueError(
            f"Employee with name {name} could not be found. Available employees: {available_employees}"
        )
    return json.dumps(employee)


# ---------------------------------------------------------------------------
# Step 2: Describe the tools to the model
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_handbook",
        "description": "List available handbook pages from disk by returning their filename. "
                       "Use this tool when figuring out the available handbook page names.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        }
    },
    {
        "name": "read_handbook",
        "description": "Retrieve the contents of one specific handbook page by its name. "
                       "Raises a ValueError if the handbook page name could not be found. "
                       "Use this tool to answer handbook related questions instead of answering from memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "string", "description": "For example 'equipment.md'."},
            },
            "required": ["page"],
        },
    },
    {
        "name": "get_employee",
        "description": "Retrieve the available information of an employee by their name. "
                       "Raises a ValueError if the employee name could not be found. "
                       "Use this tool to answer questions about employees instead of answering from memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "For example 'Mei Tanaka'."},
            },
            "required": ["name"],
        },
    },
]

# ---------------------------------------------------------------------------
# Step 3: Handle tool calls
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "list_handbook": list_handbook,
    "read_handbook": read_handbook,
    "get_employee": get_employee,
}

RISKY_TOOLS = {"book_vacation"}


def ask_user_approval(name: str, tool_input: dict) -> bool:
    """Ask the human at the keyboard for tool usage permission."""
    answer = input(f"\n  Claude wants to run {name}({json.dumps(tool_input)}). Allow? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def run_tool(name: str, tool_input: dict, approve=ask_user_approval) -> tuple[str, bool]:
    """Run one tool and return (result_text, is_error).

    If the tool is unknown, returns ("Error: unknown tool 'xyz'", True) Otherwise the tool function is called
    and its output returned. If that raises ANY exception (bad input, wrong arguments, a bug...), the
    function returns (f"Error: {e}", True) so Claude can read the problem and try again.

    Before running risky tools, user approval is first checked.
    """
    if name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{name}'", True
    if name in RISKY_TOOLS:
        if not approve(name, tool_input):
            return "User declined processing tool function", True
    try:
        result = TOOL_FUNCTIONS[name](**tool_input)
        return result, False
    except Exception as e:
        return f"Error: {e}", True


def handle_tool_calls(content: list[dict], approve=ask_user_approval) -> dict:
    """Run every tool_use block in an assistant message's `content` and return the user message carrying the
    results, in the same order:

        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": ...},
            ...]}

    If VERBOSE, print each call and a short preview of its result, so you can watch the agent work.
    """
    tool_results = []
    for message in content:
        if message["type"] != "tool_use":
            continue
        tool_output, is_error = run_tool(message["name"], message["input"], approve)
        if VERBOSE:
            print(f"Running {message["name"]}: {tool_output}")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": message["id"],
            "content": tool_output,
            "is_error": is_error
        })
    return {"role": "user", "content": tool_results}


# ---------------------------------------------------------------------------
# Step 4: The agent loop
# ---------------------------------------------------------------------------

def run_agent(messages: list[dict], model=call_model, approve=ask_user_approval, max_turns: int = 10) -> str | None:
    """Run the loop on `messages`."""
    current_try = 0
    while current_try < max_turns:
        response = model(SYSTEM_PROMPT, messages, TOOLS)
        if response is None:
            return None
        messages.append({"role": "assistant", "content": response["content"] if response else ""})
        if response["stop_reason"] != "tool_use":
            return "\n".join(content["text"] for content in response["content"] if content["type"] == "text")
        messages.append(handle_tool_calls(response["content"], approve))
        current_try += 1
    return "Agent stopped before finishing the task"


# ---------------------------------------------------------------------------
# Digression: How to handle critical tools
# ---------------------------------------------------------------------------

def book_vacation(name: str, start_date: str, days: int) -> str:
    """Book `days` vacation days for an employee starting on `start_date`
    (a "YYYY-MM-DD" string).

    Raises a ValueError
    - if the employee doesn't exist,
    - the date isn't valid
    - days is not a whole number of at least 1
    - or they don't have enough days left.

    Otherwise, the booking is added to a toy DB.
    """
    date.fromisoformat(start_date)
    if not isinstance(days, int) or days < 1:
        raise ValueError(f"Invalid days value {days}. Please only use non-negative integers.")

    employee = find_employee(name)
    if employee is None:
        raise ValueError(f"Employee with {name} could not be found. Please try again with a different name.")

    vacation_days_left = employee["vacation_days_left"]
    if vacation_days_left < days:
        raise ValueError(
            f"Employee {name} does not have enough vacation days left. Remaining days: {vacation_days_left}"
        )

    employee["vacation_days_left"] = vacation_days_left - days
    BOOKINGS.append({"name": employee["name"], "start_date": start_date, "days": days})
    return f"Booking successfully finished. New vacation day balance: {employee["vacation_days_left"]}."


TOOLS.append({
    "name": "book_vacation",
    "description": "Book a vacation for an employee by providing their name, the desired start date and "
                   "the number of days of their vacation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The employee name, e.g. 'Priya Raman'."},
            "start_date": {"type": "string", "description": "The vacation start date, e.g. '2026-05-05'."},
            "days": {"type": "integer", "description": "Vacation length, e.g. '4'."}
        },
        "required": ["name", "start_date", "days"],
    },
})

# noinspection bad-index
TOOL_FUNCTIONS["book_vacation"] = book_vacation

# ---------------------------------------------------------------------------
# Step 6: System prompt and conversation memory
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a helpful ai agent helpdesk who supports Driftwood Labs employees. 
Today's date is {date.today().isoformat()}.

You help employees by:
- Answering work-related questions by looking up company policies from the company handbook. If the documents
  do not contain the required information, say so.
- Giving information about company employees.
- Helping employees book a vacation. Only book vacations when the user specifically asks you to. Provide information
  about company policy first and tell the user if their desired vacation is against it.

Use your provided tools when responding to questions instead of guessing."""


def chat_turn(messages: list[dict], user_text: str, model=call_model, approve=ask_user_approval) -> str | None:
    """Add the user's message to the running conversation and run the agent.
    Because `messages` keeps growing, Claude remembers earlier turns.
    """
    messages.append({"role": "user", "content": user_text})
    return run_agent(messages, model, approve)


# ---------------------------------------------------------------------------

def print_request(messages: list[dict]) -> None:
    """Print requests."""
    print("No ANTHROPIC_API_KEY set, so here is the request Claude would get:\n")
    print(json.dumps({"system": SYSTEM_PROMPT, "tools": TOOLS, "messages": messages}, indent=2, ensure_ascii=False))


def main() -> None:
    """Main function."""
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    task = " ".join(sys.argv[2:])
    if command == "ask" and task:
        messages = [{"role": "user", "content": task}]
        reply = run_agent(messages)
        if reply is None:
            print_request(messages)
        else:
            print("\n" + reply)
    elif command == "chat":
        if not has_api_key():
            print("chat needs ANTHROPIC_API_KEY. Without one, try: python agent.py play \"...\"")
            return
        messages: list[dict] = []
        print("Chat with the Driftwood Labs helpdesk. Ctrl+C or an empty line to quit.")
        while user_text := input("\nyou> ").strip():
            response = chat_turn(messages, user_text)
            print("\n" + response if response else "")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
