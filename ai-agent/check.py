"""Check your work one step at a time.

    python check.py 1      # checks step 1
    python check.py all    # checks every step

No API key needed: these checks replace Claude with a "scripted model" that
plays back pre-written responses, so you can test your loop deterministically.
"""

import copy
import importlib
import inspect
import json
import sys
from datetime import date

if "--solution" in sys.argv:
    sys.argv.remove("--solution")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "solution"))
agent = importlib.import_module("agent")
agent.VERBOSE = False


def ok(msg):
    print(f"  ok    {msg}")


class ScriptedModel:
    """Stands in for Claude: returns the given responses in order and records
    every request the agent made (a deep copy, so later changes don't show)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, system, messages, tools):
        self.requests.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        assert self.responses, "the agent called the model more often than expected"
        return self.responses.pop(0)


def tool_use(id, tool_name, **tool_input):
    return {"type": "tool_use", "id": id, "name": tool_name, "input": tool_input}


def text(t):
    return {"type": "text", "text": t}


def never_called(name, tool_input):
    raise AssertionError(f"approve() should only be asked about risky tools, but was asked about {name}")


def check_step_1():
    assert agent.calculator("1200 / 12") == "100", "calculator is provided and should already work"
    pages = agent.list_handbook()
    assert isinstance(pages, str), "list_handbook should return a string (tools return text)"
    assert "vacation.md" in pages and "office.md" in pages, f"list_handbook should list the page names, got {pages!r}"
    assert len(pages.splitlines()) == 9, "list_handbook should give one page per line (9 pages)"
    ok("list_handbook")

    assert "27 paid vacation days" in agent.read_handbook("vacation.md"), "read_handbook should return the page text"
    assert agent.read_handbook("vacation") == agent.read_handbook("vacation.md"), "read_handbook('vacation') should work too"
    try:
        agent.read_handbook("parental-leave.md")
        raise AssertionError("read_handbook should raise ValueError for a page that doesn't exist")
    except ValueError as e:
        assert "vacation.md" in str(e), "the error should list the available pages, so Claude can recover"
    ok("read_handbook (including a helpful error)")

    record = agent.get_employee("priya")
    assert isinstance(record, str), "get_employee should return a string, e.g. json.dumps(person)"
    assert json.loads(record)["vacation_days_left"] == 12, f"get_employee('priya') returned {record!r}"
    assert json.loads(agent.get_employee("Tomás Silva"))["team"] == "Design", "full names should work too"
    try:
        agent.get_employee("Gandalf")
        raise AssertionError("get_employee should raise ValueError for an unknown name")
    except ValueError as e:
        assert "Priya Raman" in str(e), "the error should list the known employees"
    ok("get_employee")


def check_tool_schema(tool):
    name = tool.get("name")
    assert name in agent.TOOL_FUNCTIONS or name == "book_vacation", f"tool {name!r} has no function"
    assert len(tool.get("description", "")) >= 30, f"{name}: write a real description (when should Claude use it?)"
    schema = tool.get("input_schema", {})
    assert schema.get("type") == "object", f"{name}: input_schema needs \"type\": \"object\""
    props = schema.get("properties")
    assert isinstance(props, dict), f"{name}: input_schema needs \"properties\" (use {{}} for no inputs)"
    for prop, spec in props.items():
        assert "type" in spec, f"{name}.{prop}: give every property a \"type\""
    assert set(schema.get("required", [])) <= set(props), f"{name}: \"required\" lists unknown properties"
    return name, set(props)


def check_step_2():
    names = [t.get("name") for t in agent.TOOLS]
    for expected in ["calculator", "list_handbook", "read_handbook", "get_employee"]:
        assert expected in names, f"TOOLS is missing {expected!r}"
    for tool in agent.TOOLS:
        name, props = check_tool_schema(tool)
        fn = agent.TOOL_FUNCTIONS.get(name)
        if fn is not None:
            params = set(inspect.signature(fn).parameters)
            assert props == params, f"{name}: schema properties {sorted(props)} don't match the function's parameters {sorted(params)}"
    ok(f"{len(agent.TOOLS)} tool schemas look valid and match their functions")


def check_step_3():
    assert agent.run_tool("calculator", {"expression": "27 - 5"}, never_called) == ("22", False), \
        "run_tool should return (result_text, False) on success"
    ok("run_tool calls the right function")
    out, is_error = agent.run_tool("read_handbook", {"page": "nope.md"}, never_called)
    assert is_error is True and out.startswith("Error"), f"a tool that raises should give ('Error: ...', True), got {(out, is_error)}"
    out, is_error = agent.run_tool("calculator", {"expression": "1 / 0"}, never_called)
    assert is_error is True, "every exception should be caught, not just ValueError"
    out, is_error = agent.run_tool("launch_rocket", {}, never_called)
    assert is_error is True and "launch_rocket" in out, "an unknown tool name should give an error that names it"
    out, is_error = agent.run_tool("calculator", {"wrong_arg": "1"}, never_called)
    assert is_error is True, "wrong arguments should give an error, not crash"
    ok("run_tool turns every failure into an error message for Claude")

    content = [
        text("Let me check two things."),
        tool_use("toolu_a", "get_employee", name="Mei"),
        tool_use("toolu_b", "calculator", expression="2 * 21"),
    ]
    msg = agent.handle_tool_calls(content, never_called)
    assert msg.get("role") == "user", "tool results go back in a message with role 'user'"
    results = msg.get("content")
    assert isinstance(results, list) and len(results) == 2, "one tool_result per tool_use block, all in ONE message"
    assert [r.get("type") for r in results] == ["tool_result"] * 2, "each block needs \"type\": \"tool_result\""
    assert [r.get("tool_use_id") for r in results] == ["toolu_a", "toolu_b"], "tool_use_id must match the tool_use id, in order"
    assert "Mei Tanaka" in results[0]["content"] and results[1]["content"] == "42", "content should be the tool's output"
    assert results[0].get("is_error") is False, "include is_error on every result"
    ok("handle_tool_calls builds one user message with matching tool_results")


def check_step_4():
    model = ScriptedModel([
        {"stop_reason": "tool_use", "content": [text("Looking it up."), tool_use("toolu_1", "get_employee", name="Priya")]},
        {"stop_reason": "end_turn", "content": [text("Priya has 12 days left.")]},
    ])
    messages = [{"role": "user", "content": "How many days does Priya have?"}]
    reply = agent.run_agent(messages, model, never_called)
    assert reply == "Priya has 12 days left.", f"run_agent should return the final text, got {reply!r}"
    assert len(model.requests) == 2, f"expected 2 model calls, got {len(model.requests)}"
    second = model.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"], \
        "2nd request should be: user question, assistant tool call, user tool_result"
    assert second[1]["content"][1]["id"] == "toolu_1", "append the assistant's content unchanged (all blocks)"
    assert second[2]["content"][0]["tool_use_id"] == "toolu_1", "the tool_result must answer toolu_1"
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"], \
        "the final assistant message should be appended to messages too"
    assert model.requests[0]["tools"] is agent.TOOLS and model.requests[0]["system"] == agent.SYSTEM_PROMPT, \
        "call model(SYSTEM_PROMPT, messages, TOOLS)"
    ok("loop: model -> tool -> model -> answer")

    model = ScriptedModel([{"stop_reason": "end_turn", "content": [text("Hello"), text("there")]}])
    assert agent.run_agent([{"role": "user", "content": "hi"}], model, never_called) == "Hello\nthere", \
        "join several text blocks with newlines"
    ok("no tools needed: answers right away")

    loop = {"stop_reason": "tool_use", "content": [tool_use("toolu_x", "list_handbook")]}
    model = ScriptedModel([loop] * 3)
    reply = agent.run_agent([{"role": "user", "content": "hi"}], model, never_called, max_turns=3)
    assert len(model.requests) == 3, "stop after max_turns model calls"
    assert isinstance(reply, str) and reply, "return a message explaining the agent stopped"
    ok("max_turns stops a runaway agent")

    assert agent.run_agent([{"role": "user", "content": "hi"}], lambda *a: None, never_called) is None, \
        "if the model returns None (no API key), return None"
    ok("returns None without an API key")


def check_step_5():
    agent.reset_state()
    assert "book_vacation" in agent.TOOL_FUNCTIONS, "register book_vacation in TOOL_FUNCTIONS"
    tool = next((t for t in agent.TOOLS if t.get("name") == "book_vacation"), None)
    assert tool is not None, "add a book_vacation schema to TOOLS"
    _, props = check_tool_schema(tool)
    assert props == set(inspect.signature(agent.book_vacation).parameters), "schema doesn't match book_vacation's parameters"
    ok("book_vacation is registered")

    out = agent.book_vacation("Mei", "2026-11-02", 5)
    assert agent.find_employee("Mei")["vacation_days_left"] == 15, "booking 5 days should leave Mei with 15"
    assert agent.BOOKINGS == [{"name": "Mei Tanaka", "start_date": "2026-11-02", "days": 5}], f"BOOKINGS is {agent.BOOKINGS}"
    assert "15" in out, "tell Claude the new balance in the result"
    for bad in [("Tomás", "2026-11-02", 5), ("Mei", "next monday", 2), ("Mei", "2026-11-02", 0), ("Nobody", "2026-11-02", 1)]:
        try:
            agent.book_vacation(*bad)
            raise AssertionError(f"book_vacation{bad} should raise ValueError")
        except ValueError:
            pass
    assert agent.find_employee("Tomás")["vacation_days_left"] == 3, "a failed booking must not change the balance"
    ok("book_vacation books valid requests and rejects bad ones")

    agent.reset_state()
    asked = []
    out, is_error = agent.run_tool("book_vacation", {"name": "Mei", "start_date": "2026-11-02", "days": 2},
                                   lambda name, inp: asked.append(name) or False)
    assert asked == ["book_vacation"], "run_tool should ask approve(name, tool_input) before a risky tool"
    assert is_error is True and agent.find_employee("Mei")["vacation_days_left"] == 20, \
        "when the user declines, don't run the tool and return an error saying so"
    out, is_error = agent.run_tool("book_vacation", {"name": "Mei", "start_date": "2026-11-02", "days": 2}, lambda n, i: True)
    assert is_error is False and agent.find_employee("Mei")["vacation_days_left"] == 18, "approved bookings should go through"
    agent.reset_state()
    ok("risky tools need approval")


def check_step_6():
    assert len(agent.SYSTEM_PROMPT) > 100, "write a real SYSTEM_PROMPT"
    assert date.today().isoformat() in agent.SYSTEM_PROMPT, \
        "put today's date in SYSTEM_PROMPT (f-string with date.today()); Claude can't know it otherwise"
    ok("SYSTEM_PROMPT includes today's date")

    model = ScriptedModel([
        {"stop_reason": "end_turn", "content": [text("Hi Priya!")]},
        {"stop_reason": "end_turn", "content": [text("You are Priya.")]},
    ])
    messages = []
    assert agent.chat_turn(messages, "I'm Priya.", model, never_called) == "Hi Priya!"
    assert agent.chat_turn(messages, "Who am I?", model, never_called) == "You are Priya."
    roles_and_text = [(m["role"], m["content"] if isinstance(m["content"], str) else m["content"][0]["text"])
                      for m in model.requests[1]["messages"]]
    assert roles_and_text == [("user", "I'm Priya."), ("assistant", "Hi Priya!"), ("user", "Who am I?")], \
        f"the 2nd turn should include the whole conversation so far, got {roles_and_text}"
    ok("chat_turn keeps the conversation history")


STEPS = {str(i): globals()[f"check_step_{i}"] for i in range(1, 7)}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for step in STEPS if which == "all" else [which]:
        print(f"Step {step}")
        try:
            STEPS[step]()
        except NotImplementedError:
            print("  todo  not implemented yet")
            sys.exit(1)
        except AssertionError as e:
            print(f"  FAIL  {e}")
            sys.exit(1)
    print("\nAll checks passed!")
