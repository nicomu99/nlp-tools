# An AI agent from scratch

```
            ┌──────────────────────────────────────────────────────┐
            │                                                      │
 task ──► messages ──► Claude ──► wants a tool? ──yes──► run it ──► append tool_result
                                       │
                                       no
                                       ▼
                                 final answer
```

## Setup

```bash
conda create -n env python=3.11 -y
conda activate myenv
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Files

| File                  | What it is                                                               |
|-----------------------|--------------------------------------------------------------------------|
| `agent.py`            | File containing the AI agent.                                            |
| `check.py`            | Tests for each step: `python check.py 1`, `python check.py 2`, …         |
| `data/handbook`       | The company handbook, one Markdown page per topic.                       |
| `data/employees.json` | Toy employee data with vacation balances and on-call dates. Have a look. |
| `llm.py`              | LLM call helper.                                                         |

## How tool use works (read this first)

Every request to Claude sends the **whole conversation** plus a list of **tool descriptions**. Claude never runs anything itself. When it wants a tool, it stops and replies with a request instead of an answer:

```
{"stop_reason": "tool_use",
 "content": [{"type": "text", "text": "Let me look that up."},
             {"type": "tool_use", "id": "toolu_01A", "name": "get_employee", "input": {"name": "Priya"}}]}
```

Your program runs `get_employee(name="Priya")` and sends the result back as the next **user** message:

```
{"role": "user",
 "content": [{"type": "tool_result", "tool_use_id": "toolu_01A", "content": "{\"vacation_days_left\": 12, ...}"}]}
```

Then you call Claude again with the longer conversation. When `stop_reason` is `"end_turn"`, Claude has its answer. That's the entire protocol.

(With a real API key you may also see `"thinking"` blocks in `content`. Leave them alone: the assistant's content always goes back exactly as it came.)

---

## AI Agent Overview

The AI agent essentially is a combination of an LLM and tools it is allowed to use. To make this combination work, the LLM has must understand what each tool can do and how it can use it. It must be furthermore be able to understand the output of calling a tool and try to resolve errors if they arise. The system is therefore composed of the following steps:

### Step 1: Tools are just functions

A tool is an ordinary Python function that takes simple inputs and returns a **string**, because a string is all an LLM can read.

If the execution of a tool leads to an error, the system should not stop, but try to recover from the error. Each tool should therefore contain sensible input parameter checks and provide a meaningful error message.

The tools this AI agent is provided are:
- `list_handbook()`: Returns handbook pages, so the agent can decide which page to read.
- `read_handbook()`: Prints the contents of one handbook page. If the page can not be found, an error message notifies the agent of this occurrence. The message also contains a list of all handbook page names.
- `get_employee()`: Retrieves an employee's information. 


### Step 2: Tool descriptions

The LLM itself does not see the actual function implementations, but only a name, a description and a [JSON Schema](https://json-schema.org/understanding-json-schema/) for the inputs. The description is a prompt and tells the model what to use each tool for, the name is used as an identifier and the input schema provides information about what parameters the tool takes in and what their type is.


### Step 3: Handle tool calls

Each of the LLMs messages must be scanned for tool use requests. For Claude, the LLM will answer with message type `tool_use` if it requests using a tool.

The LLM's output should be re-evaluated, for example by checking whether the tool's name actually exists. If the tool provides access to a critical part of the system, the user should be first asked for permission.


### Step 4: The agent loop

The agent loop, also called harness, is the heart of an AI agent. 

It essentially is a simple loop that processes the tool requests of an LLM one-by-one. After the tools have been called, the output is re-fed into the LLM, and it can continue searching for an answer. 

Claude notifies the system of having finished execution by providing a `stop_reason` other than `"tool_use"`. The harness is also limited by a maximum number of iterations, so the system does not use too many tokens.

The system can be tested by running

```bash
python agent.py ask  "If Priya takes 5 days off, how many does she have left?"
```

### Digression: How to handle critical tools

If a tool is allowed to make permanent changes, it is best to add a security mechanism, e.g. by making the user agree to any changes the system will make.

The system checks whether the request tools is one such 'Risky' tools and asks the user for approval before continuing.

## Step 6: System prompt and conversation memory

Two last things every agent needs: a system prompt that gives it context it can't look up (who it is, what it's for, **what day it is**), and memory across turns.

To use the system, run: 

```bash
python agent.py chat
you> I'm Priya. How many vacation days do I have?
you> And if I take the week of November 9th off, is there a problem?
```
