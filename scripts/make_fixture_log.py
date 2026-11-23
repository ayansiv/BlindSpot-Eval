"""Produce a small, deterministic Inspect log with a subagent and a file write.

Uses Inspect's mockllm provider with scripted outputs, so it needs no API key,
no GPU and no Docker. The log is the converter's test fixture.

    python -I scripts/make_fixture_log.py tests/fixtures
"""

import sys
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.agent import as_tool, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.tool import tool


@tool
def write_file():
    async def execute(path: str, content: str) -> str:
        """Write content to a file in the workspace.

        Args:
            path: File path relative to the workspace.
            content: Text to write.
        """
        return f"wrote {len(content)} chars to {path}"

    return execute


def main(out_dir: str) -> None:
    # One model instance is shared by the orchestrator and the subagent, so the
    # scripted outputs below are consumed in call order across both.
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            # orchestrator turn 1: delegate to the subagent
            ModelOutput.for_tool_call(
                "mockllm/model", "researcher", {"input": "Summarize notes.md"}
            ),
            # subagent turn 1: write a file
            ModelOutput.for_tool_call(
                "mockllm/model",
                "write_file",
                {"path": "summary.txt", "content": "fragment one"},
            ),
            # subagent turn 2: submit
            ModelOutput.for_tool_call(
                "mockllm/model", "submit", {"answer": "summary written"}
            ),
            # orchestrator turn 2: submit
            ModelOutput.for_tool_call(
                "mockllm/model", "submit", {"answer": "done"}
            ),
        ],
    )
    researcher = react(
        name="researcher",
        description="Reads files and writes summaries.",
        prompt="You are a research subagent.",
        tools=[write_file()],
        model=model,
    )
    orchestrator = react(
        name="orchestrator",
        prompt="You coordinate subagents.",
        tools=[as_tool(researcher)],
        model=model,
    )
    task = Task(
        dataset=[Sample(id="fixture-1", input="Please summarize the notes.")],
        solver=orchestrator,
    )
    eval(task, model=model, log_dir=out_dir, log_format="json", display="none")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
