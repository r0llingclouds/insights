"""
Insights Agent - Native Anthropic implementation.

A minimal agent loop that uses Claude's tool use to handle complex queries
over ingested sources and conversations.
"""

import json
from typing import Any
from anthropic import Anthropic

from .tools import TOOL_SCHEMAS, TOOL_FUNCTIONS


SYSTEM_PROMPT = """\
You are an assistant for the Insights app, a tool for ingesting documents, \
web pages, and YouTube videos, then running Q&A and chat over them.

You help users with tasks like:
- Ingesting new sources (files, URLs, YouTube videos)
- Finding and listing their sources
- Searching and resuming past conversations
- Asking questions about sources

When handling requests:
1. Think step-by-step about what tools you need
2. Check if sources/conversations exist before trying to use them
3. Be concise in your responses
4. If a task requires resuming a chat, provide the conversation ID so the user can run:
   `uv run insights chat --conversation <id>`

For ambiguous requests, ask clarifying questions rather than guessing.
"""


class InsightsAgent:
    """
    Agent that uses Anthropic's native tool use to handle complex queries.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.client = Anthropic()
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Print debug info if verbose mode is on."""
        if self.verbose:
            print(f"[agent] {message}")

    def _execute_tool(self, name: str, input_args: dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments."""
        if name not in TOOL_FUNCTIONS:
            return {"error": f"Unknown tool: {name}"}

        self._log(f"Executing tool: {name}({json.dumps(input_args)})")

        try:
            result = TOOL_FUNCTIONS[name](**input_args)
            self._log(f"Tool result: {json.dumps(result)[:200]}...")
            return result
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def run(self, user_query: str) -> str:
        """
        Run the agent loop until completion or max steps reached.
        
        Args:
            user_query: Natural language query from the user
            
        Returns:
            Final text response from the agent
        """
        messages = [{"role": "user", "content": user_query}]

        for step in range(self.max_steps):
            self._log(f"Step {step + 1}/{self.max_steps}")

            # Call Claude with tools
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            self._log(f"Stop reason: {response.stop_reason}")

            # If Claude is done (no more tool calls), return the text
            if response.stop_reason == "end_turn":
                # Extract text from response
                text_parts = [
                    block.text
                    for block in response.content
                    if block.type == "text"
                ]
                return "\n".join(text_parts)

            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add assistant's response (with tool_use blocks) to history
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                # Add tool results to history
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason
                self._log(f"Unexpected stop reason: {response.stop_reason}")
                break

        return "I wasn't able to complete this task within the step limit. Please try a simpler query or break it into parts."

    def run_interactive(self) -> None:
        """Run an interactive session with the agent."""
        print("Insights Agent (type 'quit' to exit)")
        print("-" * 40)

        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            response = self.run(query)
            print(f"\n{response}")


def run_agent(
    query: str,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False,
) -> str:
    """
    Convenience function to run a single query through the agent.
    
    Args:
        query: Natural language query
        model: Anthropic model to use
        verbose: Print debug info
        
    Returns:
        Agent's response
    """
    agent = InsightsAgent(model=model, verbose=verbose)
    return agent.run(query)
