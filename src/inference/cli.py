#!/usr/bin/env python3
"""CLI client for the inference server using WebSocket streaming.

A beautiful TUI for interactive chat with low time-to-first-token latency.

Usage:
    uv run inference-cli [OPTIONS]

Options:
    --host HOST         Server host (default: localhost)
    --port PORT         Server port (default: 8000)
    --system PROMPT     System prompt for the conversation
    --max-tokens N      Maximum tokens to generate (default: 256)
    --temperature T     Sampling temperature (default: 0.7)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field

try:
    import betterproto
    import websockets
    from websockets.asyncio.client import connect
except ImportError:
    print("Error: websockets and betterproto libraries required.")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
except ImportError:
    print("Error: rich library required. Install with: uv add rich")
    sys.exit(1)

# Import protobuf definitions
try:
    from .proto import (
        ChatCompletionRequest,
        ChatMessage,
        ClientMessage,
        GenerationParams,
        ServerMessage,
    )
except ImportError:
    # Handle case when running as script
    sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
    from inference.proto import (
        ChatCompletionRequest,
        ChatMessage,
        ClientMessage,
        GenerationParams,
        ServerMessage,
    )

# Custom theme for the TUI
THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "user": "bold blue",
        "assistant": "bold magenta",
        "system": "dim italic",
        "stats": "dim",
        "command": "bold yellow",
    }
)

console = Console(theme=THEME)


@dataclass
class ChatConfig:
    """Configuration for the chat client."""

    host: str = "localhost"
    port: int = 8000
    system_prompt: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class ChatSession:
    """Manages conversation state and WebSocket communication."""

    config: ChatConfig
    messages: list[ChatMessage] = field(default_factory=list)
    total_tokens: int = 0
    request_count: int = 0

    @property
    def websocket_url(self) -> str:
        return f"ws://{self.config.host}:{self.config.port}/v1/stream"

    def add_system_prompt(self) -> None:
        """Add system prompt if configured."""
        if self.config.system_prompt and not self.messages:
            self.messages.append(ChatMessage(role="system", content=self.config.system_prompt))

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def build_request(self) -> ClientMessage:
        """Build a ClientMessage for the current conversation."""
        self.request_count += 1
        return ClientMessage(
            request_id=uuid.uuid4().hex[:8],
            chat_completion=ChatCompletionRequest(
                messages=self.messages,
                params=GenerationParams(
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                ),
            ),
        )

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.total_tokens = 0
        self.request_count = 0
        self.add_system_prompt()


class ChatClient:
    """WebSocket-based chat client for the inference server."""

    def __init__(self, config: ChatConfig) -> None:
        self.config = config
        self.session = ChatSession(config=config)
        self._ws: websockets.WebSocketClientProtocol | None = None

    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        try:
            self._ws = await connect(
                self.session.websocket_url,
                open_timeout=10,
                close_timeout=5,
                # Disable ping to avoid timeout during long inference
                ping_interval=None,
            )
            return True
        except Exception as e:
            console.print(f"[error]Connection failed:[/error] {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _ensure_connected(self) -> bool:
        """Ensure WebSocket connection is open, reconnecting if needed."""
        if self._ws is None:
            with console.status("[info]Reconnecting...[/info]", spinner="dots"):
                if await self.connect():
                    console.print("[success]Reconnected[/success]")
                    return True
            return False

        # Check if connection is still open
        try:
            if self._ws.close_code is not None:
                self._ws = None
                with console.status("[info]Reconnecting...[/info]", spinner="dots"):
                    if await self.connect():
                        console.print("[success]Reconnected[/success]")
                        return True
                return False
        except Exception:
            self._ws = None
            return await self._ensure_connected()

        return True

    async def send_message(self, user_input: str) -> tuple[str | None, dict]:
        """Send a message and stream the response.

        Returns tuple of (response_text, timing_stats) or (None, {}) on error.
        """
        stats: dict = {}

        if not await self._ensure_connected():
            console.print("[error]Failed to connect to server[/error]")
            return None, stats

        # Add user message to history
        self.session.add_user_message(user_input)

        # Build and send request
        request = self.session.build_request()
        try:
            await self._ws.send(bytes(request))
        except websockets.exceptions.ConnectionClosed:
            self._ws = None
            if not await self._ensure_connected():
                console.print("[error]Failed to reconnect[/error]")
                self.session.messages.pop()
                return None, stats
            await self._ws.send(bytes(request))

        # Stream and collect response
        response_text = ""
        first_token_time: float | None = None
        start_time = time.perf_counter()
        token_count = 0

        # Show spinner while waiting for first token
        spinner = Spinner("dots", text="Thinking...", style="info")

        try:
            with Live(spinner, console=console, refresh_per_second=10, transient=True):
                while True:
                    data = await self._ws.recv()
                    if isinstance(data, str):
                        continue

                    server_msg = ServerMessage().parse(data)
                    payload_type, _ = betterproto.which_one_of(server_msg, "payload")

                    if payload_type == "error":
                        console.print(f"[error]Server error:[/error] {server_msg.error.message}")
                        self.session.messages.pop()
                        return None, stats

                    elif payload_type == "chunk":
                        chunk = server_msg.chunk
                        delta = chunk.choice.delta

                        if delta.content and first_token_time is None:
                            first_token_time = time.perf_counter()
                            # Exit the Live context to start printing tokens
                            break

                    elif payload_type == "complete":
                        self.session.total_tokens += server_msg.complete.usage.total_tokens
                        break

            # Continue streaming tokens outside the spinner
            if first_token_time is not None:
                # Print the first token we already received
                if delta.content:
                    console.print(delta.content, end="")
                    response_text += delta.content
                    token_count += 1

                # Continue receiving remaining tokens
                while True:
                    data = await self._ws.recv()
                    if isinstance(data, str):
                        continue

                    server_msg = ServerMessage().parse(data)
                    payload_type, _ = betterproto.which_one_of(server_msg, "payload")

                    if payload_type == "chunk":
                        chunk = server_msg.chunk
                        delta = chunk.choice.delta

                        if delta.content:
                            console.print(delta.content, end="")
                            response_text += delta.content
                            token_count += 1

                    elif payload_type == "complete":
                        self.session.total_tokens += server_msg.complete.usage.total_tokens
                        break

        except websockets.exceptions.ConnectionClosed:
            console.print("\n[warning]Connection lost - will reconnect on next message[/warning]")
            self._ws = None
            self.session.messages.pop()
            return None, stats
        except Exception as e:
            console.print(f"\n[error]Error:[/error] {e}")
            return None, stats

        console.print()  # Newline after response

        # Calculate stats
        end_time = time.perf_counter()
        total_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else 0
        tokens_per_sec = token_count / total_time if total_time > 0 else 0

        stats = {
            "ttft_ms": ttft * 1000,
            "total_s": total_time,
            "tokens": token_count,
            "tokens_per_sec": tokens_per_sec,
        }

        # Add assistant response to history
        if response_text:
            self.session.add_assistant_message(response_text)

        return response_text, stats


def print_welcome(config: ChatConfig) -> None:
    """Print welcome banner."""
    title = Text()
    title.append("Inference CLI", style="bold magenta")
    title.append(" ", style="dim")
    title.append("WebSocket Streaming", style="dim cyan")

    # Connection info
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="dim")
    info_table.add_column()
    info_table.add_row("Server", f"ws://{config.host}:{config.port}/v1/stream")
    info_table.add_row("Max tokens", str(config.max_tokens))
    info_table.add_row("Temperature", str(config.temperature))
    if config.system_prompt:
        prompt_display = (
            config.system_prompt[:40] + "..."
            if len(config.system_prompt) > 40
            else config.system_prompt
        )
        info_table.add_row("System", f'"{prompt_display}"')

    panel = Panel(
        info_table,
        title=title,
        border_style="blue",
        padding=(0, 1),
    )
    console.print(panel)

    # Commands help
    commands = Text()
    commands.append("/clear", style="command")
    commands.append(" clear history  ", style="dim")
    commands.append("/stats", style="command")
    commands.append(" show stats  ", style="dim")
    commands.append("/help", style="command")
    commands.append(" commands  ", style="dim")
    commands.append("/quit", style="command")
    commands.append(" exit", style="dim")
    console.print(commands)
    console.print()


def print_stats(session: ChatSession) -> None:
    """Print session statistics in a nice table."""
    table = Table(title="Session Statistics", border_style="dim")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    user_msgs = sum(1 for m in session.messages if m.role == "user")
    assistant_msgs = sum(1 for m in session.messages if m.role == "assistant")

    table.add_row("Total messages", str(len(session.messages)))
    table.add_row("User messages", str(user_msgs))
    table.add_row("Assistant messages", str(assistant_msgs))
    table.add_row("Requests made", str(session.request_count))
    table.add_row("Total tokens", str(session.total_tokens))

    console.print(table)
    console.print()


def print_help() -> None:
    """Print help message."""
    table = Table(title="Commands", border_style="dim", show_header=False)
    table.add_column("Command", style="command")
    table.add_column("Description")

    table.add_row("/clear", "Clear conversation history")
    table.add_row("/stats", "Show session statistics")
    table.add_row("/help", "Show this help message")
    table.add_row("/quit, /exit", "Exit the chat")
    table.add_row("Ctrl+C", "Interrupt and exit")
    table.add_row("Ctrl+D", "Exit (EOF)")

    console.print(table)
    console.print()


def print_response_stats(stats: dict) -> None:
    """Print response timing stats."""
    if not stats:
        return

    stat_text = Text()
    stat_text.append("  ", style="dim")
    stat_text.append(f"TTFT: {stats['ttft_ms']:.0f}ms", style="stats")
    stat_text.append("  ", style="dim")
    stat_text.append(f"Total: {stats['total_s']:.2f}s", style="stats")
    stat_text.append("  ", style="dim")
    stat_text.append(f"{stats['tokens_per_sec']:.1f} tok/s", style="stats")
    console.print(stat_text)


def get_user_input() -> str | None:
    """Get user input with a styled prompt."""
    try:
        console.print()
        console.print("[user]You[/user] ", end="")
        return input().strip()
    except EOFError:
        return None
    except KeyboardInterrupt:
        return None


async def main_loop(client: ChatClient) -> None:
    """Main chat loop."""
    print_welcome(client.config)

    # Connect to server
    with console.status("[info]Connecting to server...[/info]", spinner="dots"):
        connected = await client.connect()

    if not connected:
        return

    console.print("[success]Connected![/success]")
    console.print()

    # Initialize with system prompt
    client.session.add_system_prompt()

    try:
        while True:
            user_input = get_user_input()

            if user_input is None:
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ("/quit", "/exit", "/q"):
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif cmd == "/clear":
                    client.session.clear()
                    console.print("[success]Conversation cleared.[/success]")
                    continue
                elif cmd == "/stats":
                    print_stats(client.session)
                    continue
                elif cmd == "/help":
                    print_help()
                    continue
                else:
                    console.print(f"[warning]Unknown command:[/warning] {user_input}")
                    console.print("[dim]Type /help for available commands[/dim]")
                    continue

            # Show assistant label
            console.print()
            console.print("[assistant]Assistant[/assistant]")

            # Send message and stream response
            response, stats = await client.send_message(user_input)

            if response:
                # Render as markdown for nice formatting
                console.print()
                md = Markdown(response)
                console.print(Panel(md, border_style="dim magenta", padding=(0, 1)))
                print_response_stats(stats)

    except KeyboardInterrupt:
        console.print("\n\n[dim]Interrupted. Goodbye![/dim]")
    finally:
        await client.disconnect()


def parse_args() -> ChatConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat CLI for the inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  inference-cli
  inference-cli --host 192.168.1.100 --port 8080
  inference-cli -s "You are a helpful coding assistant"
  inference-cli --max-tokens 512 --temperature 0.8
        """,
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--system",
        "-s",
        dest="system_prompt",
        help="System prompt for the conversation",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )

    args = parser.parse_args()

    return ChatConfig(
        host=args.host,
        port=args.port,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


def main() -> None:
    """Entry point for the CLI client."""
    config = parse_args()
    client = ChatClient(config)
    asyncio.run(main_loop(client))


if __name__ == "__main__":
    main()
