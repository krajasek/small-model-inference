#!/usr/bin/env python3
"""CLI client for the inference server using WebSocket streaming.

Provides an interactive chat interface with low time-to-first-token latency.

Usage:
    uv run python -m inference.cli [OPTIONS]

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
    import websockets
    from websockets.asyncio.client import connect
except ImportError:
    print("Error: websockets library required. Install with: uv add websockets")
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
            )
            return True
        except Exception as e:
            print(f"\nError connecting to server: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_message(self, user_input: str) -> str | None:
        """Send a message and stream the response.

        Returns the complete assistant response or None on error.
        """
        if not self._ws:
            print("\nNot connected to server")
            return None

        # Add user message to history
        self.session.add_user_message(user_input)

        # Build and send request
        request = self.session.build_request()
        await self._ws.send(bytes(request))

        # Stream and collect response
        response_text = ""
        first_token_time: float | None = None
        start_time = time.perf_counter()

        try:
            while True:
                data = await self._ws.recv()
                if isinstance(data, str):
                    continue  # Skip text frames

                # Parse server message
                server_msg = ServerMessage().parse(data)

                # Check for error
                if server_msg.error and server_msg.error.message:
                    print(f"\nServer error: {server_msg.error.message}")
                    # Remove the user message we just added
                    self.session.messages.pop()
                    return None

                # Handle stream chunk
                if server_msg.chunk and server_msg.chunk.choice:
                    chunk = server_msg.chunk
                    delta = chunk.choice.delta

                    # Track time to first token
                    if delta.content and first_token_time is None:
                        first_token_time = time.perf_counter()

                    # Print token
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        response_text += delta.content

                    # Check for completion
                    if chunk.choice.finish_reason:
                        break

                # Handle stream complete
                if server_msg.complete:
                    self.session.total_tokens += server_msg.complete.usage.total_tokens
                    break

        except websockets.exceptions.ConnectionClosed as e:
            print(f"\nConnection closed: {e}")
            self._ws = None
            return None
        except Exception as e:
            print(f"\nError receiving response: {e}")
            return None

        # Print timing stats
        end_time = time.perf_counter()
        total_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else 0

        print()  # Newline after response
        print(f"\n[TTFT: {ttft * 1000:.0f}ms | Total: {total_time:.2f}s]")

        # Add assistant response to history
        if response_text:
            self.session.add_assistant_message(response_text)

        return response_text


def print_welcome(config: ChatConfig) -> None:
    """Print welcome message and instructions."""
    print("=" * 60)
    print("Inference Server CLI (WebSocket)")
    print("=" * 60)
    print(f"Server: ws://{config.host}:{config.port}/v1/stream")
    print(f"Max tokens: {config.max_tokens} | Temperature: {config.temperature}")
    if config.system_prompt:
        print(f"System: {config.system_prompt[:50]}...")
    print("-" * 60)
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /stats  - Show session statistics")
    print("  /quit   - Exit the chat")
    print("=" * 60)
    print()


def print_stats(session: ChatSession) -> None:
    """Print session statistics."""
    print("\n--- Session Stats ---")
    print(f"Messages: {len(session.messages)}")
    print(f"Total tokens: {session.total_tokens}")
    print()


async def main_loop(client: ChatClient) -> None:
    """Main chat loop."""
    print_welcome(client.config)

    # Connect to server
    print("Connecting to server...", end=" ", flush=True)
    if not await client.connect():
        return
    print("Connected!\n")

    # Initialize with system prompt
    client.session.add_system_prompt()

    try:
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd == "/quit" or cmd == "/exit":
                    print("Goodbye!")
                    break
                elif cmd == "/clear":
                    client.session.clear()
                    print("Conversation cleared.\n")
                    continue
                elif cmd == "/stats":
                    print_stats(client.session)
                    continue
                else:
                    print(f"Unknown command: {user_input}")
                    continue

            # Send message and stream response
            print("\nAssistant: ", end="", flush=True)
            await client.send_message(user_input)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    finally:
        await client.disconnect()


def parse_args() -> ChatConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CLI client for the inference server using WebSocket streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
