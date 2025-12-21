"""Continuous batching for improved throughput."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .inference import GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request in a batch."""

    prompt: str
    config: GenerationConfig
    future: asyncio.Future[tuple[str, dict[str, int]]]
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    """Result for a single request in a batch."""

    text: str
    usage: dict[str, int]


class ContinuousBatcher:
    """Batches multiple requests for efficient inference.

    Collects incoming requests and processes them in batches
    to maximize GPU/CPU utilization.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str,
        max_batch_size: int = 8,
        max_wait_time_ms: int = 50,
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.pad_to_multiple_of = pad_to_multiple_of

        self._pending_requests: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._running = False
        self._batch_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

        # Statistics
        self._total_batches = 0
        self._total_requests = 0
        self._total_batch_size = 0

    async def start(self) -> None:
        """Start the batching loop."""
        if self._running:
            return

        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info(
            f"ContinuousBatcher started: max_batch={self.max_batch_size}, "
            f"max_wait={self.max_wait_time_ms}ms"
        )

    async def stop(self) -> None:
        """Stop the batching loop."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        logger.info("ContinuousBatcher stopped")

    async def submit(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> tuple[str, dict[str, int]]:
        """Submit a request for batched processing."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future[tuple[str, dict[str, int]]] = loop.create_future()

        request = BatchRequest(prompt=prompt, config=config, future=future)
        await self._pending_requests.put(request)

        return await future

    async def _batch_loop(self) -> None:
        """Main batching loop."""
        while self._running:
            try:
                batch = await self._collect_batch()

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in batch loop: {e}")
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> list[BatchRequest]:
        """Collect requests into a batch."""
        batch: list[BatchRequest] = []

        try:
            # Wait for at least one request
            first_request = await asyncio.wait_for(
                self._pending_requests.get(),
                timeout=0.1,
            )
            batch.append(first_request)

            # Collect more requests up to batch size or timeout
            deadline = time.time() + (self.max_wait_time_ms / 1000)

            while len(batch) < self.max_batch_size:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    break

                try:
                    request = await asyncio.wait_for(
                        self._pending_requests.get(),
                        timeout=remaining_time,
                    )
                    batch.append(request)
                except TimeoutError:
                    break

        except TimeoutError:
            pass

        return batch

    async def _process_batch(self, batch: list[BatchRequest]) -> None:
        """Process a batch of requests."""
        if not batch:
            return

        self._total_batches += 1
        self._total_requests += len(batch)
        self._total_batch_size += len(batch)

        logger.debug(f"Processing batch of {len(batch)} requests")

        try:
            # Group requests by similar generation config for efficiency
            # For now, process all together with the config from first request
            # (In production, you'd want to group by compatible configs)

            prompts = [req.prompt for req in batch]
            config = batch[0].config

            results = await self._generate_batch(prompts, config)

            # Distribute results
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result((result.text, result.usage))

        except Exception as e:
            logger.exception(f"Batch processing error: {e}")
            # Fail all requests in the batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    async def _generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[BatchResult]:
        """Generate completions for a batch of prompts."""
        # Run tokenization and generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_batch_sync,
            prompts,
            config,
        )

    def _generate_batch_sync(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[BatchResult]:
        """Synchronous batch generation."""
        # Tokenize all prompts with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        batch_size = inputs["input_ids"].shape[0]
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                top_k=config.top_k if config.do_sample else 0,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode outputs
        results: list[BatchResult] = []
        for i in range(batch_size):
            generated_ids = outputs[i]
            prompt_tokens = prompt_lengths[i]
            completion_tokens = len(generated_ids) - prompt_tokens

            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            results.append(BatchResult(text=text, usage=usage))

        return results

    def stats(self) -> dict[str, Any]:
        """Get batching statistics."""
        avg_batch_size = (
            self._total_batch_size / self._total_batches if self._total_batches > 0 else 0.0
        )
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "average_batch_size": avg_batch_size,
            "pending_requests": self._pending_requests.qsize(),
            "running": self._running,
        }
