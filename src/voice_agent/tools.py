"""Mock async tools exposed to the model, plus a tiny registry."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

ToolFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

ESCALATE_TOOL = "escalate_to_human"

MOCK_ORDERS: dict[str, dict[str, str]] = {
    "1042": {"status": "shipped", "carrier": "ParcelPost", "eta": "Thursday"},
    "2077": {"status": "processing", "carrier": "", "eta": "ships within 2 days"},
    "3141": {"status": "delivered", "carrier": "ParcelPost", "eta": "delivered Monday"},
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn | None  # None => handled by the orchestrator itself (e.g. handoff)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def call(self, name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool; never raises - errors are returned to the model as data."""
        tool = self._tools.get(name)
        if tool is None or tool.fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        except json.JSONDecodeError:
            return {"error": "arguments were not valid JSON"}
        try:
            return await tool.fn(args)
        except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
            return {"error": f"{type(exc).__name__}: {exc}"}


def make_order_status(latency_ms: int) -> ToolFn:
    async def order_status(args: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(latency_ms / 1000)
        order_id = str(args.get("order_id", "")).strip().lstrip("#")
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return {"order_id": order_id, "found": False}
        return {"order_id": order_id, "found": True, **order}

    return order_status


def make_book_callback(latency_ms: int) -> ToolFn:
    async def book_callback(args: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(latency_ms / 1000)
        slot = str(args.get("preferred_time") or "the next available slot")
        return {
            "booked": True,
            "slot": slot,
            "reference": "CB-" + uuid.uuid4().hex[:6].upper(),
        }

    return book_callback


def default_registry(order_latency_ms: int = 1200, callback_latency_ms: int = 250) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="order_status",
                description="Look up the shipping status of an order by its order number.",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Order number, e.g. 1042"}
                    },
                    "required": ["order_id"],
                },
                fn=make_order_status(order_latency_ms),
            ),
            Tool(
                name="book_callback",
                description="Book a callback from the support team at the caller's preferred time.",
                parameters={
                    "type": "object",
                    "properties": {
                        "preferred_time": {
                            "type": "string",
                            "description": "Natural-language time, e.g. 'tomorrow 10am'",
                        },
                        "topic": {"type": "string", "description": "What the call is about"},
                    },
                    "required": ["preferred_time"],
                },
                fn=make_book_callback(callback_latency_ms),
            ),
            Tool(
                name=ESCALATE_TOOL,
                description=(
                    "Transfer the caller to a human agent. Use when the caller asks for a person, "
                    "is frustrated, or the request is out of scope."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why the caller needs a human"}
                    },
                    "required": ["reason"],
                },
                fn=None,
            ),
        ]
    )
