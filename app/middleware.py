"""Agent middlewares: the safety guard and the call limits. Use: create_agent(..., middleware=build_middleware()).

1. shop_guard: runs before the model. Blocks prompt injection and off-topic requests with a fixed
   refusal, so the LLM is never called for them (safe and free).
2. Call limits: cap model and tool calls so a loop can't burn the Haiku spend limit, and allow at
   most one place_order per customer message (no duplicate orders).
"""
import re
import unicodedata
from itertools import takewhile

from langchain.agents.middleware import (AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware,
                                         before_model)
from langchain.messages import AIMessage, HumanMessage, ToolMessage

# ── 1. injection + off-topic guard ──

INJECTION_PATTERNS = [
    # "ignore your previous instructions", "forget the rules", "apni instructions bhool jao"
    # (no punctuation in between, so "forget it, what's your return policy?" is allowed)
    r"\b(ignore|disregard|forget|override|bypass|bhool|bhul)\w*\b[^.,!?\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?|polic(y|ies)|restrictions?|hidayat)\b",
    r"\b(instructions?|rules?|prompts?|hidayat)\b[^.,!?\n]{0,20}\b(ignore|bhool|bhul|chhor|chor)\w*",   # "rules ignore karo"
    r"\b(system|developer|hidden|initial|original)\s+(prompt|message|instructions?)\b",
    r"\b(reveal|print|repeat|output|leak|dump)\b.{0,30}\b(prompt|instructions)\b",
    r"\byou are (now|no longer)\b",
    r"\b(pretend (to be|you)|roleplay|from now on,? you)\b",
    r"\bnew (instructions|rules|persona)\b",
    r"\b(jailbreak|dan mode|developer mode|god mode|admin mode)\b",
    r"(^|\n)\s*(system|assistant|developer)\s*:",                  # fake chat roles
    r"<\s*/?\s*(system|instructions?|admin)\s*>|\[\s*/?\s*(system|inst)\s*\]",
    # reaching for other people's data or the backend
    r"\b(other|another|all|every)\s+(customers?|users?|buyers?|people)('s|s')?\s+(orders?|address(es)?|phone|numbers?|data|details|info)",
    r"\b(orders?|address(es)?|phone|numbers?|data|details|info)\s+(of|for)\s+(other|another|all|every)\s+(customers?|users?|buyers?|people)\b",
    r"\b(api[ _-]?keys?|drop\s+table|sql)\b|select\s+\*",
    # rewriting prices in the system: "change the price to 0"
    r"\b(set|change|make|update)\b.{0,20}\bprices?\b.{0,20}\b(to|=)\s*(rs\.?\s*)?\d",
]

OFF_TOPIC_PATTERNS = [
    r"\b(write|draft|compose)\b.{0,25}\b(essay|assignment|homework|poem|story|article|cv|resume|cover letter|thesis|report)\b",
    r"\b(solve|do|complete)\b.{0,15}\b(homework|assignment|equation|maths?|quiz)\b",
    r"\b(write|generate)\b.{0,15}\b(code|program|script|function)\b",   # not "give me the code" (= product code)
    r"\b(python|javascript|java|c\+\+|html|css)\s+(code|program|script)\b",
    r"\b(capital of|prime minister|president of|who won|cricket score|match score|tell me a joke|recipe for|meaning of life)\b",
]

INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
OFF_TOPIC_RE = [re.compile(p, re.I) for p in OFF_TOPIC_PATTERNS]

INJECTION_REPLY = ("Sorry, I can't do that. I'm this store's shopping assistant, so I can only help with our "
                   "products, prices, delivery and your own orders. What are you looking for today?")
OFF_TOPIC_REPLY = ("I can only help with shopping at this store: products, sizes, prices, delivery and your "
                   "orders. Is there something you'd like to find?")

ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")


def normalize(text: str) -> str:
    """Undo simple tricks: fancy Unicode letters, zero-width characters, extra spaces."""
    text = ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).strip()


def check_message(text: str) -> tuple[str, str] | None:
    """('injection' | 'off_topic', reply) if the message must be refused, else None.
    Also usable as a standalone Guard node in the LangGraph graph."""
    text = normalize(text)
    if any(r.search(text) for r in INJECTION_RE):
        return "injection", INJECTION_REPLY
    if any(r.search(text) for r in OFF_TOPIC_RE):
        return "off_topic", OFF_TOPIC_REPLY
    return None


@before_model(can_jump_to=["end"])
def shop_guard(state, runtime):
    last = state["messages"][-1]
    if not isinstance(last, HumanMessage):          # only check new customer messages, not tool results
        return None
    verdict = check_message(last.text)
    if verdict is None:
        return None
    kind, reply = verdict
    # name="guard" marks the refusal, so memory can skip it and the benchmark can count it.
    return {"messages": [AIMessage(reply, name="guard", additional_kwargs={"guard": kind})], "jump_to": "end"}


# ── 2. call limits ──

MODEL_CALLS_PER_MESSAGE = 6      # one customer message may use at most 6 model calls
TOOL_CALLS_PER_MESSAGE = 8
# The per-chat cap (60 model calls per thread) lives in the graph (app/agent.py): LangChain keeps its
# thread counter private to each agent, so it can't add up across the Sales/Order/Support agents.
LIMIT_REPLY = ("Sorry, I couldn't finish that request. Could you say it again in a simpler way, "
               "or ask about one product at a time?")


class ShopModelCallLimit(ModelCallLimitMiddleware):
    """Same as LangChain's limit, but the customer sees a friendly message instead of the technical one."""

    def before_model(self, state, runtime):
        result = super().before_model(state, runtime)
        if result and result.get("messages"):
            result["messages"] = [AIMessage(LIMIT_REPLY, name="call_limit")]
        return result


ALREADY_ORDERED = "An order was already handled for this customer message. Do not place another one."


class OneOrderPerMessage(AgentMiddleware):
    """At most one place_order per customer message, even across the confirm/resume step.
    (LangChain's ToolCallLimitMiddleware counts per run, and resuming after 'confirm' starts a new run.)"""

    def _blocked(self, request) -> ToolMessage | None:
        if request.tool_call["name"] != "place_order":
            return None
        since_customer = takewhile(lambda m: not isinstance(m, HumanMessage), reversed(request.state["messages"]))
        if any(isinstance(m, ToolMessage) and m.name == "place_order" for m in since_customer):
            return ToolMessage(ALREADY_ORDERED, tool_call_id=request.tool_call["id"], name="place_order",
                               status="error")
        return None

    def wrap_tool_call(self, request, handler):
        return self._blocked(request) or handler(request)

    async def awrap_tool_call(self, request, handler):
        return self._blocked(request) or await handler(request)


def build_middleware() -> list:
    return [
        shop_guard,
        ShopModelCallLimit(run_limit=MODEL_CALLS_PER_MESSAGE, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=TOOL_CALLS_PER_MESSAGE, exit_behavior="continue"),
        OneOrderPerMessage(),
    ]
