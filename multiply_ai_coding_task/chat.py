import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from google import genai
from multiply_ai_coding_task.factfind import (
    GoalType,
    Goal,
    NewCarInformation,
    NewHomeGoalInformation,
    OtherGoalInformation,
    User,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def llm(prompt: str) -> str:
    """
    Send a prompt to Gemini and return the raw text response.

    NOTE:
      • Retries / back-off and structured output should be added before prod.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return resp.text


# ---------------------------------------------------------------------------
# Conversation primitives
# ---------------------------------------------------------------------------


class Sender(Enum):
    USER = "user"
    AI = "ai"


@dataclass
class Message:
    text: str
    sender: Sender


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


@dataclass
class ExtractedInformation:
    """
    Ongoing snapshot of facts we have collected so far.
    """
    user: User = field(
        default_factory=lambda: User(
            first_name=None,
            last_name=None,
            email=None,
            date_of_birth=None,
            goals=[],
        )
    )
    # Maps GoalType → index in user.goals
    active_goal_indices: dict[GoalType, int] = field(default_factory=dict)

    def __str__(self) -> str:  # Handy for debugging / logging
        return json.dumps(
            {
                "user": self.user.__dict__,
                "goals": [g.__dict__ for g in self.user.goals],
            },
            indent=2,
            default=str,
        )


@dataclass
class ConversationState:
    finished: bool = False
    messages: list[Message] = field(default_factory=list)
    new_messages: list[Message] = field(default_factory=list)
    extracted_information: ExtractedInformation = field(
        default_factory=ExtractedInformation
    )


# ---------------------------------------------------------------------------
# Prompt template & helpers
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS = """
You are an assistant helping a financial adviser collect information.

Always reply with a *JSON object* with THREE keys:
  "assistant": natural-language reply
  "extracted": {
    "user": {first_name, last_name, email, date_of_birth},
    "goal_type": "new_home|new_car|other",
    "goal_name": string,
    "goal_fields": {field: value}
  }
  "finished": boolean

Valid goal types:
- new_home: requires location, house_price, deposit_amount, purchase_date
- new_car:  requires car_type, car_price, purchase_date
- other:    requires description, amount_required, target_date

Dates must be ISO (YYYY-MM-DD). Numbers should be pure numerics.
""".strip()


def _skeleton(info: ExtractedInformation) -> str:
    """
    Return the current `ExtractedInformation` as a JSON skeleton
    (with None → null) to show the LLM what we have so far.
    """
    def _nullify(v: Any):
        if isinstance(v, Enum):
            return v.value
        if isinstance(
            v,
            (
                User,
                Goal,
                NewHomeGoalInformation,
                NewCarInformation,
                OtherGoalInformation,
            ),
        ):
            return {k: _nullify(getattr(v, k)) for k in v.__dataclass_fields__}
        if isinstance(v, list):
            return [_nullify(i) for i in v]
        if isinstance(v, dict):
            return {k: _nullify(val) for k, val in v.items()}
        return v if v is not None else None

    user_info = {k: _nullify(v) for k, v in info.user.__dict__.items()}
    goals = [_nullify(g) for g in info.user.goals]

    return json.dumps({"user": user_info, "goals": goals}, indent=2, default=str)


def _cleanup_json(txt: str) -> dict[str, Any] | None:
    """
    Strip out Markdown fences (if any) and parse JSON safely.
    Returns None on decode errors.
    """
    try:
        txt = "\n".join(txt.split("\n")[1:-1])
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


_REQUIRED_FIELDS = {
    GoalType.NEW_HOME: ["location", "house_price", "deposit_amount", "purchase_date"],
    GoalType.NEW_CAR: ["car_type", "car_price", "purchase_date"],
    GoalType.OTHER: ["description", "amount_required", "target_date"],
}

_DATE_KEYS = {
    "date_of_birth",
    "purchase_date",
    "car_purchase_date",
    "other_target_date",
}


def _merge(info: ExtractedInformation, payload: dict[str, Any]) -> None:
    """
    Merge *validated* data coming back from the LLM into the existing
    `ExtractedInformation` instance. Partial updates are supported.
    """
    # --- User fields -------------------------------------------------------
    for field_name in ["first_name", "last_name", "email", "date_of_birth"]:
        if field_name in payload and payload[field_name] is not None:
            setattr(info.user, field_name, payload[field_name])

    # --- Goal fields -------------------------------------------------------
    if "goal_type" not in payload or not payload["goal_type"]:
        return

    try:
        goal_type = GoalType(payload["goal_type"].lower())
    except ValueError:
        return  # ignore invalid types

    goal_name = payload.get("goal_name", "Unnamed Goal")

    # Get or create current goal object for this type
    idx = info.active_goal_indices.get(goal_type)
    if idx is None:
        new_goal = Goal(
            goal_type=goal_type,
            goal_name=goal_name,
            goal_specific_information=None,
        )
        info.user.goals.append(new_goal)
        idx = len(info.user.goals) - 1
        info.active_goal_indices[goal_type] = idx

    current_goal = info.user.goals[idx]

    # Extract goal-specific fields
    cleaned: dict[str, Any] = {}
    for k, v in payload.items():
        if k not in _REQUIRED_FIELDS[goal_type]:
            continue

        # Numbers: strip commas / currency symbols
        if k in {"house_price", "deposit_amount", "car_price", "amount_required"}:
            try:
                v = float(str(v).replace(",", "").replace("£", ""))
            except (ValueError, TypeError):
                continue

        # Dates: store as `datetime.date`
        elif k.endswith("_date"):
            try:
                v = datetime.strptime(str(v), "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

        cleaned[k] = v

    # Update goal-specific dataclass
    try:
        if goal_type == GoalType.NEW_HOME:
            base = current_goal.goal_specific_information or NewHomeGoalInformation(
                location=None, house_price=None, deposit_amount=None, purchase_date=None
            )
        elif goal_type == GoalType.NEW_CAR:
            base = current_goal.goal_specific_information or NewCarInformation(
                car_type=None, car_price=None, purchase_date=None
            )
        else:  # OTHER
            base = current_goal.goal_specific_information or OtherGoalInformation(
                description=None, amount_required=None, target_date=None
            )

        current_goal.goal_specific_information = base.__class__(
            **{**base.__dict__, **cleaned}
        )
    except TypeError:
        pass  # ignore malformed partial payloads


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chat_response(state: ConversationState) -> ConversationState:
    """
    Given a `ConversationState`, ask the LLM for the next reply,
    merge any newly-extracted facts, and return the updated state.
    """
    # 1. Prompt construction ------------------------------------------------
    history = "\n".join(f"{m.sender.value.upper()}: {m.text}" for m in state.messages[-8:])
    prompt = (
        _SYSTEM_INSTRUCTIONS
        + "\n"
        + _skeleton(state.extracted_information)
        + "\n---- Conversation so far ----\n"
        + history
        + "\n--------------------------------\n"
        + "Reply in JSON as described."
    )

    # 2. LLM call -----------------------------------------------------------
    raw = llm(prompt)

    # 3. Parse & validate ---------------------------------------------------
    parsed = _cleanup_json(raw)
    if not parsed or "assistant" not in parsed:
        assistant_text = "I'm sorry, could you clarify that?"
        extracted_block = {}
        finished_flag = False
    else:
        assistant_text = str(parsed.get("assistant", ""))
        extracted_block = parsed.get("extracted", {}) or {}
        finished_flag = bool(parsed.get("finished", False))

    # 4. Merge new facts ----------------------------------------------------
    _merge(state.extracted_information, extracted_block)

    # 5. Return new state ---------------------------------------------------
    return ConversationState(
        finished=finished_flag,
        messages=state.messages,
        new_messages=[Message(text=assistant_text, sender=Sender.AI)],
        extracted_information=state.extracted_information,
    )
