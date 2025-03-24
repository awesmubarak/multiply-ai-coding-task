import os
from dataclasses import dataclass, field
from enum import Enum

from google import genai

# Generate your API key at https://aistudio.google.com/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def llm(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text


class Sender(Enum):
    USER = "user"
    AI = "ai"


@dataclass
class Message:
    text: str
    sender: Sender


@dataclass
class ExtractedInformation:
    pass

    def __str__(self) -> str:
        return "Information extracted:..."


@dataclass
class ConversationState:
    finished: bool = False
    messages: list[Message] = field(default_factory=list)
    new_messages: list[Message] = field(default_factory=list)
    extracted_information: ExtractedInformation = field(
        default_factory=ExtractedInformation
    )


def chat_response(state: ConversationState) -> ConversationState:
    return ConversationState(
        finished=True,
        messages=state.messages,
        new_messages=[
            Message(
                text=llm(f"Respond to this question: {state.messages[-1].text}"),
                sender=Sender.AI,
            )
        ],
        extracted_information=state.extracted_information,
    )
