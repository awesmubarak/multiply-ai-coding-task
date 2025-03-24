# Multiply AI Coding Task

## Problem overview

On of the most important tasks in financial advice is accurately capturing information from customers. A common task for an adviser is to collect information on a customer's goals and future plans.

## Task

Your task is to use the Google Gemini API to create a conversational interface by which user information can be collected. We want you to collect information about a user's goals using an LLM.

In `multiply_ai_coding_task/factfind.py` you will find some basic model definitions that show the scope of the information we would like to collect from a user. Your conversational agent should look to collect as much of this information from a user.

We have provided you with a basic set up, which uses [Poetry](https://python-poetry.org/) to manage dependencies. We have included the python Google GenAI library, but you will need to provide an API key, which you can generate for free at https://aistudio.google.com/apikey.

Within `multiply_ai_coding_task/chat.py` you will find the key function we want you to implement:

```python

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
```

This chat reponse function should take the current state of a conversation between a user and an AI agent and return an updated conversation, with new messages from the AI and any updated information extracted from the conversation about the user.

You can use `poetry run demo` to run a demo conversation using this function.

The nature of LLMs means that you are unlikely to be able to produce an agent that can handle all edge cases. We want to see an agent that can satisfy the basics of the above, with thought given to how such a system might be productionised. Any points that you do not get time to code can be discussed at subsequent interviews.

## Constraints

- For the purposes of this task, please only use the Google Gemini "gemini-2.0-flash" model. Feel free to update the model call itself, for example to make use of structured responses from the API.
- Feel free to use any additional libraries you may need, although be mindful of any complexity you introduce
- You are free to add fields to existing dataclasses as you require
- We will test your code by running the demo function, please make sure that this continues to work.
- You are free to use AI assisted coding tools, but be aware that we will discuss your task with you to ensure that you understand the code that is written.

Good luck!
