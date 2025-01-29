import os

import google.generativeai as genai
from langchain import hub
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


class GeminiWrapper:
    def __init__(self, model_name, hub_prompt, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.base_prompt: ChatPromptTemplate = hub.pull(hub_prompt)

    def predict(self, doc_input):
        messages = self.base_prompt.invoke(doc_input)
        messages = [convert_langchain_to_gemini_chat(message) for message in messages.messages]
        chat = self.model.start_chat(
            history=messages[:-1]
        )
        response = chat.send_message(messages[-1])
        return response.text


def create_default_gemini_1_5_pro(hub_prompt):
    return GeminiWrapper(
        model_name='models/gemini-1.5-pro',
        hub_prompt=hub_prompt)


def create_default_gemini_1_5_flash_8b(hub_prompt):
    return GeminiWrapper(
        model_name='models/gemini-1.5-flash-8b',
        hub_prompt=hub_prompt)


def convert_langchain_to_gemini_chat(message: BaseMessage) -> dict:
    """Convert LangChain message to ChatML format."""

    if isinstance(message, SystemMessage):
        role = "model"
    elif isinstance(message, AIMessage):
        role = "model"
    elif isinstance(message, HumanMessage):
        role = "user"
    else:
        raise ValueError(f"Unknown message type: {type(message)}")

    return {"role": role, "parts": message.content}
