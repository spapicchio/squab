import logging
import os
import re

from langchain_core.messages import MessageLikeRepresentation
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from pydantic import BaseModel

from squab.models.prompts import PROMPTS


class LangchainWrapper:
    def __init__(self,
                 model_name: str,
                 hub_prompt: str,
                 api_key: str | None = None,
                 model_kwargs: dict | None = None,
                 is_together: bool = False,
                 ):
        self.model_kwargs = model_kwargs or {'temperature': 0.0}

        self.is_together = is_together
        if self.is_together:
            api_key_name = 'TOGETHER_API_KEY'
            self.llm = ChatTogether(
                model=model_name,
                api_key=api_key or os.getenv(api_key_name),
                **self.model_kwargs
            )
        else:
            api_key_name = 'OPENAI_API_KEY'
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key or os.getenv(api_key_name),
                **self.model_kwargs
            )

        self.base_prompt = ChatPromptTemplate.from_messages(PROMPTS[hub_prompt])
        self.messages = []
        self.parser = None

    @property
    def llm_prompt(self):
        return self.base_prompt + self.messages if self.messages else self.base_prompt

    @property
    def model(self):
        if self.parser:
            return self.llm_prompt | self.llm | self.parser
        return self.llm_prompt | self.llm | StrOutputParser()

    def set_parser(self, parser: PydanticOutputParser | JsonOutputParser):
        self.parser = parser()

    def append_llm_prompt(self, messages: list[MessageLikeRepresentation]):
        self.messages = self.messages + messages

    def reset_messages(self):
        self.messages = []

    def predict(self,
                doc_input: dict,
                append_messages: list[MessageLikeRepresentation] | None = None
                ) -> str | BaseModel | dict:
        if append_messages:
            self.append_llm_prompt(messages=append_messages)
        output = self.model.invoke(doc_input)
        self.reset_messages()
        return output


def getter_json_output_from_resoning(model_output: str) -> dict:
    matches = re.findall(r'```json.*?```', model_output, re.DOTALL)
    if matches:
        return JsonOutputParser().invoke(matches[-1])
    else:
        logging.warning('No json output found in model output')
        return {}


def create_default_gpt4o(hub_prompt, model_kwargs: dict | None = None):
    model = LangchainWrapper(
        model_name='gpt-4o',  # gpt-4o-2024-11-20
        hub_prompt=hub_prompt,
        model_kwargs=model_kwargs,
    )
    return model


def create_default_gpt4o_mini(hub_prompt, model_kwargs: dict | None = None):
    model = LangchainWrapper(
        model_name='gpt-4o-mini-2024-07-18',
        hub_prompt=hub_prompt,
        model_kwargs=model_kwargs,
    )
    return model


def create_default_gpt35(hub_prompt, model_kwargs: dict | None = None):
    return LangchainWrapper(
        model_name='gpt-3.5-turbo-0125',
        hub_prompt=hub_prompt,
        model_kwargs=model_kwargs,
    )


def create_default_llama405(hub_prompt, model_kwargs=None):
    return LangchainWrapper(
        model_name='accounts/fireworks/models/llama-v3p1-405b-instruct',
        hub_prompt=hub_prompt,
        is_together=True,
        model_kwargs=model_kwargs
    )


def create_default_llama70(hub_prompt, model_kwargs=None):
    return LangchainWrapper(
        model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        hub_prompt=hub_prompt,
        is_together=True,
        model_kwargs=model_kwargs
    )


def create_default_llama31_8b(hub_prompt, model_kwargs=None):
    return LangchainWrapper(
        model_name='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        hub_prompt=hub_prompt,
        is_together=True,
        model_kwargs=model_kwargs
    )


def create_default_llama32_3b(hub_prompt, model_kwargs=None):
    return LangchainWrapper(
        model_name='meta-llama/Llama-3.2-3B-Instruct-Turbo',
        hub_prompt=hub_prompt,
        is_together=True,
        model_kwargs=model_kwargs
    )


def create_default_gemma_2b(hub_prompt, model_kwargs=None):
    return LangchainWrapper(
        model_name='google/gemma-2b-it',
        hub_prompt=hub_prompt,
        is_together=True,
        model_kwargs=model_kwargs
    )


def create_default_qwen_coder(hub_prompt, model_kwargs=None):
    return LangchainWrapper(
        model_name='Qwen/Qwen2.5-Coder-32B-Instruct',
        hub_prompt=hub_prompt,
        is_together=True,
        model_kwargs=model_kwargs
    )
