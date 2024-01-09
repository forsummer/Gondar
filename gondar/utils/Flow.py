import functools
from typing import List, Literal

from gondar.utils.base import GondarModel
from gondar.utils.Callback import Callback
from gondar.utils.LLM import LLM
from gondar.utils.Memory import Memory
from gondar.utils.Message import Messages, Responses
from gondar.utils.Parser import Parser
from gondar.utils.PromptTemplate import Prompt
from gondar.utils.types import POS_INT


class FlowWrapper(GondarModel):
    llm: LLM = None
    prompt: Prompt = None
    parser: Parser = None
    memory: List[Memory] = []
    callbacks: List[Callback] = []

    tokens_per_min: POS_INT = 10_000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._call = self.callback_manager(self._call)

    def callback_manager(self, call_func):
        if self.callbacks is []:
            return call_func

        return self._callback_wrapper(call_func)

    def _callback(self, messages: Messages, response: Responses = None):
        for cb in self.callbacks:
            cb.run(messages=messages, response=response)

    def _callback_wrapper(self, call_func):
        @functools.wraps(call_func)
        def wrapper(messages: Messages):
            self._callback(messages)

            response: Responses = call_func(messages)

            self._callback(messages, response)

        return wrapper

    def _call(self, messages: Messages) -> Responses:
        response = self.llm(messages)

        return response

    def run(self, messages: Messages):
        return self._call(messages)
