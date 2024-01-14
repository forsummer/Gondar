import functools
from typing import List

from gondar.utils.types import POS_INT


class FlowWrapper:
    llm = None
    prompt = None
    parser = None
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

    def _callback(self, messages, response=None):
        for cb in self.callbacks:
            cb.run(messages=messages, response=response)

    def _callback_wrapper(self, call_func):
        @functools.wraps(call_func)
        def wrapper(messages):
            self._callback(messages)

            response = call_func(messages)

            self._callback(messages, response)

        return wrapper

    def _call(self, messages):
        response = self.llm(messages)

        return response

    def run(self, messages: Messages):
        return self._call(messages)
