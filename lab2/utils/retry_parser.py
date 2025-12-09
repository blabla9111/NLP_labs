from langchain_core.runnables import Runnable, RunnableConfig
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from typing import Dict, Any, cast
import random


class RetryParser(Runnable):
    def __init__(self, llm: ChatDeepSeek, parser: PydanticOutputParser, retry_temperature: float = 0.3, max_retries=3, delay=1):
        self._retry_llm = llm.bind(temperature=retry_temperature)
        self._parser = parser
        self.max_retries = max_retries
        self.delay = delay

    def invoke(self, input: Dict[str, Any], config: RunnableConfig) -> BaseModel:
        attempts = 0
        llm_response = cast(AIMessage, input["output"]).content
        prompt = cast(PromptValue, input["prompt"])
        while attempts < self.max_retries:
            try:
                if random.random() < 0.0:
                    error_types = [
                        "Случайная ошибка JSON parsing",
                        "Внезапная ошибка валидации",
                        "Неожиданный формат данных",
                        "Ошибка структуры ответа",
                        "Случайный сбой парсера"
                    ]
                    random_error = random.choice(error_types)
                    print(f"🎲 ТЕСТ: Имитация случайной ошибки: {random_error}")
                    raise OutputParserException(random_error)
                return self._parser.parse(llm_response)

            except Exception as e:
                attempts += 1
                llm_response = self._retry_llm.invoke(prompt, config).content

        raise OutputParserException("Не получилось распарсить ответ(")
