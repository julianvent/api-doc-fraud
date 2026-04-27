from abc import ABC, abstractmethod


class LLMBackend(ABC):

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt and return the raw text response."""