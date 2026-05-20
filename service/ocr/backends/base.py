from abc import ABC, abstractmethod


class VisionBackend(ABC):

    @abstractmethod
    def describe(self, image_path: str, prompt: str, max_tokens: int = 50) -> str:
        """Send an image and a prompt to a vision-capable LLM and return the raw text response."""
