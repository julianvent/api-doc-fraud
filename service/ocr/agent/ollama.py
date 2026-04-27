import requests
from .base import LLMBackend


class OllamaBackend(LLMBackend):

    def __init__(self, url: str, model: str):
        self._url   = url
        self._model = model

    def complete(self, prompt: str) -> str:
        response = requests.post(self._url, json={
            "model" : self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,   
                "num_predict": 512 
            }
        })
        response.raise_for_status()
        return response.json()["response"]