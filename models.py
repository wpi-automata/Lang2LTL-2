import base64
import os
import logging


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class LLMClient:
    """
    Unified interface for switching between OpenAI and Ollama.
    Uses OpenAI if OPENAI_API_KEY is set, otherwise falls back to Ollama.
    """

    def __init__(self):
        if os.getenv("OPENAI_API_KEY"):
            logging.info("Using OpenAI backend")
            from openai_models import OpenAIClient
            print("Using OPENai")
            self.backend = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            logging.info("Using Ollama backend")
            from ollama_models import OllamaClient
            self.backend = OllamaClient(
            )

    def chat(self, messages, **kwargs):
        return self.backend.chat(messages, **kwargs)

    def embed(self, text, **kwargs):
        return self.backend.embed(text, **kwargs)

    def caption(self, img_fpath, **kwargs):
        return self.backend.caption(img_fpath)

    def translate(self, query, examples, **kwargs):
        return self.backend.translate(query, examples)

    def extract(self, command):
        return self.backend.extract(command)

    def get_embed(self, txt):
        return self.backend.get_embed(txt)
