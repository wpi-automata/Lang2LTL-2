import os
import base64
import json
import logging
from time import sleep

import ollama

from models import encode_image
from utils import load_from_file

# OLLAMA_HOST config should be an env variable and is auto pulled by ollama.
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "deepseek-r1:8b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b")

srer_prompt_fpath = os.path.join(os.path.expanduser("~"), "ground", "data", "srer_prompt.txt")

class OllamaClient:

    def chat(self, messages, model=OLLAMA_TEXT_MODEL, stream=False, options=None):
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=stream,
            options=options or {}
        )
        return response["message"]["content"]

    def embed(self, texts, model="mxbai-embed-large:335m"):
        response = ollama.embeddings(
            model=model,
            prompt=texts
        )
        return response["embedding"]

    def extract(self, command):
        messages = [
            {"role": "system", "content": load_from_file(srer_prompt_fpath)},
            {
                "role": "user",
                "content": (
                    "Extract the referring expressions to predicates map, "
                    "lifted command, and symbol map for the following command:\n\n"
                    f"Command:{command}"
                )
            }
        ]
        return self.chat(messages)

    def caption(self, img_fpath):
        complete = False
        ntries = 0
        while not complete:
            try:
                img_b64 = encode_image(img_fpath)
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "What's the most obvious object in this image in one sentence?\n"
                            f"Image (base64): {img_b64}"
                        )
                    }
                ]
                raw_response = self.chat(messages, model=OLLAMA_VISION_MODEL)
                complete = True
            except Exception as e:
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec... {e}")
                sleep(30)
                logging.info("OK continue")
                ntries += 1
        return raw_response

    def get_embed(self, txt):
        txt = json.dumps(txt).replace("\n", " ")
        complete = False
        ntries = 0
        while not complete:
            try:
                embedding = self.embed(txt)
                complete = True
            except Exception as e:
                sleep(30)
                print(f"{ntries}: waiting for the server. sleep for 30 sec... {e}")
                print("OK continue")
                ntries += 1

        return embedding

    def translate(self, query, examples):
        complete = False
        ntries = 0
        task = "You are an expert at translating natural language commands to linear temporal logic (LTL) formulas."
        while not complete:
            try:
                messages = [
                    {"role": "system", "content": f"{task}\n\nHere are some examples:\n\n{examples}"},
                    {"role": "user", "content": f"Translate the following command to an LTL formula\n\nCommand: \"{query}\""}
                ]
                raw_response = self.chat(messages)
                complete = True
            except Exception as e:
                sleep(30)
                print(f"{ntries}: waiting for the server. sleep for 30 sec... {e}")
                print("OK continue")
                ntries += 1

        response = raw_response.replace("\"", "").split(": ")[-1]
        return response, None
