import time
from openai import OpenAI
import json
from retry import retry
import os

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE_URL")
EMBEDDING_MODEL_KEY = os.getenv("EMBEDDING_MODEL_KEY")
EMBEDDING_MODEL_BASE_URL = os.getenv("EMBEDDING_MODEL_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = 0.2
TOP_P = 1
MAX_TOKENS = 4096

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def get_response(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    if not hasattr(response, "error"):
        return response.choices[0].message.content
    return response.error.message

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def get_llm_response(messages, is_string=False):
    ans = get_response(messages)
    if is_string:
        return ans
    else:
        cleaned_text = ans.strip("`json\n").strip("`\n").strip("```\n")
        ans = json.loads(cleaned_text)
        return ans

from langchain_openai import OpenAIEmbeddings

def get_embedding_model():
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=EMBEDDING_MODEL_KEY,
        openai_api_base=EMBEDDING_MODEL_BASE_URL,
        max_retries=10
    )
    return embedding

if __name__ == "__main__":
    message = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    print(get_response(message))
    # test embedding
    embedding = get_embedding_model()
    print(embedding.embed_query("Hello, how are you?"))