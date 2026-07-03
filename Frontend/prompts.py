import json
import re

import httpx
import openai
from openai import AsyncOpenAI

import config

AI_SUMMARY_PROMPT = """"I would like you to help me by summarizing a group of tweets, delimited by triple backticks, and each tweet is labeled by a number in a given format: number-[tweet]. Give me a comprehensive summary in a concise paragraph and as you generate each sentence, provide the identifying number of tweets on which that sentence is based:"""


class BackendError(Exception):
    """An LLM backend is unreachable or returned an unusable response."""

    def __init__(self, backend, url, message):
        self.backend = backend
        self.url = url
        super().__init__(f"{backend} backend at {url}: {message}")


_summarizer_client = None
_stance_client = None


def _get_summarizer_client():
    global _summarizer_client
    if _summarizer_client is None:
        _summarizer_client = openai.OpenAI(
            base_url=config.SUMMARIZER_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=config.SUMMARY_TIMEOUT_S,
            max_retries=1,
        )
    return _summarizer_client


def _get_stance_client():
    global _stance_client
    if _stance_client is None:
        _stance_client = AsyncOpenAI(
            base_url=config.STANCE_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=config.STANCE_TIMEOUT_S,
            max_retries=1,
        )
    return _stance_client


def check_backends():
    """Ping both vLLM servers. Returns {"summarizer": bool, "stance": bool}; never raises."""
    status = {}
    for name, base_url in (
        ("summarizer", config.SUMMARIZER_BASE_URL),
        ("stance", config.STANCE_BASE_URL),
    ):
        try:
            resp = httpx.get(
                base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                timeout=5.0,
            )
            status[name] = resp.status_code == 200
        except httpx.HTTPError:
            status[name] = False
    return status


def ai_summarize(tweets):
    llama3_gen_prompt = """system

{}user

{}assistant {}"""
    input_text = llama3_gen_prompt.format(
        AI_SUMMARY_PROMPT,
        tweets,
        ""
    )
    client = _get_summarizer_client()
    try:
        completion = client.chat.completions.create(
            model=config.SUMMARIZER_MODEL,
            messages=[{"role": "user", "content": input_text}],
            temperature=0,
        )
    except openai.OpenAIError as e:
        raise BackendError("summarizer", config.SUMMARIZER_BASE_URL, str(e)) from e
    return completion.choices[0].message.content


def parse_stance_response(text, start, end):
    """Map tweet ids in [start, end) to int stances from a model response.

    Extracts the first {...} block; any id that is missing, out of range of the
    JSON, or has a non-integer value gets -1 instead of raising.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    results = {}
    for j in range(start, end):
        raw = parsed.get(f"tweet-{j}", -1)
        try:
            results[j] = int(raw)
        except (TypeError, ValueError):
            results[j] = -1
    return results


async def stance_annotation(tweets, topic, stances, examples):
    formattedStances = []
    examplePrompt = "### Examples: \ninput:\n"
    for i in range(len(stances)):
        if stances[i] != "":
            currStanceNum = len(formattedStances)
            formattedStances.append(f"{currStanceNum}: {stances[i]}")

    exampleCount = 0
    for key in examples.keys():
        examplePrompt += f"{exampleCount}-{key}\n"
        exampleCount += 1
    examplePrompt += "output:\n{\n"
    exampleCount = 0
    for key in examples.keys():
        examplePrompt += f'"tweet-{exampleCount}": {examples[key]},\n'
        exampleCount += 1
    examplePrompt += "}"
    if len(examples) == 0:
        examplePrompt = ""

    prompt = [
        {
            "role": "system",
            "content": f"You are a human annotator. You will be presented with a list of tweets (labeled with id numbers), delimited by triple backticks, concerning '{topic}'. Please make the following assessment without further commentary:",
        },
        {
            "role": "user",
            "content": f"""
    Determine whether each tweet discusses the topic of '{topic}'. If it does, indicate the stance of the Twitter user who posted the tweet as one of '{formattedStances}', otherwise label the stance as -1. Your response should be in JSON format as shown below, do not provide any other output:
    {{
        "tweet-<tweetID>" : "stance_number"
    }}

    The stance number must be between -1 and {len(formattedStances)-1}, no other text should be in the stance number field.

    {examplePrompt}

    ### Your Task:
    Tweets: ```{tweets}```
    """,
        },
    ]
    client = _get_stance_client()
    request = {"model": config.STANCE_MODEL, "messages": prompt}
    try:
        try:
            completion = await client.chat.completions.create(
                **request, response_format={"type": "json_object"}
            )
        except openai.BadRequestError:
            completion = await client.chat.completions.create(**request)
    except openai.OpenAIError as e:
        raise BackendError("stance", config.STANCE_BASE_URL, str(e)) from e

    return completion.choices[0].message.content
