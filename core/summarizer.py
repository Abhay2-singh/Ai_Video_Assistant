from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from tenacity import retry, wait_exponential, stop_after_attempt

import os


def get_llm():
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "open-mistral-nemo"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


# Retry actual Mistral API call
@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True
)
def invoke_with_retry(chain, input_data):
    return chain.invoke(input_data)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript concisely."
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")

        summary = invoke_with_retry(
            map_chain,
            {"text": chunk}
        )

        chunk_summaries.append(summary)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial "
                "summaries into one final professional meeting summary in "
                "bullet points.",
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return invoke_with_retry(
        combined_chain,
        combined
    )


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short "
                    "professional meeting title (max 8 words). "
                    "Only return the title, nothing else.",
                ),
                (
                    "human",
                    "{text}"
                ),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return invoke_with_retry(
        title_chain,
        transcript[:2000]
    )