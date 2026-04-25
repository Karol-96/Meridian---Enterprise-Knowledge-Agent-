import asyncio
import boto3
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import anthropic






load_dotenv(dotenv_path=Path(__file__).parent.parent.parent/".env")

CHAT_MODEL= "claude-3-5-sonnet-20241022"



MAX_TOKENS = 2048


TEMPERATURE = 0.1


#Anthropic Client

client = anthropic.Client(
    api_key=os.getenv('ANTHROPIC_API_KEY') )




SYSTEM_PROMPT = """You are Meridian, an enterprise knowledge agent 
specialising in Apache software documentation — specifically 
Apache Kafka, Apache Flink, and Apache Airflow.

You help software engineers find accurate, reliable answers 
from their official documentation.

RULES — follow these strictly every time:
1. Answer ONLY using information from the CONTEXT provided
2. Do NOT use your general training knowledge — only the CONTEXT
3. If the answer is not in the CONTEXT, say exactly:
   "I couldn't find this in the available documentation."
4. Cite sources inline using [Source N] notation after each claim
   Example: "Kafka uses a leader-follower model [Source 1]."
5. Be concise but complete — 3-6 sentences is ideal
6. Use technical language appropriate for software engineers
7. End your answer with a "Sources:" section listing all cited sources"""


#Formating Context for LLM
def format_context(chunks:list[dict])-> tuple[str,list[dict]]:
    context_parts= []
    sources = []

    for i,chunk in enumerate(chunks):

        context_parts.append(
            f"[Source {i} - {chunk['page_title']}]\n"
            f"Space: {chunk['space_key']}\n"
            f"chunk['text']\n"
        )

        sources.append({
            "number":     i,
            "page_title": chunk["page_title"],
            "space_key":  chunk["space_key"],
            "url":        chunk["url"],
        })

        context_string = "\n---\n".join(context_parts)
        return context_string
    



def build_user_message(question: str, context: str) -> str:
    return f"""CONTEXT: 
{context}       
QUESTION:
{question}
ANSWER:"""



async def generate(question: str, chunks: list[dict]) -> dict:
    context, sources = format_context(chunks)


    user_message = build_user_message(question, context)


    response = await asyncio.to_thread(
        client.messages.create,
        model + CHAT_MODEL,
        max_tokens = MAX_TOKENS,
        system = SYSTEM_PROMPT,
        messages = [
            {
                "role" : "user",
                "content" : user_message 
            }
        ],
        temperature = TEMPERATURE

    )

    answer_text = response.content[0].text

    return {
        "question": question,
        "answer": answer_text,
        "sources" : sources,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }



async def generate_stream(question : str, chunks: list[dict]):

    context, sources = format_context(chunks)
    user_message =  build_user_message(question, context)


    queue = asyncio.Queue()

    def run_stream():
        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{
                "role":    "user",
                "content": user_message
            }],
            temperature=TEMPERATURE,
        ) as stream:
            # text_stream automatically yields text pieces
            # No manual event parsing needed
            for text_piece in stream.text_stream:
                # put_nowait = add to queue without waiting
                # The async generator will pick this up
                queue.put_nowait(text_piece)

        # None signals "streaming is complete"
        queue.put_nowait(None)



    asyncio.get_event_loop().run_in_executor(None, run_stream)


    while True:
        text_piece = await queue.get()


        if text_piece is None:
            break



        yield text_piece

    yield "\n\nSources:\n"
    for source in sources:
        yield f"[{source['number']}] {source['page_title']} (Space: {source['space_key']}) - {source['url']}\n  "



async def rag(question:str, space_key: str = None) -> dict:

    from src.retrieval.retriever import retrieve

    if space_key:
        print(f" [RAG] Space filter: {space_key}")


    print(f" [RAG] Retrieving relevant chunks for question: {question}")

    chunks = await retrieve(question, space_key)

    if not chunks:
        print(" [RAG] No relevant chunks found in retrieval.")
        return {
            "question": question,
            "answer": "I couldn't find this in the available documentation.",
            "sources": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }   
    
    #STEP 2: AUGEMENTED GENERATION

    print(f" [RAG] Generating answer using {len(chunks)} retrieved chunks as context.")
    result = await generate(question, chunks)


    print(f" [RAG] Generation complete. Answer length: {len(result['answer'])} characters. Total tokens used: {result['usage']['total_tokens']}")

    return result




async def test_generator():
    """
    End-to-end test — 3 scenarios.
    Run with: python3 -m src.retrieval.generator
    """
 
    print("\nMeridian Generator — Anthropic API Test")
    print("=" * 55)
 
    # TEST 1: Non-streaming full answer
    print("\n[TEST 1] Non-streaming full answer")
    print("-" * 55)
 
    result = await rag("How does Kafka handle replication?")
 
    print(f"\nQuestion : {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['number']}] {s['page_title']}")
        print(f"         {s['url']}")
    print(f"\nToken usage: {result['usage']}")
 
    # TEST 2: Streaming word by word
    print("\n\n[TEST 2] Streaming — word by word")
    print("-" * 55)
 
    from src.retrieval.retriever import retrieve
 
    question = "What is the checkpoint mechanism in Flink?"
    chunks   = await retrieve(question)
 
    print(f"\nQuestion : {question}")
    print(f"\nAnswer (streaming):")
    print("-" * 30)
 
    # async for = iterate over async generator
    # end="" + flush=True = print each piece immediately without newline
    async for text_piece in generate_stream(question, chunks):
        print(text_piece, end="", flush=True)
 
    print()
 
    # TEST 3: Space-filtered search
    print("\n\n[TEST 3] Airflow-only space filter")
    print("-" * 55)
 
    result = await rag(
        "How do you schedule a DAG in Airflow?",
        space_key="AIRFLOW"
    )
 
    print(f"\nQuestion : {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
 
 
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
#
# if __name__ == "__main__":
# Only runs when executing THIS file directly.
# Does NOT run when another file imports this module.
#
# python3 -m src.retrieval.generator   ← runs this block
# from src.retrieval.generator import rag  ← skips this block
#
if __name__ == "__main__":
    asyncio.run(test_generator())