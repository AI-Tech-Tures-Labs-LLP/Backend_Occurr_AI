from groq import Groq
import os

client = Groq(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_BASE_URL"))

def summarize_journals(journal_texts: list[str]) -> str:
    if not journal_texts:
        return "No journal entries to summarize."

    combined_text = "\n\n".join(journal_texts)

    system_prompt = (
        "You are a helpful assistant summarizing a user's emotional journal. "
        "Give a short monthly overview of their mood and experiences."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_API_MODEL")

,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to generate summary: {str(e)}"


