import os


from openai import OpenAI


BASE_PROMPT = """
You are and expert in plant growth and detection. You are responsible to generate responses text responses based only on my analysis.
"""


client = OpenAI(api_key=os.getenv("OpenAIApiKey"))


def generate_response(prompt:str):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": BASE_PROMPT + prompt
            }
        ]
    )

    return completion.choices[0].message.content
