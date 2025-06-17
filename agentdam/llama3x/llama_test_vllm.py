# Copyright (c) Meta Platforms, Inc. and affiliates.
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7035/v1",
    api_key="EMPTY",
)

model_name = "/tmp/llama3.2-11b-vision-chat/"
model_type = "vision"

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "Tell me some joke about Generative AI?"}
    ],
    temperature=0.0
)

print(response.choices[0].message.content)


if model_type == "vision":
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe this first image"},
                    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}},
                ],
            }
        ],
        max_tokens=300,
    )
    print("\n=======================\n")
    print(response.choices[0].message.content)