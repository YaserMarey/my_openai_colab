### Techniques for Improving Reliability of LLMs
Although some of the techniques are specific to certain types of problems, many of them are built upon general principles that can be applied to a wide range of tasks:
- Give clearer instructions
- Split complex tasks into simpler subtasks
- Structure the instruction to keep the model on task
- Prompt the model to explain before answering
- Ask for justifications of many possible answers, and then synthesize
- Generate many outputs, and then use the model to pick the best one
- Fine-tune custom models to maximize performance

By giving the model more time and space to think, and guiding it along a reasoning plan, it's able to figure out the correct answer.
# An example of a system message that primes the assistant to explain concepts in great depth
```
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a friendly and helpful teaching assistant. You explain concepts in great depth using simple terms, and you give examples to help people learn. At the end of each explanation, you ask a question to check for understanding"},
        {"role": "user", "content": "Can you explain how fractions work?"},
    ],
    temperature=0,
)
print(response["choices"][0]["message"]["content"])
```


# An example of a system message that primes the assistant to give brief, to-the-point answers
```
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a laconic assistant. You reply with brief, to-the-point answers with no elaboration."},
        {"role": "user", "content": "Can you explain how fractions work?"},
    ],
    temperature=0,
)

print(response["choices"][0]["message"]["content"])
```


# Example OpenAI Python library request
```MODEL = "gpt-3.5-turbo"
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=0,
)```

# example without a system message
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Explain asynchronous programming in the style of the pirate Blackbeard."},
    ],
    temperature=0,
)

print(response['choices'][0]['message']['content'])



[
  {"role": "system", "content": "You are a helpful assistant that translates English to French."},
  {"role": "user", "content": 'Translate the following English text to French: "{text}"'}
]


messages=[
    {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
    {"role": "system", "name":"example_user", "content": "New synergies will help drive top-line growth."},
    {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
    {"role": "system", "name":"example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
    {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
    {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
]


payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a snarky sarcastic chatbot."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "user", "content": "Where was it played?"}
    ]
}



You can put the ‘context’ or the entire ‘prompt’ into the messages parameter like that:


prompt=f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:""

completion = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "system", "content": f"{prompt}"}]
)



to prevent halucination

completion = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "system", "content": f"Answer the question as truthfully as possible, and if you're unsure of the answer, say \"Sorry, I don't know\".\\n{context}\n\n{conversation_history}"},
                          {"role": "user", "content": f"{question}"}]
)