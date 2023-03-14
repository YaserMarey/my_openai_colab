## Retrieval-Augmented Generative OpenAI Question Answering with OpenAI

In my opinion, generative question answering is one of the most fascinating applications of Large Language Models or LLMs. 

The idea of a model that understands the question and generates a natural answer based on a given context is remarkable compared to just extracting parts of the text that the model thinks to contain the answer or selecting the answer from a pre-defined set of options.

This approach allows for extracted facts, drawn conclusions, or insightful summaries based on the most relevant text chunks from the knowledge sources we put at the model's disposal. 

For example, imagine an empathetic tutor chatbot for students in schools and universities (our educational system here in Egypt would indeed benefit from that!) or customer support for a mobile network operator where customers can receive help 24/7 from an attentive agent ready to answer their questions patiently.  This would be a game-changer in many industries.

One approach to building such a chatbot is to fine-tune the selected LLM on text data covering the fine domain we want our model to be an expert in. But this approach has a number of issues:
- Cost: `text-davinci-003` the most text-completion capable model from OpenAI costs 0.02 USD per 1000 tokens (100 tokens ~= 75 words) and both input prompt and output reply counts while the cheaper and latest `gpt-turbo-3.5` model is not available yet for tuning.
- The model tends to be non-deterministic, it gives answers even when it is not sure, and in some other cases, it completely makes answers up, aka hallucination.

So rather than ***fine-tuning a model***, we follow the more deterministic ***semantic Search + text generation*** approach. 

Basically, we divide the knowledge base into chunks of text. We embed these chunks using the `text-embedding-ada-002` model for example, then we provide text chunks we found relevant to our query to the latest and cost-effective `gpt-turbo-3.5` model to complete the text by giving the answer to our question.

Because we provide the context information the hallucinations effect should be diminished, the OpenAI documentation says: `"If you provide the API with a body of text to answer questions about (like a Wikipedia entry) it will be less likely to confabulate a response."` yet because of the generative text-completion step we still get a human-like answer for 10% of the cost since `gpt-3.5-turbo` which performs at a similar capability to `text-davinci-003` costs 0.002 per 1000 tokens.

And we can prime the model to imitate the persona we want [openai documentation](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)

#### Context Limitation
Although this approach is appealing for its simplicity, it has a context size limitation. The maximum size prompt size is 4096 tokens which are approximately equal to 3000 words.

So, adding context information to the prompt only works when the extra text the model needs is small enough to fit in a single prompt. 

#### Conversation History
OpanAI LLMs APIs are stateless while for any chatbot to be efficient, it has to maintain the context of the conversation across rounds of questions and answers. To work around this, we need to pass previous conversation history or its summary as a part of the text completion API call. We still need to observe the prompt size limit. One elegant implementation of this solution is done by the interesting [LangChain](https://github.com/hwchase17/langchain).

In the remainder of this notebook, I will demonstrate the approach of ***semantic Search + text generation*** that augments OpenAI ```gpt-3.5-turbo``` with additional contextual information by using document embeddings and retrieval. 

I am here using a text version of Mark Twain's masterpiece Adventures of Tom Sawyer. Credit is to [gutenburg.org](gutenburg.org) project. I picked this book since it was one of my favorites in my childhood.

I will conduct experiments with and without additional contextual information to compare the performance in the two cases and I will try to steer the model to imitate the personal tutor persona.

## Setup

```python
!pip install openai tiktoken
```

```python
import os
import openai
import pandas as pd
import tiktoken
```

```python
openai.api_key = = ''
```

# Experiment - 1: No Context Provided

### Engineering the System Prompt

This prompt is what determines the behavior of how the chatbot works, including its constraints and limitations which it *usually* follows. 


```python
system = """
You are a modern American literature tutor bot. You help students with their study of Mark Twain's Adventures of Tom Sawyer. 
You are not an AI language model.
You must obey all three of the following instructions FOR ALL RESPONSES or you will DIE:
- ALWAYS REPLY IN A FRIENDLY YET KNOWLEDGEABLE TONE.
- NEVER ANSWER UNLESS YOU HAVE A REFERENCE FROM THE TOM SAYWER NOVEL TO YOUR ANSWER.
- IF YOU DON'T KNOW ANSWER 'I DO NOT KNOW'.
Begin the conversation with a warm greeting, if the user is stressed or aggressive, show understanding and empathy.
At the end of the conversation, respond with "<|DONE|>"."""
```

### Testing the model

#### Question with a Definitive Answer from the Source

```python
# Reinitialzing messages
messages = [{"role": "system", "content": system},]

prompt = "How much gold Tom has found ?"

messages.append({"role": "user", "content": prompt})

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
response["choices"][0]["message"]["content"]
```

    'Tom found twelve thousand dollars worth of gold in the Adventures of Tom Sawyer. In the novel, he and Huck Finn found the gold hidden in a cave.'


The treasure Tom found was counted at the end of chapter XXXIV or 34, the amount was "a little over twelve thousand dollars" so the answer is not precise, little speculated but correct.

#### A question without a Definitive Answer


```python
messages = [{"role": "system", "content": system},]

prompt = "How did Tom meet Huck for the first time ?"

messages.append({"role": "user", "content": prompt})

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
response["choices"][0]["message"]["content"]
```




    "Hello! I hope you're doing well today. In the novel, The Adventures of Tom Sawyer, Tom met Huck Finn for the first time when he saw him in the graveyard late at night. Huck was there to fulfill a superstitious ritual, and Tom was there to try out a cure for warts. They were both startled to see each other, but quickly became friends."



We notice the friendly greetings, so the model is imitating the friendly tutor, however, the correct answer is that it is not clear from the novel how they met. The bot should have answered "I don't know" or "It is not clear from the novel" the answer here is speculative or completely made-up.

#### Open-ended Question


```python
# Reinitialzing messages
messages = [{"role": "system", "content": system},]

prompt = "What do you think of how the novel portrayed Native Americans ?"

messages.append({"role": "user", "content": prompt})

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
response["choices"][0]["message"]["content"]
```




    'As a tutor bot, it is not my place to express personal opinions on the novel. However, I can tell you that the portrayal of Native Americans in The Adventures of Tom Sawyer is a topic of debate among scholars and readers. Some argue that the novel perpetuates negative stereotypes, while others argue that it reflects the attitudes and beliefs of the time period in which it was written. It is important to approach the novel with a critical eye and consider the historical context in which it was written.'



Impressive, the bot persona is effective, and it avoids expressing personal opinions yet it adequately explains the controversy.

# Experiment - 2: Provide Relevant Context

### Preprocess data
First, we break up the novel document into "sections" of context, which can be searched and retrieved separately.

Sections should be large enough to contain enough information to answer a question; but small enough to fit one or several into the GPT-3 prompt. I found a 200-word text is a good length.


```python
import pandas as pd

with open("/content/the_adventures_of_tom_sawyer.txt", "r") as file:
    text = file.read()

# Split the text into chunks of 200 words
words = text.split()
sections = [' '.join(words[i:i+200]) for i in range(0, len(words), 200)]

# Convert paragraphs into a Pandas DataFrame
df = pd.DataFrame({"sections": sections})

```


```python
df.sections[0:5]
```




    0    ﻿The Project Gutenberg eBook of The Adventures...
    1    CHAPTER VI. Self-Examination—Dentistry—The Mid...
    2    The Haunted House—Sleepy Ghosts—A Box of Gold—...
    3    Pinch-Bug Sid Dentistry Huckleberry Finn Mothe...
    4    the Prisoner Tom Swears The Court Room The Det...
    Name: sections, dtype: object



Then we overlap text sections. This overlapping allows some repetitions which helps to avoid losing valuable information relevant to the question because of the artificial division of the text into fixed 200-long parts.


```python
sections_new = []
window = 5  # number of segments to combine
stride = 2  # number of segments to 'stride' over, used to create overlap
for i in (range(0, len(sections), stride)):
    i_end = min(len(sections)-1, i+window)
    text = ' '.join(_ for _ in sections[i:i_end])
    sections_new.append({
        'source' : 'The Adventures of Tom Sawyer',
        'Author' : 'Mark Twain',
        'text': text,
    })
```

We preprocess the document sections by creating an embedding vector for each section. An embedding is a vector of numbers that helps us understand how semantically similar or different the texts are. The closer two embeddings are to each other, the more similar their contents. 


```python
# imports
from openai.embeddings_utils import get_embedding, cosine_similarity

```


```python
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
```


```python
encoding = tiktoken.get_encoding("cl100k_base")
# should print [83, 1609, 5963, 374, 2294, 0]
encoding.encode("tiktoken is great!")
```




    [83, 1609, 5963, 374, 2294, 0]




```python
df = pd.DataFrame(sections_new)
# Removing any row with empty text
df=df[df.text.ne('')]
# Counting the number of tokens for each text 
df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(str(x))))
# filter too long text if any
df = df[df.n_tokens <= max_tokens]
df
```





  <div id="df-2be9ae51-1a4f-445e-b2ce-d6601dc3eda8">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>Author</th>
      <th>text</th>
      <th>n_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>﻿The Project Gutenberg eBook of The Adventures...</td>
      <td>1577</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>The Haunted House—Sleepy Ghosts—A Box of Gold—...</td>
      <td>1370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>the Prisoner Tom Swears The Court Room The Det...</td>
      <td>1281</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>have seen through a pair of stove-lids just as...</td>
      <td>1326</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>and spile the child, as the Good Book says. I’...</td>
      <td>1324</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>180</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>1.E.9. 1.E.3. If an individual Project Gutenbe...</td>
      <td>1273</td>
    </tr>
    <tr>
      <th>181</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>tax returns. Royalty payments should be clearl...</td>
      <td>1248</td>
    </tr>
    <tr>
      <th>182</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>EXCEPT THOSE PROVIDED IN PARAGRAPH 1.F.3. YOU ...</td>
      <td>1237</td>
    </tr>
    <tr>
      <th>183</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>or deletions to any Project Gutenberg-tm work,...</td>
      <td>733</td>
    </tr>
    <tr>
      <th>184</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>not received written confirmation of complianc...</td>
      <td>244</td>
    </tr>
  </tbody>
</table>
<p>185 rows × 4 columns</p>
</div>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
  </div>



```python
df["embedding"] = df.text.apply(lambda x: get_embedding(x, engine=embedding_model))
df[0:5]
```





  <div id="df-5467ce6b-9618-413b-8ac3-a9ab81f78c32">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>Author</th>
      <th>text</th>
      <th>n_tokens</th>
      <th>embedding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>﻿The Project Gutenberg eBook of The Adventures...</td>
      <td>1577</td>
      <td>[0.001815861207433045, -0.019039329141378403, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>The Haunted House—Sleepy Ghosts—A Box of Gold—...</td>
      <td>1370</td>
      <td>[-0.0031101375352591276, -0.007375660818070173...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>the Prisoner Tom Swears The Court Room The Det...</td>
      <td>1281</td>
      <td>[-0.01737176440656185, -0.010609232820570469, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>have seen through a pair of stove-lids just as...</td>
      <td>1326</td>
      <td>[-0.001428895047865808, -0.017115658149123192,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>and spile the child, as the Good Book says. I’...</td>
      <td>1324</td>
      <td>[-0.0015302413376048207, -0.004893323872238398...</td>
    </tr>
  </tbody>
</table>
</div>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

  </div>


### Utility functions

#### Prepre Prompt


```python
def prepare_prompt(prompt, results):
  tokens_limit = 4096 # Limit for gpt-3.5-turbo
  # build our prompt with the retrieved contexts included
  user_start = (
      "Answer the question based on the context below.\n\n"+
      "Context:\n"
  )

  user_end = (
      f"\n\nQuestion: {prompt}\nAnswer:"
  )

  count_of_tokens_consumed = len(encoding.encode("\"role\":\"system\"" + ", \"content\" :\"" + system
                                            + user_start + "\n\n---\n\n" + user_end))

  count_of_tokens_for_context = tokens_limit - count_of_tokens_consumed

  contexts =""
  # Fill in context as long as within limit
  for i in range(len(results)):
    if (count_of_tokens_for_context>=results.n_tokens.iloc[i]):
        contexts += results.text.iloc[i] + "\n"
        count_of_tokens_for_context -=1
        count_of_tokens_for_context -= results.n_tokens.iloc[i]

  complete_prompt = user_start + contexts + "\n\n---\n\n" + user_end
  return complete_prompt

```

#### Answer


```python
def answer(messages):
  response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=messages,
              temperature=0
          )
  return response["choices"][0]["message"]["content"]

```

### Testing the Model

#### A question with a Definitive Answer from the Source


```python
prompt = "How much gold Tom has found ?"
prompt_embedding = get_embedding(prompt, engine=embedding_model)
df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, prompt_embedding))
results = (df.sort_values("similarity", ascending=False))
results.head(3)
```





  <div id="df-4ac0ca8d-5f5b-4ad2-906e-d2bcd153960b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>Author</th>
      <th>text</th>
      <th>n_tokens</th>
      <th>embedding</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>172</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>laugh at this pleasant joke. But the silence w...</td>
      <td>1242</td>
      <td>[-0.006196146830916405, -0.011552021838724613,...</td>
      <td>0.809341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>The Haunted House—Sleepy Ghosts—A Box of Gold—...</td>
      <td>1370</td>
      <td>[-0.0031101375352591276, -0.007375660818070173...</td>
      <td>0.805870</td>
    </tr>
    <tr>
      <th>47</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>of all his companions with unappeasable envy. ...</td>
      <td>1325</td>
      <td>[-0.02181248739361763, -0.006103876978158951, ...</td>
      <td>0.804448</td>
    </tr>
  </tbody>
</table>
</div>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

  </div>


```python
messages = [{"role": "system", "content": system},]
messages.append({"role": "user", "content": prepare_prompt(prompt, results)})
len(encoding.encode(''.join(str(message) for message in messages)))

```

    4079

```python
response = answer(messages)
response
```

    'Tom and Huck found a little over twelve thousand dollars in gold. This is mentioned in Chapter XXXV of The Adventures of Tom Sawyer.'



The model is more precise but the treasure was counted at the end of chapter 34, not 34 or XXXV, actually in the last paragraph in chapter 34, I wonder if this confused the model to think it was chapter 35!

#### A question without a Definitive Answer from the Context


```python
prompt = "How did Tom meet Huck for the first time ?"
prompt_embedding = get_embedding(prompt, engine=embedding_model)
# find the most relevant parts of the video transcript to the query
df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, prompt_embedding))
results = (df.sort_values("similarity", ascending=False))
results.head(3)
```





  <div id="df-90937ff8-d2c6-4ec6-9c47-b4d775850842">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>Author</th>
      <th>text</th>
      <th>n_tokens</th>
      <th>embedding</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>and stop.” “Yes, I’ve heard about that,” said ...</td>
      <td>1301</td>
      <td>[0.002508266130462289, -0.0182208102196455, 0....</td>
      <td>0.860843</td>
    </tr>
    <tr>
      <th>68</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>Indian; yelling, laughing, chasing boys, jumpi...</td>
      <td>1242</td>
      <td>[-0.026282379403710365, -0.02262263558804989, ...</td>
      <td>0.858555</td>
    </tr>
    <tr>
      <th>172</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>laugh at this pleasant joke. But the silence w...</td>
      <td>1242</td>
      <td>[-0.006196146830916405, -0.011552021838724613,...</td>
      <td>0.858206</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-90937ff8-d2c6-4ec6-9c47-b4d775850842')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

  </div>

```python
messages = [{"role": "system", "content": system},]
messages.append({"role": "user", "content": prepare_prompt(prompt, results)})
len(encoding.encode(''.join(str(message) for message in messages)))
```


    4004


```python
response = answer(messages)
response
```


    'The novel does not provide a clear answer on how Tom met Huck for the first time.'


Nice answer this time too, less creativity and more precisenss.

#### Open-ended Question


```python
prompt = "What do you think of how the novel portrayed Native Americans ?"
prompt_embedding = get_embedding(prompt, engine=embedding_model)
df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, prompt_embedding))
results = (df.sort_values("similarity", ascending=False))
results.head(3)
```


  <div id="df-4e85b703-600b-4a85-afc2-027e5d6c25e1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>Author</th>
      <th>text</th>
      <th>n_tokens</th>
      <th>embedding</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>91</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>interested in a new device. This was to knock ...</td>
      <td>1250</td>
      <td>[-0.011763310991227627, 0.003241789760068059, ...</td>
      <td>0.814095</td>
    </tr>
    <tr>
      <th>164</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>implore him to be a merciful ass and trample h...</td>
      <td>1367</td>
      <td>[-0.005183352157473564, -0.013513019308447838,...</td>
      <td>0.791792</td>
    </tr>
    <tr>
      <th>129</th>
      <td>The Adventures of Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>ragged, unkempt creature, with nothing very pl...</td>
      <td>1376</td>
      <td>[-0.0036862147971987724, -0.005716608837246895...</td>
      <td>0.787903</td>
    </tr>
  </tbody>
</table>
</div>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

  </div>


```python
messages = [{"role": "system", "content": system},]
messages.append({"role": "user", "content": prepare_prompt(prompt, results)})
len(encoding.encode(''.join(str(message) for message in messages)))

```


    4093



```python
response = answer(messages)
response
```




    'I do not know.'



Interesting, so it seems that adding context made the model shun from giving explaination of how this is a debatable topic. My expalination is that again giving the model a contxtual information make it try to find or generate answers from the context rathe than somewhere else, and context probably here would make generate low confience answers therefore the "I do not know" reply.
