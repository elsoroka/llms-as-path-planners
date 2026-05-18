import os
import json
import time
import sys
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


eng = []
acts = []

test_file = sys.argv[1]
eng = json.load(open(test_file))
with open(str(sys.argv[2]), 'r') as f:
    prompt = f.read()

out = []

inp_tokens = 0

inputs = []
for i in range(len(eng)):

    print('Processing example #', i)

    try:

        prompt_=f""" {prompt[:-1]}
###
Task: {eng[i]['nl_description']}
Actions: 
        """
        print(prompt_)

        inputs.append({
            'prompt': prompt_,
            'token_count': inp_tokens
        })

        response = client.chat.completions.create(
            model="gpt-4",
            messages = [{"role": "user", "content": prompt_}],
            temperature = 0.0,
            max_tokens=250
        )

        print(response)
        test = {
            "english": eng[i]['nl_description'],
            "ground_truth":eng[i]['agent_as_a_point'],
            "predicted": response.choices[0].message.content,
            'world': eng[i]['world'],
            'prompt_tokens': response.usage.prompt_tokens,
            'gen_tokens': response.usage.completion_tokens,
            'prompt': prompt_
        }
          
        out.append(test)
    except Exception as e: 
        print(e)
        time.sleep(25)
        prompt_=f""" {prompt[:-1]}
###
Task: {eng[i]['nl_description']}
Actions: 
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages = [{"role": "user", "content": prompt_}],
            temperature = 0.0,
            max_tokens=250
        )

        print(response)
        test = {
            "english": eng[i]['nl_description'],
            "ground_truth":eng[i]['agent_as_a_point'],
            "predicted": response.choices[0].message.content,
            'world': eng[i]['world'],
            'prompt_tokens': response.usage.prompt_tokens,
            'gen_tokens': response.usage.completion_tokens
        }
          
        out.append(test)
    
    with open('outputs/'+ os.path.basename(str(sys.argv[2])), 'w') as fo:
      json_object = json.dumps(out, indent = 4)
      fo.write(json_object)
      fo.write('\n')
   