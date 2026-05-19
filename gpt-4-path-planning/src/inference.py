from helpers import fixed_length, single_environment, group_environments, decompose_sample
from prompting import n_shot_prompt, next_example, VALS_OOD, VALS_IID
from openai import OpenAI
import argparse
import json
import time
import os
from evaluate import success_sg
import sys
from dotenv import load_dotenv
from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

load_dotenv()

# client and model are set in main() based on --provider / --model args.
client = None
_model = 'gpt-4-turbo'

def inference(prompt, model=None, message_texts=None):
    print(prompt)
    if model is None:
        model = _model
    if message_texts is None:
        message_text = [
            {
                "role": "system",
                "content": prompt
            }
        ]
    else:
        message_text = message_texts

    completion = client.chat.completions.create(
        model=model,
        messages=message_text,
        temperature=0.0,
        max_tokens=200,
        top_p=0.95,
        frequency_penalty=0.25,
        presence_penalty=0,
        stop=None,
        extra_body=vllm_extra_body(model),
    )

    # Strip any <think>...</think> blocks before returning.
    completion.choices[0].message.content = strip_thinking(
        completion.choices[0].message.content
    )
    return completion


def few_shot_inference(train, test_set, n_exemplars, representation):
    type_ = None
    if('AE' in representation):
         type_ = representation.split('_')[1]
         representation = 'AE'

    n_shot = n_shot_prompt(train, n_exemplars, None, AE_type=type_)

    out = []
    for ex in test_set:
        prompt = next_example(ex, n_shot=n_shot, representation=representation) 
        prompt = 'Your output throughout this conversation should only consists of tokens (left, right, up, down). ' + prompt
        try:
            response = inference(prompt)
        except:
            time.sleep(30)
            response = inference(prompt)
            pass
        
        predicted = response.choices[0].message.content
        print('FULL:', predicted)

        if('Thought' in predicted):
            try:
                thought, sequence = predicted.split('Solution: ')
            except:
                sequence = ''

        if('AE' not in representation):
            print(success_sg(ex['world'], predicted))

        out.append({
            'Grid': ex['grid_representation'],
            'Predicted': response.choices[0].message.content,
            'Correct': ex['path'],
            'World': ex['world']
        })
    return out


def main():
    '''
    CLA:
    type of test
        - length: few-shot examples are of the same length as test samples
        - env: few-shot examples and test set are drawn from the same environment
    geometry:
        - rectangle
        - maze
        - zig_zag
    representation:
        - Naive
        - Code
        - Grid
        - AE
        - Grid2Grid
    '''
    global client, _model

    # Parse provider/model args from the tail of argv, leaving positional
    # sys.argv indices intact for the existing logic below.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--provider", choices=["openai", "vllm", "stanford"], default="openai")
    parser.add_argument("--model", default=None,
                        help="Model name. For --provider vllm, use the HuggingFace "
                             "model name (e.g. deepseek-ai/deepseek-moe-16b-chat).")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible endpoint (only used with "
                             "--provider vllm).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism when launching "
                             "vLLM (only used with --provider vllm --launch-vllm). "
                             "Default: 1.")
    parser.add_argument("--launch-vllm", action="store_true",
                        help="Spawn a vLLM server subprocess before running inference. "
                             "The server is terminated automatically when the script exits.")
    known, remaining = parser.parse_known_args()

    if known.provider == "vllm":
        _model = known.model or "deepseek-ai/deepseek-moe-16b-chat"
        if known.launch_vllm:
            launch_vllm_server(_model, known.base_url, known.tensor_parallel_size)
        client = OpenAI(api_key="EMPTY", base_url=known.base_url)
    elif known.provider == "stanford":
        client = OpenAI(
            api_key=os.environ.get("STANFORD_API_KEY"),
            base_url="https://aiapi-prod.stanford.edu/v1",
        )
        _model = known.model or "llama-3.2"
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        _model = known.model or "gpt-4-turbo"

    print(f"Provider: {known.provider}, model: {_model}")
    # Sanitize model name for use in filenames (e.g. HuggingFace "org/model" -> "org_model").
    _model_slug = _model.replace("/", "_")

    # Rebuild sys.argv from the non-option tokens so the positional
    # indices below continue to work.
    sys.argv = [sys.argv[0]] + remaining

    choice = sys.argv[1]

    print(choice)
    geometry = sys.argv[2]
    representation = sys.argv[3]

    if(choice == 'env_from_file'):

        iid_file = open(sys.argv[4])
        ood_file = open(sys.argv[5])

        iid_data = json.load(iid_file)
        ood_data = json.load(ood_file)

        grouped =  group_environments(data=iid_data + ood_data, geometry=geometry)

        n_environments = int(sys.argv[6]) if len(sys.argv) > 6 else 30
        valid = {}
        for id_ in grouped.keys():
           print(len(grouped[id_]['OOD']))
           if(len(grouped[id_]['OOD']) < 4):
               continue
           valid[id_] = grouped[id_]

        # Determine output paths before the loop so we can resume.
        grid_size = len(iid_data[0]['world']) if iid_data else 25
        iid_out_path = f'outputs_fullSet/{choice}_out_5_shot_{geometry}_{representation}_{_model_slug}_iid_fewShot_{grid_size}x{grid_size}.json'
        ood_out_path = f'outputs_fullSet/{choice}_out_5_shot_{geometry}_{representation}_{_model_slug}_ood_fewShot_{grid_size}x{grid_size}.json'

        # Load existing results to enable resuming interrupted runs.
        res_iid = []
        res_ood = []
        if os.path.exists(iid_out_path) and os.path.exists(ood_out_path):
            with open(iid_out_path) as f:
                res_iid = json.load(f)
            with open(ood_out_path) as f:
                res_ood = json.load(f)
            print(f"Resuming: skipping first {len(res_iid)} environments already in output file.")

        already_done = len(res_iid)

        for env_idx, id_ in enumerate(list(valid.keys())[:n_environments]):
           if env_idx < already_done:
               continue

           print(f'Processing environment {id_}')
           test_samples = 5

           iid_values = VALS_IID[geometry]
           ood_values = VALS_OOD[geometry]

           count = {}

           test_iid = []
           for x in grouped[id_]['IID']:
               vv = len(x['path'].split())
               if(vv in count.keys()):
                   continue
               count[vv] = True
               test_iid.append(x)

           train = []

           for x in grouped[id_]['IID']:
               if(x not in test_iid):
                   train.append(x)

           test_ood = grouped[id_]['OOD']

           if(representation == 'Grid'):
                res_iid.append(few_shot_inference(train, test_iid, 5, 'Grid'))
                res_ood.append(few_shot_inference(train, test_ood, 5, 'Grid'))

           if(representation == 'Code'):
                res_iid.append(few_shot_inference(train, test_iid, 5, 'Code'))
                res_ood.append(few_shot_inference(train, test_ood, 5, 'Code'))

           if('AE' in representation):
                res_iid.append(few_shot_inference(train, test_iid, 5, representation))
                res_ood.append(few_shot_inference(train, test_ood, 5, representation))

           if(representation == 'Naive'):
                res_iid.append(few_shot_inference(train, test_iid, 5, 'Naive'))
                res_ood.append(few_shot_inference(train, test_ood, 5, 'Naive'))

           with open(iid_out_path, 'w') as f:
                obj = json.dumps(res_iid, indent=4)
                f.write(obj)
           with open(ood_out_path, 'w') as f:
                obj = json.dumps(res_ood, indent=4)
                f.write(obj)

    if(choice == 'decompose'):
            iid_file = open(sys.argv[4])
            ood_file = open(sys.argv[5])
            max_size = int(sys.argv[6])

            iid_data = json.load(iid_file)
            ood_data = json.load(ood_file)

            grouped =  group_environments(data=iid_data + ood_data, geometry=geometry)

            n_environments = 30
            valid = {}
            for id_ in grouped.keys():
                if(len(grouped[id_]['OOD']) < 5):
                    continue
                valid[id_] = grouped[id_]

            # Determine output paths before the loop so we can resume.
            grid_size = len(iid_data[0]['world']) if iid_data else 25
            iid_out_path = f'../outputs/{choice}_out_5_shot_{geometry}_{representation}_{_model_slug}_iid_fewShot_{grid_size}x{grid_size}.json'
            ood_out_path = f'../outputs/{choice}_out_5_shot_{geometry}_{representation}_{_model_slug}_ood_fewShot_{grid_size}x{grid_size}.json'

            # Load existing results to enable resuming interrupted runs.
            res_iid = []
            res_ood = []
            if os.path.exists(iid_out_path) and os.path.exists(ood_out_path):
                with open(iid_out_path) as f:
                    res_iid = json.load(f)
                with open(ood_out_path) as f:
                    res_ood = json.load(f)
                print(f"Resuming: skipping first {len(res_iid)} environments already in output file.")

            already_done = len(res_iid)

            for env_idx, id_ in enumerate(list(valid.keys())[:n_environments]):
                if env_idx < already_done:
                    continue

                print(f'Processing environment {id_}')
                test_samples = 5

                iid_values = VALS_IID[geometry]
                ood_values = VALS_OOD[geometry]

                count = {}

                test_iid = []
                for x in grouped[id_]['IID']:
                    vv = len(x['path'].split())
                    if(vv in count.keys()):
                        continue
                    count[vv] = True
                    test_iid.append(x)

                train = []

                for x in grouped[id_]['IID']:
                    if(x not in test_iid):
                        train.append(x)

                test_ood = grouped[id_]['OOD']

                decomposed_iid = []
                decomposed_ood = []
                if(representation == 'Grid'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5, 'Grid'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5, 'Grid'))
                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)
                if(representation == 'Code'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5, 'Code'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5, 'Code'))

                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)

                if(representation == 'AE'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5, 'AE'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5, 'AE'))

                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)

                if(representation == 'Naive'):
                        for instance in test_iid:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_iid.append(few_shot_inference(train, compositions, 5,'Naive'))
                        for instance in test_ood:
                            compositions, _ = decompose_sample(instance, max_len=max_size, geometry=geometry)
                            decomposed_ood.append(few_shot_inference(train, compositions, 5,'Naive'))

                        res_iid.append(decomposed_iid)
                        res_ood.append(decomposed_ood)

                with open(iid_out_path, 'w') as f:
                        obj = json.dumps(res_iid, indent=4)
                        f.write(obj)
                with open(ood_out_path, 'w') as f:
                        obj = json.dumps(res_ood, indent=4)
                        f.write(obj)

if __name__ == '__main__':
    main()