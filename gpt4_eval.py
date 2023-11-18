import openai
import json
import argparse
import tqdm
import time

def inference_manager(message_list, manager_system_message):

    prompt = ''
    print(message_list)
    for m in message_list:
        prompt += m['content']
        prompt += '\n'

    _response = openai.ChatCompletion.create(
        model=args.model,
        messages=[{"role": "system", "content": manager_system_message + prompt}],
        temperature=0,
        max_tokens=5,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=1
    )

    all_responses = [_response['choices'][i]['message']['content'] for i in
                     range(len(_response['choices']))]

    return all_responses[0]


def inference_multi(api_key, scorer_system_message):
    import autogen

    config_list_gpt4 = [
        {
            'model': 'gpt-4',
            'api_key': api_key,
        },
    ]

    llm_config = {"config_list": config_list_gpt4, 'temperature': 0, 'n': 1}
    #cache_seed

    scorer_1 = autogen.AssistantAgent(
        name="scorer_1",
        system_message=scorer_system_message,
        llm_config=llm_config,
    )

    scorer_2 = autogen.AssistantAgent(
        name="scorer_2",
        system_message=scorer_system_message,
        llm_config=llm_config,
    )

    scorer_3 = autogen.AssistantAgent(
        name="scorer_3",
        system_message=scorer_system_message,
        llm_config=llm_config,
    )

    groupchat = autogen.GroupChat(agents=[scorer_1, scorer_2, scorer_3], messages=[], max_round=3)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    scorer_1.initiate_chat(manager, message='')

    return groupchat.messages


#openai.api_key = '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'
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='./prompts/summeval/con_detailed.txt')
    argparser.add_argument('--prompt_scorer', type=str, default='./prompts/summeval/con_scorer.txt')
    argparser.add_argument('--prompt_manager', type=str, default='./prompts/summeval/con_manager.txt')
    argparser.add_argument('--save_fp', type=str, default='./results/gpt4_con_detailed_openai.json')
    argparser.add_argument('--summeval_fp', type=str, default='./data/summeval.json')
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-4-0613')
    args = argparser.parse_args()
    openai.api_key = args.key

    import pandas as pd
    from datasets import load_dataset

    dataset = load_dataset("mteb/summeval")
    sampled = pd.DataFrame(dataset['test'])
    sampled = sampled.sample(frac=1, random_state=1234).reset_index(drop=True)
    sampled = sampled[:10]

    #summeval = json.load(open(args.summeval_fp))
    scorer_prompt = open(args.prompt_scorer).read()
    manager_prompt = open(args.prompt_manager).read()

    ct, ignore = 0, 0

    new_json = []
    for i in tqdm.tqdm(range(10)):
        instance = sampled.iloc[i]
        instance_json = {}
        source = instance['text']
        system_output = instance['human_summaries']

        instance_json['source'] = source
        instance_json['system_output'] = system_output

        for s in system_output:
            cur_scorer_prompt = scorer_prompt.replace('{{Document}}', source).replace('{{Summary}}', s)
            cur_manager_prompt = manager_prompt.replace('{{Document}}', source).replace('{{Summary}}', s)

            while True:
                try:
                    res_multi = inference_multi(args.key, cur_scorer_prompt)
                    res_manager = inference_manager(res_multi, cur_manager_prompt)

                    instance_json['result'] = res_manager

                    new_json.append(instance_json)
                    ct += 1
                    break
                except Exception as e:
                    print(e)
                    if ("limit" in str(e)):
                        time.sleep(2)
                    else:
                        ignore += 1
                        print('ignored', ignore)
                        
                        
                        

                        break

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)

