import os
import openai
from litellm import completion

import yaml
import alfworld
import alfworld.agents.environment
from alfworld.agents.environment import get_environment
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import sys
from Alfworld.prompts import alfworld_system_prompt
from ProcedureMem.memory import Memory
import copy
import argparse
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ['OPENAI_API_BASE'] = "YOUR_API_BASE"
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["ALFWORLD_DATA"] = "../alfworld/datasets"






def llm(prompt,stop=None, model="YOUR_MODEL_NAME"):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')
    response = completion(
        model=model, 
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['OPENAI_API_BASE'],
        num_retries=10,
        temperature=1,
        stop=stop
    )
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    return "Output Error"



# os.environ["ALFWORLD_DATA"] = "../alfworld/datasets"




# import time
# time.sleep(20)



def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob




def get_example(name,examples_list):
    prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
    for k, v in prefixes.items():
        if name.startswith(k):
            for example in examples_list:
                if example['task'] == v:
                    return example['example']
    assert False, f'{name} not found'


def alfworld_run_batch(obs:list=[],names:list=[],few_shot=True,max_steps=30,examples_list=[]):
    # Initialize messages list and active tasks
    messages_list = []
    active_tasks = list(range(len(obs)))  # Track active task indices
    



    for ob, name in zip(obs, names):
        messages = []
        messages.append({"role": "system", "content": alfworld_system_prompt})
        if few_shot:
            example = get_example(name,examples_list)
            example_copy = copy.deepcopy(example)
            example_copy[0]['content'] = "Here is an example of how to solve the task:\nExample:\n" + example_copy[0]['content']

            messages.extend(example_copy)
            messages.append({"role": "user", "content": "Now it's your turn.\n" + ob})
        else:
            messages.append({"role": "user", "content": ob})
        messages_list.append(messages)

    for i in range(max_steps):
        if not active_tasks:  # If no active tasks, break the loop
            break
        print(f'\033[91mActive tasks: {active_tasks}\033[0m')


        responses = {}
        with ThreadPoolExecutor(max_workers=len(active_tasks)) as executor:
            futures = {executor.submit(llm, messages_list[idx]): idx for idx in active_tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    response = future.result()
                    print(f'\033[92mAgent {idx}: \n{response}\033[0m')
                    responses[idx] = response
                except Exception as e:
                    print(f'Error {idx}: {e}')
                    continue
        
        responses = dict(sorted(responses.items(), key=lambda x: x[0]))
        for idx, response in responses.items():
            messages_list[idx].append({"role": "assistant", "content": response})
        
        # Only process actions for active tasks
        action_list = [""] * len(obs)  # Initialize with empty actions
        for idx in active_tasks:
            if idx in responses:
                if 'Action: ' in responses[idx]:
                    action_list[idx] = responses[idx].split('Action: ')[-1].strip()
                else:
                    action_list[idx] = ""
        
        observation, reward, done, info = env.step(action_list)
        observation = [process_ob(ob) for ob in observation]
        print(f'\033[93mObservation: \n{observation}\033[0m')
        reward = info['won']

        # Update active tasks list
        new_active_tasks = []
        for idx in active_tasks:
            messages_list[idx].append({"role": "user", "content": f'Observation: {observation[idx]}'})
            if not done[idx]:
                new_active_tasks.append(idx)
        active_tasks = new_active_tasks
    
    return [{"messages": messages, "reward": reward, "name": name} for messages, reward, name in zip(messages_list, reward, names)]
        


def main(args):
    model_name = args.model
    with open('Alfworld/base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    if args.split == 'dev':
        split = "eval_in_distribution"
    else:
        split = "eval_out_of_distribution"

    output_path = f'Alfworld/results/{model_name}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_memory_{args.use_memory}'


    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    #  memory init
    if args.use_memory:
        Memory_config = yaml.safe_load(open('ProcedureMem/config.yaml'))
        Memory_config["memory_dir"] = f"{Memory_config['memory_dir']}_{args.exp_name}"
        Pro_Mem = Memory(**Memory_config)

    # env init

    import json
    folder = './Alfworld'
    prompt_file = 'alfworld_examples.json'

    with open(folder + prompt_file, 'r') as f:
        examples_list = json.load(f)


    import math
    from tqdm import tqdm
    finished_games = 0
    all_reward = 0


    # load finished games
    for file in os.listdir(output_path):
        if file.endswith('.json'):
            finished_games += 1
            with open(f'{output_path}/{file}', 'r') as f:
                result = json.load(f)
                all_reward += result['reward']



    for idx in tqdm(range(math.ceil(num_games/env.batch_size))):

        ob_list, info = env.reset()
        if idx*env.batch_size + env.batch_size <=finished_games:
            continue
        ob_list = ['\n'.join(ob.split('\n\n')[1:]) for ob in ob_list]
        query_list = [ob.split("\nYour task is to: ")[-1] for ob in ob_list]

        new_ob_list = []
        workflow_list = []
        memory_list = []
        for ob in ob_list:
            query = ob.split("\nYour task is to: ")[-1]
            print(query)
            if args.use_memory and len(Pro_Mem.documents) > 0:
                workflow = Pro_Mem.retrieve(query)
                workflow = [{"task_name": w[0].metadata.get("query"), "guidelines": w[0].metadata.get('workflow')} for w in workflow]
                memory_list.append(workflow[0]["task_name"])
                workflow_list.append(workflow[0]["guidelines"])
                workflow = json.dumps(workflow,indent=4,ensure_ascii=False)
                print(ob+'\n\n'+workflow)

                ob = ob + f'Here are some guidelines of how to solve the similar task:\n{workflow}\n'
                new_ob_list.append(ob)
                ob_list = new_ob_list


        print(ob)
        name_list = ['/'.join(info['extra.gamefile'][i].split('/')[-3:-1]) for i in range(len(ob_list))]
        # get_prompt_list
        batch_results = alfworld_run_batch(obs=ob_list,names=name_list, few_shot=args.few_shot, max_steps=args.max_steps,examples_list=examples_list)


        for result in batch_results:
            all_reward += result['reward']
            finished_games += 1
        tqdm.write(f'Avg reward: {all_reward/finished_games}')

        

        for i, result in enumerate(batch_results):
            with open(f'{output_path}/idx_{idx*env.batch_size+i}.json', 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

        print(f'Finished {idx*env.batch_size+i+1} games')


        if Pro_Mem.is_cold_start == False:
            trajectory_list = [result['messages'] for result in batch_results]
            reward_list = [result['reward'] for result in batch_results]
            Pro_Mem.update(query_list, trajectory_list, reward_list, workflow_list, memory_list)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--use_memory', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    if args.overwrite and os.path.exists(f'Alfworld/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_memory_{args.use_memory}'):
        for file in os.listdir(f'Alfworld/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_memory_{args.use_memory}'):
            os.remove(f'Alfworld/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_memory_{args.use_memory}/{file}')

    # env init
    with open('Alfworld/base_config.yaml') as reader:
        config = yaml.safe_load(reader)

    if args.split == 'dev':
        split = "eval_in_distribution"
    else:
        split = "eval_out_of_distribution"
    env = get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=args.batch_size)
    num_games = len(env.gamefiles)
    print(num_games)
    main(args)


