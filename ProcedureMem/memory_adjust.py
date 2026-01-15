import json
import os
from litellm import completion
import openai

prompt = """
You are a helpful assistant.
You are given a workflow, a reward, and a trajectory.
Reward is a number between 0 and 1, 1 means the trajectory is successful, 0 means the trajectory is failed.
If the reward is False, which means that the trajectory guided by the workflow did not successfully complete the task, then you need to analyze why the task was not completed successfully based on the trajectory and the workflow.
After that, refine the workflow to make it more accurate and robust, so that it can better guide the completion of the task.

Workflow:
{workflow}

Reward:
{reward}

Trajectory:
{trajectory}

Output Example Workflow:
'pick_and_place': 'When looking for an object, if you want to find a kitchen-related object like a spatula, you should start from the most possible locations. The action workflows are as follow:\n1)go to receptacle\n2)take object from receptacle\n3)go to the place to put the object\n4)put object in/on receptacle',
'pick_clean_then_place': 'When pick an object, clean it and place, you should first go to the possible locations of the object, then take the object, clean it, and put it in the place. The action workflows are as follow:\n1)go to receptacle\n 2)take object from receptacle\n3)clean object with receptacle\n4)go to the place to put the object\n5)put object in/on receptacle',
'pick_heat_then_place': 'When pick an object, heat it and place, you should first go to the possible locations of the object, then take the object, heat it with micorwave, and put it in the place. The action workflows are as follow:\n1)go to receptacle\n 2)take object from receptacle\n3)heat object with receptacle\n4)go to the place to put the object\n5)put object in/on receptacle',
'pick_cool_then_place': 'When pick an object, cool it and place, you should first go to the possible locations of the object, then take the object, cool it with fridge, and put it in the place. The action workflows are as follow:\n1)go to receptacle\n 2)take object from receptacle\n3)cool object with receptacle\n4)go to the place to put the object\n5)put object in/on receptacle',
'look_at_obj': 'When look at an object to find it, before you open receptacle, you should first go to the possible locations of the object, then open the receptacle to find the object. The action workflows are as follow:\n1)go to receptacle\n 2)open receptacle\n3)take object from receptacle\n4)close receptacle\n5)go to the place to put the object',
'pick_two_obj': 'When pick two objects, you should pick object one by one, and put them in the place one by one. The action workflows are as follow:\n1)go to receptacle\n 2)take object from receptacle\n3)go to the place to put the object\n4)put object in/on receptacle'


Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Workflow> your adjusted workflow here </Workflow>
"""


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


def adjust_memory(worfklow, reward, trajectory):
    input_prompt = prompt.format(workflow=worfklow, reward=reward, trajectory=trajectory)
    output = llm(input_prompt)
    new_workflow = output.split("<Workflow>")[1].split("</Workflow>")[0]
    return new_workflow


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
    os.environ['OPENAI_API_BASE'] = "YOUR_API_BASE"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    workflow = "You are a helpful assistant. You need to help the user to find the answer to the question."
    reward = 1
    trajectory = "The user asked me to find the answer to the question. I found the answer and told the user."
    new_workflow = adjust_memory(workflow, reward, trajectory)
    print(new_workflow)
