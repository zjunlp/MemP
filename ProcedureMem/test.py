import json
with open("/Users/fangrunnan/Documents/zjunlp/work/code/memp/ProcedureMem/Alfworld/alfworld_traj.json", "r") as f:
    data = json.load(f)

def process_format(d):
    query = d["conversations"][2]["value"].split("\n\nYour task is to: ")[-1].split('.')[0]
    format_data = {
        "query": query,
        "trajectory": d["conversations"],
        "facts": None,
        "source": "alfworld"
    }
    return format_data
format_data = []
for d in data:
    format_data.append(process_format(d))

with open("/Users/fangrunnan/Documents/zjunlp/work/code/memp/ProcedureMem/Alfworld/alfworld_format_traj.json", "w") as f:
    json.dump(format_data, f, ensure_ascii=True, indent=4)