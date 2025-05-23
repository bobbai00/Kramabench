# Read all json files from results/deep-research/astronomy

import os
import json


# open workload
n_subtasks = 0

for domain in ['astronomy','archeology', 'legal', 'environment', 'biomedical', 'wildfire']:
    with open(f'workload/{domain}.json', 'r') as f:
        try:
            workload = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {domain}.json: {e}")
            continue

    domain_subtasks = 0
    for task in workload:
        assert 'answer_type' in task.keys(), f"{task['id']} has no answer type!"
        for subtask in task['subtasks']:
            assert 'answer_type' in subtask.keys(), f"{subtask['id']} has no answer type!"
            n_subtasks += 1
            domain_subtasks += 1
    print(f"Domain: {domain}, Number of subtasks: {domain_subtasks}")

print(f"Total number of subtasks: {n_subtasks}")
raise NotImplementedError("Check if all tasks have answer types")
# fpaths = [f for f in os.listdir(f'results/deep-research/{domain}') if f.endswith('.json')]
# parent_results = []
# for fpath in fpaths:
#     with open(os.path.join(f'results/deep-research/{domain}', fpath), 'r') as f:
#         data = json.load(f)
#     basename = os.path.basename(fpath)
#     # Check if the file name is a valid task ID
#     if basename.endswith('.json'):
#         task_id = basename[:-5]
#     else:
#         task_id = basename
#     data['id'] = task_id
#     gold_task = [task for task in workload if task['id'] == task_id][0]
#     data['query'] = gold_task['query']
#     data['reference_answer'] = gold_task['answer']

#     parent_results.append(data)

# # Sort the parent_results list by the 'last digit in the id' field
# parent_results.sort(key=lambda x: int(x['id'].split("-")[-1]))
# # Save parent_results to a json file in the same directory
# with open(f'results/deep-research/{domain}.json', 'w') as f:
#     json.dump(parent_results, f, indent=2)

domains = ['astronomy','archeology', 'legal', 'environment', 'biomedical']
for domain in domains:

    # open workload
    with open(f'workload/{domain}.json', 'r') as f:
        workload = json.load(f)

    task_ids = [task['id'] for task in workload]

    # assert that all task_ids are present in the json file
    with open(f'results/deep-research/{domain}.json', 'r') as f:
        data = json.load(f)
        result_ids = []
        for task in data:
            try:
                result_ids.append(task['id'])
            except :
                breakpoint()
        for task_id in task_ids:
            if task_id not in result_ids:
                print(f'Task ID {task_id} not found in {domain}.json')


# # Take wildfire.json, read the runtimes and update them to be in *60
# domain = 'wildfire'
# output = []
# with open(f'results/deep-research/{domain}.json', 'r') as f:
#     data = json.load(f)
#     for task in data:
#         if 'runtime' in task:
#             task['runtime'] = task['runtime'] * 60
#         else:
#             print(f"Task {task['id']} does not have a runtime field.")
#         gold_task = [task for task in workload if task['id'] == task['id']][0]
#         task['reference_answer'] = gold_task['answer']

#         output.append(task)
#     # Save the updated data back to the file
#     with open(f'results/deep-research/{domain}.json', 'w') as f:
#         json.dump(output, f, indent=2)
