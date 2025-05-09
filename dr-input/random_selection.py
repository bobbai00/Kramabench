# This script does the following:
# 1. Loads a JSON file with the task data (e.g., 'astronomy.json').
# 2. Randomly selects a specified number of tasks from the loaded data (e.g., 6).
# 3. Saves the selected tasks to a new JSON file placed in dr-input/{domain}/{domain}_random.json
# 4. Gathers all of the files found in sub-directories of data/{domain}/input/
# 5. For each randomly selected task, it creates a new directory in dr-input/{domain}/
# 6. It is given a budget of 10 maximum files per task.
# 7. First, it copies in that folder all necessary files for the task, found in the json file under 'data_sources'.
# 8. If they are more than the budget, it randomly selects a maximum of 10 files from the data_sources and copies them into the task folder.
# 8. If they are less than the budget, it randomly selects (budget remaining - n_files) from all domain files found in the sub-directories of data/{domain}/input/


import os
import json
import numpy as np

np.random.seed(42)  # Set the random seed for reproducibility

domain = "astronomy"  # Specify the domain
FILE_BUDGET = 10  # Maximum number of files per task
SAMPLE_SIZE = 6  # Number of tasks to randomly select

workload_path = f"workload/{domain}.json"
with open(workload_path, 'r') as file:
    data = json.load(file)

tasks = {x["id"]:x for x in data}
# delete subtasks and key funcionalities from tasks
for task_id in list(tasks.keys()):
    if "subtasks" in tasks[task_id]:
        del tasks[task_id]["subtasks"]
    if "key_functionalities" in tasks[task_id]:
        del tasks[task_id]["key_functionalities"]

# Randomly select 6 tasks
selected_tasks = np.random.choice(list(tasks.keys()), size=SAMPLE_SIZE, replace=False)

# Create the output directory if it doesn't exist
output_dir = f"dr-input/{domain}/"
os.makedirs(output_dir, exist_ok=True)
# remove all subfolders in dr-input/{domain}/
for root, dirs, files in os.walk(output_dir):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if dir_path != output_dir:  # Avoid removing the main output directory
            os.system(f"rm -rf {dir_path}")


# Create a new JSON file with the selected tasks
random_tasks = {task_id: tasks[task_id] for task_id in selected_tasks}
# sort the tasks by the number that appears at the end of the task_id
random_tasks = dict(sorted(random_tasks.items(), key=lambda item: int(item[0].split("-")[-1])))
with open(f"{output_dir}{domain}_random.json", 'w') as file:
    json.dump(random_tasks, file, indent=4)

# Gather all files found in sub-directories of data/{domain}/input/
input_dir = f"data/{domain}/input/"
all_files = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        all_files.append(os.path.join(root, file))

# For each randomly selected task, create a new directory and copy files
for task_id in selected_tasks:
    print(f"Processing task {task_id}...")
    task = tasks[task_id]
    task_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Copy files from data_sources
    data_sources = task.get("data_sources", [])
    n_files = len(data_sources)
    
    # If there are more than 10 files, randomly select 10
    if n_files > FILE_BUDGET:
        selected_files = np.random.choice(data_sources, size=10, replace=False)
    else:
        selected_files = data_sources

    for file in selected_files:
        src_file = os.path.join(input_dir, file)
        dst_file = os.path.join(task_dir, os.path.basename(file))
        if os.path.exists(src_file):
            os.system(f"cp {src_file} {dst_file}")

    print("\tGround truth files:", len(selected_files))
    # If there are less than 10 files, randomly select from all domain files
    budget_remaining = FILE_BUDGET - len(selected_files)
    while budget_remaining > 0:
        additional_files = np.random.choice(all_files, size=budget_remaining, replace=False)

        for file in additional_files:
            dst_file = os.path.join(task_dir, os.path.basename(file))
            os.system(f"cp {file} {dst_file}")

        # sometimes you might copy one of selected_files again
        budget_remaining = FILE_BUDGET - len(os.listdir(task_dir))
        print("\tSampling additional files:", len(additional_files))
        print("\tThere are in the folder:", len(os.listdir(task_dir)))
        print("\tBudget remaining:", budget_remaining)

    assert len(os.listdir(task_dir)) == FILE_BUDGET, f"Task {task_id} does not have correct # files ({len(os.listdir(task_dir))})."