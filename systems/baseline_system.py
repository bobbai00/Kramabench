import sys
sys.path.append('./')
from typeguard import typechecked

from systems.generator_util import Generator
from benchmark.benchmark_api import System
from utils.baseline_utils import * 

QUESTION_PROMPT = """
You are a helpful assistant that generates a plan to solve the given request, and you'll be given:Your task is to answer the following question based on the provided data sources.
Question: {query}
Data file names: {file_names}

The following is a snippet of the data files:
{data}

Now think step-by-step carefully. 
First, provide a step-by-step reasoning of how you would arrive at the correct answer.
Do not assume the data files are clean or well-structured (e.g., missing values, inconsistent data type in a column).
Do not assume the data type of the columns is what you see in the data snippet (e.g., 2012 in Year could be a string, instead of an int). So you need to convert it to the correct type if your subsequent code relies on the correct data type (e.g., cast two columns to the same type before joining the two tables).
You have to consider the possible data issues observed in the data snippet and how to handle them.
Output the steps in a JSON format with the following keys:
- id: always "main-task" for the main task. For each subtask, use "subtask-1", "subtask-2", etc.
- query: the question the step is trying to answer. Copy down the question from above for the main task.
- data_sources: the data sources you need to check to answer the question. Include all the file names you need for the main task.
- subtasks: a list of subtasks. Each subtask should have the same structure as the main task.
For example, a JSON object for the task might look like this:
{example_json}
You can have multiple steps, and each step should be a JSON object.
Your output for this task should be a JSON array of JSON objects.
Mark the JSON array with ````json` and ````json` to indicate the start and end of the code block.

Then, provide the corresponding Python code to extract the answer from the data sources. 
The data sources you may need to answer the question are: {file_paths}.

If possible, print the answer (in a JSON format) to each step you provided in the JSON array using the print() function.
Use "id" as the key to print the answer.
For example, if you have an answer to subtask-1, subtask-2, and main-task (i.e., the final answer), you should print it like this:
print(json.dumps(
{{"subtask-1": answer1, 
"subtask-2": answer2, 
"main-task": answer
}}, indent=4))
You can find a suitable indentation for the print statement. Always import json at the beginning of your code.

Mark the code with ````python` and ````python` to indicate the start and end of the code block.
"""

class BaselineLLMSystem(System):
    """
    A baseline system that uses a large language model (LLM) to process datasets and serve queries.
    """

    def __init__(self, model: str, name="baseline", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dataset_directory = None  # TODO(user): Update me
        self.model = model
        self.llm = Generator(model, verbose=self.verbose)

        # Initialize directories for output and intermediate files
        if variance := kwargs.get('variance'):
            self.variance = variance
        else: self.variance = "one_shot" # Default variance
        # variance has to be one_shot or few_shot
        assert self.variance in ["one_shot", "few_shot"], f"Invalid variance: {self.variance}. Must be one_shot or few_shot."
        
        if verbose := kwargs.get('verbose'):
            self.verbose = verbose
        else: self.verbose = False # Default verbosity
        self.debug = False
        # Set the output directory
        if output_dir := kwargs.get('output_dir'):
            self.output_dir = output_dir
        else: self.output_dir = os.path.join(os.getcwd(), 'testresults') # Default output directory
        self.question_output_dir = None # to be set in run()
        self.question_intermediate_dir = None # to be set in run()
    
    def _init_output_dir(self, query_id:str) -> None:
        """
        Initialize the output directory for the question.
        :param question: Question object
        """
        question_output_dir = os.path.join(self.output_dir, query_id)
        if not os.path.exists(question_output_dir):
            os.makedirs(question_output_dir)
        self.question_output_dir = question_output_dir
        self.question_intermediate_dir = os.path.join(self.question_output_dir, '_intermediate')
        if not os.path.exists(self.question_intermediate_dir):
            os.makedirs(self.question_intermediate_dir)

    def get_input_data(self) -> str:
        """
        Get the input data from the data sources.
        Naively read the first 10 rows of each data source.
        :return: Data as a string
        """
        data_string = ""
        for file_name, data in self.dataset.items():
            if self.verbose: print(f"{self.name}: Reading {file_name}...")
            data_string += f"\nFile name: {file_name}\n"
            data_string += f"Column data types: {data.dtypes}\n"
            data_string += f"Table:\n"
            data_string += get_table_string(data, row_limit=100)
            data_string += "\n"+ "="*20 + "\n"
        return data_string
    
    def generate_error_handling_prompt(self, code_fp:str, error_fp:str) -> str:
        """
        Generate a prompt for the LLM to handle errors in the code.
        :param code_fp: Path to the code file
        :param error_fp: Path to the error file
        :return: Prompt string
        """
        with open(code_fp, 'r') as f:
            code = f.read()
        with open(error_fp, 'r') as f:
            errors = f.read()

        prompt = f"""
        Your task is to fix the following Python code based on the provided error messages.
        Code:
        {code}

        Errors:
        {errors}

        Please provide the fixed code. 
        Mark the code with ````python` and ````python` to indicate the start and end of the code block.
        """
        return prompt
    
    def generate_prompt(self, query:str) -> str:
        """
        Generate a prompt for the LLM based on the question.
        :param query: str
        :return: Prompt string
        """
        # Generate the RAG plan
        # TODO: use process_dataset() to get the data
        data = self.get_input_data()
        file_names = list(self.dataset.keys()) # get the file names from the dataset directory
        file_paths = [os.path.join(self.dataset_directory, file) for file in file_names] # get the file paths

        example_json = {
            "id": "main-task",
            "query": query,
            "data_sources": ["file1.csv", "file2.csv"],
            "subtasks": [{
                "id": "subtask-1",
                "query": "What is the exceedance rate in 2022?",
                "data_sources": ["water-body-testing-2022.csv"]
            },{
                "id": "subtask-2",
                "query": "What is the column name for the exceedance rate?",
                "data_sources": ["water-body-testing-2022.csv"]
            }]
        }
        prompt = QUESTION_PROMPT.format(
            query=query,
            file_names=str(file_names),
            data=data,
            example_json=json.dumps(example_json, indent=4),
            file_paths= str(file_paths) #", ".join(file_paths)
        )

        # Save the prompt to a txt file
        prompt_fp = os.path.join(self.question_output_dir, f"prompt.txt")
        with open(prompt_fp, 'w') as f:
            f.write(prompt)
        if self.verbose:
            print(f"{self.name}: Prompt saved to {prompt_fp}")
        return prompt
    
    def extract_response(self, response, try_number:int):
        """
        Process the LLM response.
        :param response: LLM response string
        :return: Processed response
        """
        json_fp = ""
        code_fp = ""
        # Save the full response to a txt file
        response_fp = os.path.join(self.question_output_dir, f"initial_response.txt")
        with open(response_fp, 'w') as f:
            f.write(response)
        if self.verbose:
            print(f"{self.name}: Response saved to {response_fp}")

        # Assume the step-by-step plan is fixed after the first try
        if try_number == 0:
            # Extract the JSON array from the response
            json_response = extract_code(response, pattern=r'```json(.*?)```')
            #print("Extracted JSON:", json_response)
            # Save the JSON response to a file
            json_fp = os.path.join(self.question_output_dir, f"answer.json")
            with open(json_fp, 'w') as f:
                f.write(json_response)
            if self.verbose:
                print(f"{self.name}: JSON response saved to {json_fp}")
        
        # Extract the code from the response
        code = extract_code(response, pattern=r'```python(.*?)```')
        #print("Extracted Code:", code)
        # Save the code to a file
        code_fp = os.path.join(self.question_output_dir, '_intermediate', f"pipeline-{try_number}.py") #{question.id}-{try_number}
        with open(code_fp, 'w') as f:
            f.write(code)
        if self.verbose:
            print(f"{self.name}: Code saved to {code_fp}")

        return json_fp, code_fp
    
    def execute_code(self, code_fp, try_number:int):
        """
        Execute the code in the file and save the output.
        :param code_fp: Path to the code file
        :return: Execution result
        """
        # Execute the code and save error messages
        result = subprocess.run(
            ['python', code_fp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Save the printed output of the code execution
        output_fp = os.path.join(self.question_intermediate_dir, f"pipeline-{try_number}_out.json")
        with open(output_fp, 'w') as f:
            # Clean NaN values to null for strict JSON compliance
            stdout = clean_nan(result.stdout)
            f.write(stdout)

        # Save only compile/runtime errors
        error_fp = os.path.join(self.question_intermediate_dir, f"errors-{try_number}.txt")
        with open(error_fp, 'w') as f:
            f.write(result.stderr)

        return output_fp, error_fp
    
    @typechecked
    def _format_answer_on_fail(self, answer: Dict | List, error_msg: str = "Warning: No answer found in the Python pipeline.") -> Dict[str, str | Dict | List]:
        formatted_answer = {}
        for step in answer:
            if step["id"] == "main-task":
                formatted_answer |= step
                if "answer" not in formatted_answer:
                    formatted_answer["answer"] = error_msg
            else:
                if "subtasks" in formatted_answer:
                    found = False
                    for subtask in formatted_answer["subtasks"]:
                        if subtask["id"] == step["id"]:
                            found = True
                        if "answer" not in subtask:
                            subtask["answer"] = error_msg
                    if not found:
                        formatted_answer["subtasks"].append(step)
        return formatted_answer
    
    @typechecked
    def process_response(self, json_fp, output_fp, error_fp=None) -> Dict[str, str | Dict | List]:
        """
        Process the response and fill in the JSON response with the execution result.
        :param question: Question object
        :param json_fp: Path to the JSON file
        :param output_fp: Path to the output file
        """
        # Load the JSON response
        try: 
            with open(json_fp, 'r') as f:
                answer = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: {self.name}: ** ERRORS ** decoding answer JSON: {e}")
            return {
                "id": "main-task",
                "answer": "SUT failed to answer this question."
                }

        # Check if the error file is empty
        if error_fp is not None:
            if os.path.getsize(error_fp) > 0:
                print(f"ERROR: {self.name}: ** ERRORS ** found in {error_fp}. Skipping JSON update.")
                return self._format_answer_on_fail(answer, f"** ERRORS ** found during execution in {error_fp}")

        # Load the output file
        try: 
            with open(output_fp, 'r') as f:
                output = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: {self.name}: ** ERRORS ** decoding output JSON: {e}")
            return self._format_answer_on_fail(answer, f"** ERRORS ** decoding output in {output_fp}")

        # Fill in the JSON answer with the execution result
        for step in answer:
            id = "main-task"
            if id in output:
                step['answer'] = output[id]
            else: step['answer'] = "Warning: No answer found in the Python pipeline."
            # Update the subtasks with the output
            subtasks = step.get('subtasks', [])
            for subtask in subtasks:
                subtask_id = subtask['id']
                if subtask_id in output:
                    subtask['answer'] = output[subtask_id]
                else: subtask['answer'] = "Warning: No answer found in the Python pipeline."
        
        overall_answer = {"system_subtasks_responses": []}
        for step in answer:
            if step["id"] == "main-task":
                overall_answer |= step
            else:
                overall_answer["system_subtasks_responses"].append(step)

        # Save the updated JSON answer
        with open(json_fp, 'w') as f:
            # Clean NaN values to null for strict JSON compliance
            overall_answer = clean_nan(overall_answer)
            json.dump(overall_answer, f, indent=4)
        if self.verbose:
            print(f"{self.name}: Updated JSON answer saved to {json_fp}")
        return overall_answer
    
    @typechecked
    def run_one_shot(self, query:str, query_id:str) -> Dict[str, str | Dict | List]:
        """
        This function demonstrates a simple one-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(query_id)
            
        # Generate the prompt
        prompt = self.generate_prompt(query)
        if self.debug:
            print(f"{self.name}: Prompt:", prompt)

        # Get the model's response
        messages=[
            {"role": "system", "content": "You are an experienced data scientist."},
            {"role": "user", "content": prompt}
        ]
        response = call_gpt(messages)
        if self.debug:
            print(f"{self.name}: Response:", response)

        # Process the response
        json_fp, code_fp = self.extract_response(response, try_number=0)

        # Execute the code (if necessary)
        output_fp, error_fp = self.execute_code(code_fp, try_number=0)
        # print("Execution Result:", result)

        # Fill in JSON response with the execution result
        answer = self.process_response(json_fp, output_fp, error_fp)

        return answer
    
    @typechecked
    def run_few_shot(self, query: str, query_id: str) -> Dict[str, str | Dict | List]:
        """
        This function demonstrates a simple few-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(query_id)

        # Generate the prompt
        prompt = self.generate_prompt(query)
        if self.debug:
            print(f"{self.name}: Prompt:", prompt)

        messages=[
            {"role": "system", "content": "You are an experienced data scientist."},
        ]

        answer = None

        for try_number in range(5):
            messages.append({"role": "user", "content": prompt})
            # Get the model's response
            response = call_gpt(messages)
            if self.debug:
                print(f"{self.name}: Response:", response)
            messages.append({"role": "assistant", "content": response})

            # Process the response
            if try_number == 0:
                json_fp, code_fp = self.extract_response(response, try_number)
            else: _, code_fp = self.extract_response(response, try_number)

            # Execute the code (if necessary)
            output_fp, error_fp = self.execute_code(code_fp, try_number)
            # print("Execution Result:", result)

            if os.path.getsize(error_fp) > 0:
                prompt = self.generate_error_handling_prompt(code_fp, error_fp)
            else:
                # Fill in JSON response with the execution result
                answer = self.process_response(json_fp, output_fp, error_fp=None)
                break
        
        if answer is None:
            answer = {"id": "main-task", "answer": "Pipeline not successful after 5 tries."}
        return answer
    
    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """
        Process the dataset located in the specified directory.
        The dataset can contain files in various formats (e.g., PDF, CSV).
        """
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for file in os.listdir(dataset_directory):
            if file.endswith(".csv"):
                try:
                    self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file), engine='python', on_bad_lines='warn')
                except UnicodeDecodeError:
                    self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file), engine='python', on_bad_lines='warn', encoding='latin-1')
    
    @typechecked
    def serve_query(self, query: str, query_id:str="default_name-0") -> Dict:
        """
        Serve a query using the LLM.
        The query should be in natural language, and the response can be in either natural language or JSON format.
        """
        # TODO: Implement the logic to handle different types of queries
        if self.variance == "one_shot":
            results = self.run_one_shot(query, query_id)
        elif self.variance == "few_shot":
            results = self.run_few_shot(query, query_id)
        return results

def main():
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    questions = [
        {
            "id": "environment-easy-1",
            "query": "In 2018, how many bacterial exceedances were observed in freshwater beaches?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-medium-1",
            "query": "For the freshwater beaches, what is the difference between the percentage exceedance rate in 2023 and the historic average from 2002 to 2022?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-hard-1",
            "query": "For the marine beaches, what is the Pearson product-moment correlation from 2002 to 2023 between the rainfall amount in inches during the months June, July, August, and September and the percentage exceedance rate?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        }
    ]
    
    # Process each question
    out_dir = os.path.join(current_dir, "../testresults/run2")
    baseline_llm = BaselineLLMSystem(model="gpt-4o", output_dir=out_dir, variance="few_shot", verbose=True)
    # Process the dataset
    baseline_llm.process_dataset(questions[0]["dataset_directory"])
    for question in questions:
        print(f"Processing question: {question['id']}")
        # For debugging purposes, also input question.id
        response = baseline_llm.serve_query(question["query"], question["id"])

if __name__ == "__main__":
    main()