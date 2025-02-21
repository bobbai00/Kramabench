import json

from system.baseline import generator_factory

class Executor:
    def __init__(self, model_name, data_directory, workload_path, task_fixture_directory, results_directory, verbose=False):
        self.model = generator_factory(model_name, verbose)
        with open(workload_path) as f:
            self.workload = json.load(f)
        self.num_queries = len(self.workload)
        self.cur_query_id = 0
    
    def execute_query(self, query):
        cur_query = self.workload[self.cur_query_id]
        model_output = self.model(query)
        response = {}
        response["model_output"] = model_output
        response["subresponses"] = []
        for subquery in cur_query["subqueries"]:
            subresponse = self.execute_query(subquery)
            response["subresponse"].append(subresponse)
        return response
    
    def execute_next_query(self):
        """
        Iterator model
        """
        if self.cur_query_id >= self.num_queries:
            return None
        results = self.execute_query(self.workload[self.cur_query_id])
        self.cur_query_id += 1
        return results
    
    def execute_full_workload():
        pass

class Evaluator:
    def __init__(self, responses, reference_results, metric_name):
        pass

