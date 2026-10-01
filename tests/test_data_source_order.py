# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from systems.data_source_utils import expand_data_sources


class DataSourceOrderTest(unittest.TestCase):
    def test_deduplication_preserves_the_supplied_file_order(self):
        paths = [f"file_{i}.csv" for i in range(20)]
        result = expand_data_sources(paths + paths[:3], "data", paths)
        self.assertEqual(result, [f"data/{p}" for p in paths])


if __name__ == "__main__":
    unittest.main()
