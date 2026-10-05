# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

import unittest

from benchmark.metrics import F1, Success


def success(predicted, target):
    return Success()(predicted=predicted, target=target)[0]


class SuccessTest(unittest.TestCase):
    def test_rounds_to_the_target_precision(self):
        self.assertEqual(success("0.476481368", 0.4765), 1)
        self.assertEqual(success("4.6512", 4.65), 1)
        self.assertEqual(success("4.6", 4.65), 0)

    def test_tiny_targets_are_not_matched_by_an_absolute_tolerance(self):
        # Upstream's 1e-6 absolute tolerance passes any answer near 1e-13.
        self.assertEqual(success("7.95e-13", 7.95e-13), 1)
        self.assertEqual(success("8.02e-13", 7.95e-13), 0)
        self.assertEqual(success("3.402e-12", 1.211e-13), 0)

    def test_a_trailing_percent_sign_is_dropped(self):
        self.assertEqual(success("4.65%", 4.65), 1)
        self.assertEqual(success("0.0465", 4.65), 0)

    def test_integer_targets(self):
        self.assertEqual(success("13596", 13596), 1)
        self.assertEqual(success(13596.0, 13596), 1)
        self.assertEqual(success("15387", 13596), 0)

    def test_numbers_written_as_text_compare_as_numbers(self):
        self.assertEqual(success("2020.0", "2020"), 1)

    def test_strings_compare_without_case_or_outer_space(self):
        self.assertEqual(success("atlanta-sandy springs ", "Atlanta-Sandy Springs"), 1)
        self.assertEqual(success("Atlanta", "Washington"), 0)

    def test_a_missing_answer_scores_zero(self):
        self.assertEqual(success(None, 5), 0)
        self.assertEqual(success("N/A", 5), 0)


class F1Test(unittest.TestCase):
    def test_comma_separated_text_is_read_as_a_list(self):
        score = F1()(predicted="Bucks Creek, Pleasant Street", target=["Bucks Creek", "Pleasant Street"])[0]
        self.assertEqual(score, 1.0)

    def test_json_lists_still_parse(self):
        score = F1()(predicted='["California", "Idaho"]', target=["California", "Washington", "Idaho"])[0]
        self.assertAlmostEqual(score, 0.8)


if __name__ == "__main__":
    unittest.main()
