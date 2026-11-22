import json
import unittest

from OCR_MetaReasoning.evaluation.answer_utils import (
    flatten_json,
    score_json_f1,
    score_prediction,
)


class JsonF1Tests(unittest.TestCase):
    def assert_score(self, prediction, target, expected):
        prediction, target = json.dumps(prediction), json.dumps(target)
        self.assertAlmostEqual(score_json_f1(prediction, target), expected)
        self.assertAlmostEqual(score_json_f1(target, prediction), expected)

    def test_dotted_key_is_distinct_from_nested_keys(self):
        self.assert_score({"a.b": 1}, {"a": {"b": 1}}, 0.0)

    def test_bracketed_key_is_distinct_from_array_index(self):
        self.assert_score({"items[0]": "yes"}, {"items": ["yes"]}, 0.0)

    def test_empty_key_remains_a_path_segment(self):
        self.assert_score({"": {"a": "x"}}, {"a": "x"}, 0.0)
        self.assert_score({"": 1}, 1, 0.0)

    def test_numeric_object_key_is_distinct_from_array_index(self):
        self.assert_score({"items": {"0": "yes"}}, {"items": ["yes"]}, 0.0)

    def test_punctuation_at_multiple_levels(self):
        self.assert_score(
            {"a.b": [{"c[0]": " YES "}]},
            {"a": {"b[0]": {"c": ["yes"]}}},
            0.0,
        )

    def test_partial_overlap_counts_only_the_shared_path(self):
        prediction = {"match": "yes", "a.b": "no"}
        target = {"match": "YES", "a": {"b": "NO"}}
        # One matching leaf among two in each answer.
        self.assert_score(prediction, target, 0.5)

    def test_same_normalized_value_can_appear_at_distinct_paths(self):
        prediction = {"a.b": " YES ", "a": {"b": "yes"}}
        target = {"a": {"b": "YES"}}
        # One matching leaf, precision 1/2 and recall 1.
        self.assert_score(prediction, target, 2 / 3)

    def test_flattened_paths_and_normalized_values(self):
        data = {"items": [{"0": " A ", "": True}], "items[0]": "B", "": None}
        expected = {
            (("items", 0, "0"), "a"),
            (("items", 0, ""), "true"),
            (("items[0]",), "b"),
            (("",), ""),
        }
        self.assertEqual(set(flatten_json(data)), expected)

    def test_equivalent_objects_arrays_and_normalized_leaves(self):
        cases = [
            ({"a": 1}, {"a": 1}),
            ({"a": {"b": 1}}, {"a": {"b": 1}}),
            ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
            ([1, 2], [1, 2]),
            ({"list": [[{"key": "  YES "}]]}, {"list": [[{"key": "yes"}]]}),
            ({"cost": "$1,200.00", "rate": "20.0%"}, {"cost": "$1200", "rate": "20%"}),
        ]
        for prediction, target in cases:
            with self.subTest(prediction=prediction, target=target):
                self.assert_score(prediction, target, 1.0)

    def test_different_values_positions_and_paths(self):
        cases = [
            ({"a": 1}, {"a": 2}),
            ({"items": [1, 2]}, {"items": [2, 1]}),
            ({"a": {"b": 1}}, {"a": {"c": 1}}),
            ({"cost": "$12"}, {"cost": "€12"}),
        ]
        for prediction, target in cases:
            with self.subTest(prediction=prediction, target=target):
                self.assert_score(prediction, target, 0.0)

    def test_missing_and_extra_leaves(self):
        self.assert_score({"a": 1}, {"a": 1, "b": 2}, 2 / 3)

    def test_empty_containers_and_scalars(self):
        cases = [
            ({}, {}, 1.0),
            ([], [], 1.0),
            ({}, [], 1.0),
            ({"empty": []}, {}, 1.0),
            ({}, {"null": None}, 0.0),
            ({"null": None}, {"null": None}, 1.0),
            ({"bool": True}, {"bool": False}, 0.0),
            (7, 7, 1.0),
            (True, True, 1.0),
            (None, None, 1.0),
        ]
        for prediction, target, expected in cases:
            with self.subTest(prediction=prediction, target=target):
                self.assert_score(prediction, target, expected)

    def test_invalid_json_and_supported_wrappers(self):
        target = '{"a": 1}'
        self.assertEqual(score_json_f1('```json\n{"a": 1}\n```', target), 1.0)
        for invalid in ('{"a":', '{"a": 1,}', "", "not JSON"):
            with self.subTest(invalid=invalid):
                self.assertEqual(score_json_f1(invalid, target), 0.0)
                self.assertEqual(score_json_f1(target, invalid), 0.0)

    def test_score_prediction_json_dispatch(self):
        prediction = '{"a.b": 1}'
        target = '{"a": {"b": 1}}'
        for metric, answer_type in (("json_f1", "string"), ("exact_match", "json")):
            with self.subTest(metric=metric, answer_type=answer_type):
                self.assertEqual(score_prediction(prediction, target, metric, answer_type), 0.0)
                self.assertEqual(score_prediction(target, target, metric, answer_type), 1.0)

    def test_score_prediction_non_json_dispatch(self):
        self.assertEqual(score_prediction("1,200.00", "1200", "numeric", "string"), 1.0)
        self.assertEqual(score_prediction("1,200.00", "1200", "exact_match", "float"), 1.0)
        self.assertEqual(score_prediction(" YES ", "yes", "exact_match", "string"), 1.0)
        self.assertEqual(score_prediction("yes", "no", "exact_match", "string"), 0.0)
