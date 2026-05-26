from __future__ import annotations

import unittest

from src.decision import decide_consensus
from src.graph_fusion import fuse_mermaid_outputs
from src.schema import CaptionStructured, ModelOutput, ParsedLabel, StructuredLabel
from src.validators.validation import ValidationResult


def _label(mermaid: str) -> ParsedLabel:
    return ParsedLabel(
        image_type="flowchart",
        caption="流程图",
        caption_structured=CaptionStructured(
            brief="流程图",
            visual_type="flowchart",
            key_visible_text=["开始", "处理", "结束"],
        ),
        structured_label=StructuredLabel(
            kind="mermaid",
            content=mermaid,
            format="mermaid",
        ),
        visible_text=["开始", "处理", "结束"],
    )


def _output(model_name: str) -> ModelOutput:
    return ModelOutput(
        image_id="img-1",
        model_name=model_name,
        success=True,
        raw_text="{}",
    )


def _score_result() -> dict[str, float | list[str] | str]:
    return {
        "type_agreement": 1.0,
        "caption_agreement": 1.0,
        "structure_agreement": 1.0,
        "overall_score": 1.0,
        "reasons": [],
    }


def _validation_result() -> ValidationResult:
    return ValidationResult(
        image_id="img-1",
        evidence_score=1.0,
        validator_score=1.0,
        hallucination_risk=0.0,
    )


class GraphFusionTests(unittest.TestCase):
    def test_fuses_equivalent_mermaid_with_different_node_ids(self) -> None:
        labels = [
            _label('flowchart TD\nA["开始"] --> B[处理]\nB --> C[结束]'),
            _label('flowchart TD\nX(开始) --> Y("处理")\nY --> Z{结束}'),
        ]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "处理", "结束"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.nodes), 3)
        self.assertTrue(all(node.support_count == 2 for node in result.nodes))
        self.assertEqual(len(result.edges), 2)
        self.assertTrue(all(edge.support_count == 2 for edge in result.edges))
        self.assertIn("flowchart TD", result.mermaid)
        self.assertIn('["开始"]', result.mermaid)

    def test_direction_mismatch_creates_edge_disagreement_warning(self) -> None:
        labels = [
            _label("flowchart TD\nA[开始] --> B[结束]"),
            _label("flowchart TD\nB[结束] --> A[开始]"),
        ]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "结束"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.edges), 2)
        self.assertTrue(
            any("edge_disagreement:" in warning for warning in result.warnings)
        )
        self.assertIn("edge_disagreement", result.critical_errors)

    def test_two_of_three_models_can_keep_edge(self) -> None:
        labels = [
            _label("flowchart TD\nA[开始] --> B[结束]"),
            _label("flowchart TD\nX[开始] --> Y[结束]"),
            _label("flowchart TD\nN1[开始]\nN2[结束]"),
        ]
        outputs = [_output("m1"), _output("m2"), _output("m3")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "结束"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0].support_count, 2)
        self.assertAlmostEqual(result.edges[0].confidence, 2 / 3, places=3)
        self.assertIn("-->", result.mermaid)

    def test_unsupported_node_is_flagged_by_evidence(self) -> None:
        labels = [
            _label("flowchart TD\nA[开始] --> B[虚构步骤]\nB --> C[结束]"),
            _label("flowchart TD\nX[开始] --> Y[虚构步骤]\nY --> Z[结束]"),
        ]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "结束"])

        self.assertIsNotNone(result)
        assert result is not None
        unsupported_nodes = [
            node for node in result.nodes if node.text == "虚构步骤" and not node.evidence_supported
        ]
        self.assertTrue(unsupported_nodes)
        self.assertIn("unsupported_node:虚构步骤", result.warnings)

    def test_multiline_and_semicolon_node_text_does_not_trigger_parse_error(self) -> None:
        labels = [
            _label(
                'flowchart TD\nA["年龄>75岁，ECOG PS 2，体质虚弱\n或mFOLFIRINOX禁忌"] --> B["诱导疗法; FOLFIRINOX±CRT GN±CRT"]\nB --> C[结束]'
            ),
            _label(
                'flowchart TD\nX["年龄>75岁，ECOG PS 2，体质虚弱 或mFOLFIRINOX禁忌"] --> Y["诱导疗法; FOLFIRINOX±CRT GN±CRT"]\nY --> Z[结束]'
            ),
        ]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(
            labels,
            outputs,
            [
                "年龄>75岁，ECOG PS 2，体质虚弱 或mFOLFIRINOX禁忌",
                "诱导疗法 FOLFIRINOX±CRT GN±CRT",
                "结束",
            ],
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(any(warning.startswith("parse_error:") for warning in result.warnings))
        self.assertTrue(any("诱导疗法" in node.text for node in result.nodes))

    def test_decision_reviews_flowchart_when_edge_disagreement_exists(self) -> None:
        labels = [
            _label("flowchart TD\nA[开始] --> B[结束]"),
            _label("flowchart TD\nB[结束] --> A[开始]"),
        ]
        outputs = [_output("m1"), _output("m2")]
        graph_result = fuse_mermaid_outputs(labels, outputs, ["开始", "结束"])

        self.assertIsNotNone(graph_result)
        assert graph_result is not None
        consensus = decide_consensus(
            image_id="img-1",
            labels=labels,
            model_outputs=outputs,
            score_result=_score_result(),
            validation_result=_validation_result(),
            graph_fusion_result=graph_result,
        )

        self.assertEqual(consensus.decision, "review")
        self.assertIn("edge_disagreement", consensus.escalation_reasons)

    def test_decision_reviews_flowchart_when_parse_errors_exist(self) -> None:
        labels = [
            _label("flowchart TD\nA[开始] -->"),
            _label("flowchart TD\nX[开始] --> Y[结束]"),
        ]
        outputs = [_output("m1"), _output("m2")]
        graph_result = fuse_mermaid_outputs(labels, outputs, ["开始", "结束"])

        self.assertIsNotNone(graph_result)
        assert graph_result is not None
        self.assertIn("mermaid_parse_errors_present", graph_result.critical_errors)
        consensus = decide_consensus(
            image_id="img-1",
            labels=labels,
            model_outputs=outputs,
            score_result=_score_result(),
            validation_result=_validation_result(),
            graph_fusion_result=graph_result,
        )

        self.assertEqual(consensus.decision, "review")
        self.assertIn("graph_parse_errors", consensus.escalation_reasons)

    def test_decision_reviews_duplicate_node_texts_in_same_graph(self) -> None:
        labels = [
            _label(
                "flowchart TD\nA[局部进展期] --> B[是否符合临床试验入选标准?]\n"
                "C[临界可切除PC] --> D[是否符合临床试验入选标准?]"
            ),
            _label(
                "flowchart TD\nX[局部进展期] --> Y[是否符合临床试验入选标准?]\n"
                "Z[临界可切除PC] --> W[是否符合临床试验入选标准?]"
            ),
        ]
        outputs = [_output("m1"), _output("m2")]
        graph_result = fuse_mermaid_outputs(
            labels,
            outputs,
            ["局部进展期", "临界可切除PC", "是否符合临床试验入选标准?"],
        )

        self.assertIsNotNone(graph_result)
        assert graph_result is not None
        self.assertIn("duplicate_node_texts_present", graph_result.critical_errors)
        consensus = decide_consensus(
            image_id="img-1",
            labels=labels,
            model_outputs=outputs,
            score_result=_score_result(),
            validation_result=_validation_result(),
            graph_fusion_result=graph_result,
        )

        self.assertEqual(consensus.decision, "review")
        self.assertIn("duplicate_node_texts", consensus.escalation_reasons)


if __name__ == "__main__":
    unittest.main()
