from __future__ import annotations

import json
import unittest

from src.decision import decide_consensus
from src.graph_fusion import fuse_mermaid_outputs
from src.normalizer import normalize_model_output
from src.schema import CaptionStructured, ModelOutput, ParsedLabel, StructuredLabel
from src.validators.validation import ValidationResult


def _flowchart_graph(
    nodes: list[dict],
    edges: list[dict],
    graph_source: str = "model",
    weak_candidate: bool = False,
) -> dict:
    return {
        "node_order_rule": "top_to_bottom_left_to_right",
        "nodes": nodes,
        "edges": edges,
        "graph_source": graph_source,
        "weak_candidate": weak_candidate,
    }


def _label(
    nodes: list[dict],
    edges: list[dict],
    mermaid: str | None = None,
    graph_source: str = "model",
    weak_candidate: bool = False,
) -> ParsedLabel:
    visible_text = [str(node.get("text", "") or "").strip() for node in nodes if str(node.get("text", "") or "").strip()]
    return ParsedLabel(
        image_type="flowchart",
        caption="流程图",
        caption_structured=CaptionStructured(
            brief="流程图",
            visual_type="flowchart",
            key_visible_text=visible_text[:10],
        ),
        structured_label=StructuredLabel(
            kind="mermaid" if mermaid else "text",
            content=mermaid or "",
            format="mermaid" if mermaid else "plain_text",
        ),
        flowchart_graph=_flowchart_graph(
            nodes=nodes,
            edges=edges,
            graph_source=graph_source,
            weak_candidate=weak_candidate,
        ),
        visible_text=visible_text,
    )


def _output(model_name: str, raw_text: str = "{}") -> ModelOutput:
    return ModelOutput(
        image_id="img-1",
        model_name=model_name,
        success=True,
        raw_text=raw_text,
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
    def test_same_text_different_visual_nodes_must_not_merge(self) -> None:
        nodes = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.1, 0.1, 0.2, 0.2], "shape": "rectangle", "text": "随访"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.3, 0.2, 0.4], "shape": "rectangle", "text": "检查A"},
            {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 1, "bbox_hint": [0.1, 0.5, 0.2, 0.6], "shape": "rectangle", "text": "随访"},
            {"node_id": "N004", "order_index": 4, "row_index": 4, "col_index": 1, "bbox_hint": [0.1, 0.7, 0.2, 0.8], "shape": "rectangle", "text": "检查B"},
        ]
        edges = [
            {"source": "N001", "target": "N002", "label": ""},
            {"source": "N003", "target": "N004", "label": ""},
        ]
        labels = [_label(nodes, edges), _label(nodes, edges)]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["随访", "检查A", "检查B"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([node.fused_id for node in result.nodes], ["N001", "N002", "N003", "N004"])
        self.assertEqual(sum(1 for node in result.nodes if node.representative_text == "随访"), 2)
        self.assertEqual(len(result.nodes), 4)

    def test_identical_text_with_different_node_ids_stays_as_two_nodes(self) -> None:
        nodes = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.1, 0.1, 0.2, 0.2], "shape": "rectangle", "text": "随访"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.3, 0.2, 0.4], "shape": "rectangle", "text": "随访"},
        ]
        labels = [_label(nodes, []), _label(nodes, [])]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["随访"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([node.fused_id for node in result.nodes], ["N001", "N002"])
        self.assertEqual(len(result.nodes), 2)

    def test_inconsistent_node_count_goes_to_review_without_text_backfill(self) -> None:
        nodes_a = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.1, 0.1, 0.2, 0.2], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.25, 0.2, 0.35], "shape": "rectangle", "text": "随访"},
            {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 1, "bbox_hint": [0.1, 0.4, 0.2, 0.5], "shape": "rectangle", "text": "复查"},
            {"node_id": "N004", "order_index": 4, "row_index": 4, "col_index": 1, "bbox_hint": [0.1, 0.55, 0.2, 0.65], "shape": "rectangle", "text": "MDT讨论"},
            {"node_id": "N005", "order_index": 5, "row_index": 5, "col_index": 1, "bbox_hint": [0.1, 0.7, 0.2, 0.8], "shape": "rectangle", "text": "结束"},
        ]
        nodes_b = nodes_a[:-1]
        edges_a = [{"source": "N001", "target": "N002", "label": ""}, {"source": "N004", "target": "N005", "label": ""}]
        edges_b = [{"source": "N001", "target": "N002", "label": ""}]
        labels = [_label(nodes_a, edges_a), _label(nodes_b, edges_b)]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "随访", "复查", "MDT讨论", "结束"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn(result.fusion_status, {"ambiguous", "partial"})
        self.assertGreater(result.inconsistent_node_count, 0)
        self.assertNotIn("N005", [node.fused_id for node in result.nodes])

        consensus = decide_consensus(
            image_id="img-1",
            labels=labels,
            model_outputs=outputs,
            score_result=_score_result(),
            validation_result=_validation_result(),
            graph_fusion_result=result,
        )
        self.assertEqual(consensus.decision, "review")
        self.assertIn("inconsistent_node_count", consensus.escalation_reasons)

    def test_same_visual_node_with_text_difference_keeps_single_node(self) -> None:
        nodes_a = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.1, 0.1, 0.2, 0.2], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.3, 0.3, 0.4], "shape": "diamond", "text": "是否可以进行增强CT?"},
            {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 1, "bbox_hint": [0.1, 0.5, 0.2, 0.6], "shape": "rectangle", "text": "结束"},
        ]
        nodes_b = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.1, 0.1, 0.2, 0.2], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.11, 0.31, 0.31, 0.41], "shape": "diamond", "text": "能否增强CT?"},
            {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 1, "bbox_hint": [0.1, 0.5, 0.2, 0.6], "shape": "rectangle", "text": "结束"},
        ]
        edges = [{"source": "N001", "target": "N002", "label": ""}, {"source": "N002", "target": "N003", "label": "是"}]
        labels = [_label(nodes_a, edges), _label(nodes_b, edges)]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "结束", "是否可以进行增强CT?", "能否增强CT?"])

        self.assertIsNotNone(result)
        assert result is not None
        node = next(item for item in result.nodes if item.fused_id == "N002")
        self.assertEqual(node.support_count, 2)
        self.assertLess(node.text_consistency, 1.0)
        self.assertIn(node.representative_text, {"是否可以进行增强CT?", "能否增强CT?"})
        self.assertEqual(len(result.nodes), 3)

    def test_visual_reindex_aligns_models_when_node_ids_are_swapped(self) -> None:
        nodes_a = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.4, 0.05, 0.6, 0.1], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.4, 0.15, 0.6, 0.2], "shape": "rectangle", "text": "判断"},
            {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 1, "bbox_hint": [0.2, 0.3, 0.4, 0.35], "shape": "rectangle", "text": "左分支"},
            {"node_id": "N004", "order_index": 4, "row_index": 3, "col_index": 2, "bbox_hint": [0.6, 0.3, 0.8, 0.35], "shape": "rectangle", "text": "右分支"},
        ]
        nodes_b = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.42, 0.05, 0.62, 0.1], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.42, 0.15, 0.62, 0.2], "shape": "rectangle", "text": "判断"},
            {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 2, "bbox_hint": [0.62, 0.3, 0.82, 0.35], "shape": "rectangle", "text": "右分支"},
            {"node_id": "N004", "order_index": 4, "row_index": 3, "col_index": 1, "bbox_hint": [0.18, 0.3, 0.38, 0.35], "shape": "rectangle", "text": "左分支"},
        ]
        edges_a = [
            {"source": "N001", "target": "N002", "label": ""},
            {"source": "N002", "target": "N003", "label": "否"},
            {"source": "N002", "target": "N004", "label": "是"},
        ]
        edges_b = [
            {"source": "N001", "target": "N002", "label": ""},
            {"source": "N002", "target": "N004", "label": "否"},
            {"source": "N002", "target": "N003", "label": "是"},
        ]
        labels = [_label(nodes_a, edges_a), _label(nodes_b, edges_b)]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "判断", "左分支", "右分支"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.fusion_status, "fused")
        self.assertEqual(len(result.edges), 3)
        self.assertTrue(all(edge.support_count == 2 for edge in result.edges))
        self.assertFalse(any(error.startswith("node_position_conflict:") for error in result.node_alignment_errors))

    def test_three_model_two_of_three_edges_are_kept(self) -> None:
        nodes = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.4, 0.05, 0.6, 0.1], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.4, 0.15, 0.6, 0.2], "shape": "rectangle", "text": "判断"},
            {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 1, "bbox_hint": [0.2, 0.3, 0.4, 0.35], "shape": "rectangle", "text": "左"},
            {"node_id": "N004", "order_index": 4, "row_index": 3, "col_index": 2, "bbox_hint": [0.6, 0.3, 0.8, 0.35], "shape": "rectangle", "text": "右"},
        ]
        labels = [
            _label(nodes, [{"source": "N001", "target": "N002", "label": ""}, {"source": "N002", "target": "N003", "label": "否"}]),
            _label(nodes, [{"source": "N001", "target": "N002", "label": ""}, {"source": "N002", "target": "N003", "label": "否"}]),
            _label(nodes, [{"source": "N001", "target": "N002", "label": ""}, {"source": "N002", "target": "N004", "label": "是"}]),
        ]
        outputs = [_output("m1"), _output("m2"), _output("m3")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "判断", "左", "右"])

        self.assertIsNotNone(result)
        assert result is not None
        node_text_by_id = {node.fused_id: node.representative_text for node in result.nodes}
        self.assertTrue(
            any(
                node_text_by_id.get(edge.source) == "判断" and node_text_by_id.get(edge.target) == "左"
                for edge in result.edges
            )
        )
        self.assertFalse(
            any(
                node_text_by_id.get(item["source"]) == "判断" and node_text_by_id.get(item["target"]) == "左"
                for item in result.low_support_edges
            )
        )

    def test_filtered_low_support_edges_do_not_block_acceptance(self) -> None:
        nodes = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.45, 0.05, 0.55, 0.1], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.15, 0.15, 0.35, 0.25], "shape": "rectangle", "text": "A"},
            {"node_id": "N003", "order_index": 3, "row_index": 2, "col_index": 2, "bbox_hint": [0.38, 0.15, 0.58, 0.25], "shape": "rectangle", "text": "B"},
            {"node_id": "N004", "order_index": 4, "row_index": 2, "col_index": 3, "bbox_hint": [0.6, 0.15, 0.8, 0.25], "shape": "rectangle", "text": "C"},
            {"node_id": "N005", "order_index": 5, "row_index": 2, "col_index": 4, "bbox_hint": [0.82, 0.15, 0.95, 0.25], "shape": "rectangle", "text": "D"},
            {"node_id": "N006", "order_index": 6, "row_index": 3, "col_index": 1, "bbox_hint": [0.15, 0.3, 0.25, 0.35], "shape": "rectangle", "text": "FOLFIRINOX"},
            {"node_id": "N007", "order_index": 7, "row_index": 3, "col_index": 2, "bbox_hint": [0.35, 0.3, 0.45, 0.35], "shape": "rectangle", "text": "GN"},
            {"node_id": "N008", "order_index": 8, "row_index": 3, "col_index": 3, "bbox_hint": [0.55, 0.3, 0.65, 0.35], "shape": "rectangle", "text": "GEM"},
            {"node_id": "N009", "order_index": 9, "row_index": 3, "col_index": 4, "bbox_hint": [0.75, 0.3, 0.85, 0.35], "shape": "rectangle", "text": "支持治疗"},
            {"node_id": "N010", "order_index": 10, "row_index": 4, "col_index": 1, "bbox_hint": [0.15, 0.45, 0.25, 0.55], "shape": "rectangle", "text": "GN 或吉西他滨"},
            {"node_id": "N011", "order_index": 11, "row_index": 4, "col_index": 2, "bbox_hint": [0.35, 0.45, 0.65, 0.55], "shape": "rectangle", "text": "二线治疗"},
        ]
        edges_a = [
            {"source": "N001", "target": "N002", "label": ""},
            {"source": "N001", "target": "N003", "label": ""},
            {"source": "N001", "target": "N004", "label": ""},
            {"source": "N001", "target": "N005", "label": ""},
            {"source": "N002", "target": "N006", "label": ""},
            {"source": "N003", "target": "N007", "label": ""},
            {"source": "N004", "target": "N008", "label": ""},
            {"source": "N005", "target": "N009", "label": ""},
            {"source": "N006", "target": "N010", "label": ""},
            {"source": "N007", "target": "N011", "label": ""},
            {"source": "N008", "target": "N011", "label": ""},
            {"source": "N009", "target": "N011", "label": ""},
        ]
        edges_b = list(edges_a)
        edges_c = [
            {"source": "N001", "target": "N002", "label": ""},
            {"source": "N001", "target": "N003", "label": ""},
            {"source": "N001", "target": "N004", "label": ""},
            {"source": "N001", "target": "N005", "label": ""},
            {"source": "N002", "target": "N006", "label": ""},
            {"source": "N003", "target": "N007", "label": ""},
            {"source": "N004", "target": "N008", "label": ""},
            {"source": "N005", "target": "N009", "label": ""},
            {"source": "N006", "target": "N010", "label": ""},
            {"source": "N007", "target": "N010", "label": ""},
            {"source": "N010", "target": "N011", "label": ""},
            {"source": "N008", "target": "N011", "label": ""},
            {"source": "N009", "target": "N011", "label": ""},
        ]
        labels = [_label(nodes, edges_a), _label(nodes, edges_b), _label(nodes, edges_c)]
        outputs = [_output("m1"), _output("m2"), _output("m3")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "A", "B", "C", "D", "FOLFIRINOX", "GN", "GEM", "支持治疗", "GN 或吉西他滨", "二线治疗"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.fusion_status, "fused")
        self.assertEqual(len(result.low_support_edges), 2)

        consensus = decide_consensus(
            image_id="img-1",
            labels=labels,
            model_outputs=outputs,
            score_result=_score_result(),
            validation_result=_validation_result(),
            graph_fusion_result=result,
        )
        self.assertEqual(consensus.decision, "accepted")
        self.assertNotIn("low_support_edges", consensus.escalation_reasons)
        self.assertNotIn("partial_visual_graph_alignment", consensus.escalation_reasons)

    def test_depth_adjusted_reindex_stabilizes_leaf_nodes(self) -> None:
        nodes_a = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.4, 0.05, 0.6, 0.1], "shape": "rectangle", "text": "根"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.18, 0.3, 0.28], "shape": "rectangle", "text": "左分支"},
            {"node_id": "N003", "order_index": 3, "row_index": 2, "col_index": 2, "bbox_hint": [0.4, 0.18, 0.6, 0.28], "shape": "rectangle", "text": "中分支"},
            {"node_id": "N004", "order_index": 4, "row_index": 2, "col_index": 3, "bbox_hint": [0.7, 0.18, 0.9, 0.28], "shape": "rectangle", "text": "右分支"},
            {"node_id": "N005", "order_index": 5, "row_index": 3, "col_index": 1, "bbox_hint": [0.1, 0.35, 0.3, 0.45], "shape": "rectangle", "text": "左中间"},
            {"node_id": "N006", "order_index": 6, "row_index": 4, "col_index": 1, "bbox_hint": [0.1, 0.55, 0.3, 0.65], "shape": "rectangle", "text": "左终点"},
            {"node_id": "N007", "order_index": 7, "row_index": 3, "col_index": 2, "bbox_hint": [0.4, 0.35, 0.6, 0.45], "shape": "rectangle", "text": "中终点"},
            {"node_id": "N008", "order_index": 8, "row_index": 3, "col_index": 3, "bbox_hint": [0.7, 0.35, 0.9, 0.45], "shape": "rectangle", "text": "右终点"},
        ]
        nodes_b = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.4, 0.05, 0.6, 0.1], "shape": "rectangle", "text": "根"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.18, 0.3, 0.28], "shape": "rectangle", "text": "左分支"},
            {"node_id": "N003", "order_index": 3, "row_index": 2, "col_index": 2, "bbox_hint": [0.4, 0.18, 0.6, 0.28], "shape": "rectangle", "text": "中分支"},
            {"node_id": "N004", "order_index": 4, "row_index": 2, "col_index": 3, "bbox_hint": [0.7, 0.18, 0.9, 0.28], "shape": "rectangle", "text": "右分支"},
            {"node_id": "N005", "order_index": 5, "row_index": 3, "col_index": 1, "bbox_hint": [0.1, 0.35, 0.3, 0.45], "shape": "rectangle", "text": "左中间"},
            {"node_id": "N006", "order_index": 6, "row_index": 4, "col_index": 2, "bbox_hint": [0.4, 0.55, 0.6, 0.65], "shape": "rectangle", "text": "中终点"},
            {"node_id": "N007", "order_index": 7, "row_index": 4, "col_index": 3, "bbox_hint": [0.7, 0.55, 0.9, 0.65], "shape": "rectangle", "text": "右终点"},
            {"node_id": "N008", "order_index": 8, "row_index": 4, "col_index": 1, "bbox_hint": [0.1, 0.55, 0.3, 0.65], "shape": "rectangle", "text": "左终点"},
        ]
        edges_a = [
            {"source": "N001", "target": "N002", "label": ""},
            {"source": "N001", "target": "N003", "label": ""},
            {"source": "N001", "target": "N004", "label": ""},
            {"source": "N002", "target": "N005", "label": ""},
            {"source": "N005", "target": "N006", "label": ""},
            {"source": "N003", "target": "N007", "label": ""},
            {"source": "N004", "target": "N008", "label": ""},
        ]
        edges_b = [
            {"source": "N001", "target": "N002", "label": ""},
            {"source": "N001", "target": "N003", "label": ""},
            {"source": "N001", "target": "N004", "label": ""},
            {"source": "N002", "target": "N005", "label": ""},
            {"source": "N005", "target": "N008", "label": ""},
            {"source": "N003", "target": "N006", "label": ""},
            {"source": "N004", "target": "N007", "label": ""},
        ]
        labels = [_label(nodes_a, edges_a), _label(nodes_b, edges_b)]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["根", "左分支", "中分支", "右分支", "左中间", "左终点", "中终点", "右终点"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.fusion_status, "fused")
        left_leaf = next(node for node in result.nodes if node.representative_text == "左终点")
        middle_leaf = next(node for node in result.nodes if node.representative_text == "中终点")
        right_leaf = next(node for node in result.nodes if node.representative_text == "右终点")
        self.assertLess(middle_leaf.order_index, left_leaf.order_index)
        self.assertLess(right_leaf.order_index, left_leaf.order_index)

    def test_edges_are_fused_by_node_id_and_reverse_direction_creates_conflict(self) -> None:
        nodes = [
            {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.1, 0.1, 0.2, 0.2], "shape": "rectangle", "text": "开始"},
            {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.3, 0.2, 0.4], "shape": "rectangle", "text": "结束"},
        ]
        labels_same = [
            _label(nodes, [{"source": "N001", "target": "N002", "label": ""}]),
            _label(nodes, [{"source": "N001", "target": "N002", "label": ""}]),
        ]
        outputs_same = [_output("m1"), _output("m2")]

        result_same = fuse_mermaid_outputs(labels_same, outputs_same, ["开始", "结束"])

        self.assertIsNotNone(result_same)
        assert result_same is not None
        self.assertEqual(len(result_same.edges), 1)
        self.assertEqual(result_same.edges[0].support_count, 2)

        labels_conflict = [
            _label(nodes, [{"source": "N001", "target": "N002", "label": ""}]),
            _label(nodes, [{"source": "N002", "target": "N001", "label": ""}]),
        ]
        outputs_conflict = [_output("m1"), _output("m2")]

        result_conflict = fuse_mermaid_outputs(labels_conflict, outputs_conflict, ["开始", "结束"])

        self.assertIsNotNone(result_conflict)
        assert result_conflict is not None
        self.assertEqual(len(result_conflict.edges), 0)
        self.assertTrue(
            any(error.startswith("edge_direction_conflict:") for error in result_conflict.edge_alignment_errors)
        )

    def test_normalizer_derives_mermaid_fallback_flowchart_graph(self) -> None:
        raw_payload = {
            "image_type": "flowchart",
            "caption": "流程图",
            "caption_structured": {
                "brief": "流程图",
                "visual_type": "flowchart",
                "key_visible_text": ["开始", "结束"],
                "structure_summary": "流程节点与连线。",
                "caption_source": "generated",
                "confidence": "medium",
            },
            "structured_label": {
                "kind": "mermaid",
                "format": "mermaid",
                "content": "flowchart TD\nA[开始] --> B[结束]",
            },
            "visible_text": ["开始", "结束"],
            "warnings": [],
        }
        output = _output("m1", raw_text=json.dumps(raw_payload, ensure_ascii=False))

        normalized_output, normalized_label = normalize_model_output(output)

        self.assertIsNone(normalized_output.error)
        self.assertIsNotNone(normalized_label)
        assert normalized_label is not None
        self.assertIsNotNone(normalized_label.flowchart_graph)
        assert normalized_label.flowchart_graph is not None
        self.assertEqual(normalized_label.flowchart_graph.get("graph_source"), "mermaid_fallback")
        self.assertTrue(normalized_label.flowchart_graph.get("weak_candidate"))

    def test_normalizer_recovers_truncated_json_output(self) -> None:
        raw_payload = {
            "image_type": "flowchart",
            "caption": "流程图",
            "caption_structured": {
                "brief": "流程图",
                "visual_type": "flowchart",
                "key_visible_text": ["开始", "判断", "结束"],
                "structure_summary": "流程节点与连线。",
                "caption_source": "generated",
                "confidence": "medium",
            },
            "structured_label": {
                "kind": "mermaid",
                "format": "mermaid",
                "content": "flowchart TD\nN001[开始] --> N002[判断]\nN002 --> N003[结束]",
            },
            "flowchart_graph": {
                "node_order_rule": "top_to_bottom_left_to_right",
                "nodes": [
                    {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": [0.1, 0.1, 0.2, 0.2], "shape": "rectangle", "text": "开始"},
                    {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": [0.1, 0.3, 0.2, 0.4], "shape": "diamond", "text": "判断"},
                    {"node_id": "N003", "order_index": 3, "row_index": 3, "col_index": 1, "bbox_hint": [0.1, 0.5, 0.2, 0.6], "shape": "rectangle", "text": "结束"},
                ],
                "edges": [
                    {"source": "N001", "target": "N002", "label": ""},
                    {"source": "N002", "target": "N003", "label": "是"},
                ],
            },
            "visible_text": ["开始", "判断", "结束", "是"],
            "warnings": [],
        }
        raw_text = json.dumps(raw_payload, ensure_ascii=False)
        truncated_raw_text = raw_text[:-80]
        output = _output("m1", raw_text=truncated_raw_text)

        normalized_output, normalized_label = normalize_model_output(output)

        self.assertIsNone(normalized_output.error)
        self.assertIsNotNone(normalized_label)
        assert normalized_label is not None
        self.assertIn("recovered_truncated_json_output", normalized_label.warnings)
        self.assertEqual(normalized_label.image_type, "flowchart")
        self.assertIsNotNone(normalized_label.flowchart_graph)

    def test_decision_reviews_mermaid_fallback_graph_fusion(self) -> None:
        labels = [
            _label(
                nodes=[
                    {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": None, "shape": "rectangle", "text": "开始"},
                    {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": None, "shape": "rectangle", "text": "结束"},
                ],
                edges=[{"source": "N001", "target": "N002", "label": ""}],
                mermaid="flowchart TD\nN001[开始] --> N002[结束]",
                graph_source="mermaid_fallback",
                weak_candidate=True,
            ),
            _label(
                nodes=[
                    {"node_id": "N001", "order_index": 1, "row_index": 1, "col_index": 1, "bbox_hint": None, "shape": "rectangle", "text": "开始"},
                    {"node_id": "N002", "order_index": 2, "row_index": 2, "col_index": 1, "bbox_hint": None, "shape": "rectangle", "text": "结束"},
                ],
                edges=[{"source": "N001", "target": "N002", "label": ""}],
                mermaid="flowchart TD\nN001[开始] --> N002[结束]",
                graph_source="mermaid_fallback",
                weak_candidate=True,
            ),
        ]
        outputs = [_output("m1"), _output("m2")]

        result = fuse_mermaid_outputs(labels, outputs, ["开始", "结束"])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.fusion_method, "mermaid_fallback")

        consensus = decide_consensus(
            image_id="img-1",
            labels=labels,
            model_outputs=outputs,
            score_result=_score_result(),
            validation_result=_validation_result(),
            graph_fusion_result=result,
        )
        self.assertEqual(consensus.decision, "review")
        self.assertIn("flowchart_graph_missing_used_mermaid_fallback", consensus.escalation_reasons)


if __name__ == "__main__":
    unittest.main()
