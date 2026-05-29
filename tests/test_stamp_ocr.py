from __future__ import annotations

import json
import unittest

from src.decision import decide_consensus
from src.normalizer import normalize_model_output
from src.schema import CaptionStructured, ModelOutput, OcrRegion, ParsedLabel, StructuredLabel
from src.scorer import score_consensus
from src.validators import validate_labels
from src.validators.validation import ValidationResult
from src.writer import build_final_label


def _doc_label(*regions: OcrRegion) -> ParsedLabel:
    region_texts = [region.text for region in regions if region.text.strip()]
    return ParsedLabel(
        image_type="document",
        caption="文档",
        caption_structured=CaptionStructured(
            brief="文档",
            visual_type="document",
            key_visible_text=region_texts[:10],
        ),
        structured_label=StructuredLabel(kind="text", content="\n".join(region_texts), format="plain_text"),
        visible_text=region_texts,
        ocr_regions=list(regions),
    )


def _output(model_name: str, raw_text: str = "{}") -> ModelOutput:
    return ModelOutput(
        image_id="img-ocr-1",
        model_name=model_name,
        success=True,
        raw_text=raw_text,
    )


def _validation_result() -> ValidationResult:
    return ValidationResult(
        image_id="img-ocr-1",
        evidence_score=1.0,
        validator_score=1.0,
        hallucination_risk=0.0,
    )


class StampOcrTests(unittest.TestCase):
    def test_normalizer_preserves_ocr_regions_and_role_aliases(self) -> None:
        payload = {
            "image_type": "document",
            "caption": "盖章文档",
            "caption_structured": {
                "brief": "盖章文档",
                "visual_type": "document",
                "key_visible_text": ["上海市第一人民医院", "内部资料"],
            },
            "structured_label": {
                "kind": "text",
                "content": "上海市第一人民医院",
                "format": "plain_text",
            },
            "ocr_regions": [
                {"role": "印章", "text": "上海市第一人民医院", "bbox_hint": [0.1, 0.6, 0.3, 0.9], "confidence": "high"},
                {"role": "水印", "text": "内部资料", "bbox_hint": [0.4, 0.2, 0.8, 0.5], "confidence": "medium"},
                {"role": "底部文字", "text": "第1页", "bbox_hint": [0.2, 0.92, 0.4, 0.98], "confidence": "low"},
                {"role": "unknown-role", "text": "其他", "bbox_hint": None, "confidence": "weird"},
                {"role": "seal", "text": "   ", "bbox_hint": [0.0, 0.0, 0.1, 0.1], "confidence": "high"},
            ],
            "visible_text": ["上海市第一人民医院", "内部资料", "第1页"],
        }
        model_output, normalized = normalize_model_output(
            _output("m1", raw_text=json.dumps(payload, ensure_ascii=False))
        )

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual([region.role for region in normalized.ocr_regions], ["seal", "watermark", "footer", "other"])
        self.assertEqual(normalized.ocr_regions[0].text, "上海市第一人民医院")
        self.assertIn("ocr_regions[4].role normalized from 'unknown-role' to 'other'", model_output.parsed["warnings"])
        self.assertIn("ocr_regions[5] dropped because text is empty", model_output.parsed["warnings"])

    def test_score_consensus_uses_primary_seal_text_only(self) -> None:
        labels = [
            _doc_label(
                OcrRegion(role="seal", text="上海市第一人民医院", bbox_hint=[0.1, 0.6, 0.3, 0.9]),
                OcrRegion(role="seal", text="4541982082", bbox_hint=[0.2, 0.3, 0.5, 0.4]),
            ),
            _doc_label(
                OcrRegion(role="seal", text="上海市第一人民医院 4541982082", bbox_hint=[0.1, 0.6, 0.3, 0.9]),
            ),
        ]
        score_result = score_consensus("img-ocr-1", labels, [_output("m1"), _output("m2")])

        self.assertTrue(score_result["stamp_mode"])
        self.assertTrue(score_result["has_seal_regions"])
        self.assertEqual(score_result["seal_agreement"], 1.0)
        self.assertEqual(score_result["caption_agreement"], 1.0)
        self.assertEqual(score_result["structure_agreement"], 1.0)

    def test_validate_labels_flags_seal_role_conflict(self) -> None:
        labels = [
            _doc_label(OcrRegion(role="seal", text="上海市第一人民医院", bbox_hint=[0.1, 0.6, 0.3, 0.9])),
            _doc_label(OcrRegion(role="watermark", text="上海市第一人民医院", bbox_hint=[0.4, 0.2, 0.8, 0.5])),
        ]
        validation = validate_labels("img-ocr-1", labels)

        self.assertIn("primary seal text is missing in some model outputs", validation.critical_errors)
        self.assertIn(
            "seal texts conflict with watermark/footer roles across model outputs",
            validation.critical_errors,
        )

    def test_build_final_label_uses_single_caption_for_stamp_mode(self) -> None:
        labels = [
            _doc_label(
                OcrRegion(role="seal", text="上海市第一人民医院", bbox_hint=[0.1, 0.6, 0.3, 0.9], confidence="high"),
                OcrRegion(role="footer", text="第1页", bbox_hint=[0.2, 0.92, 0.4, 0.98]),
            ),
            _doc_label(
                OcrRegion(role="seal", text="上海市第一人民医院 4541982082", bbox_hint=[0.12, 0.61, 0.31, 0.89], confidence="high"),
                OcrRegion(role="footer", text="第1页", bbox_hint=[0.19, 0.91, 0.39, 0.98]),
            ),
            _doc_label(
                OcrRegion(role="seal", text="上海市第二人民医院", bbox_hint=[0.1, 0.6, 0.3, 0.9], confidence="medium"),
            ),
        ]
        final_label = build_final_label(labels, [_output("m1"), _output("m2"), _output("m3")])

        self.assertEqual(final_label, {"caption": "上海市第一人民医院"})

    def test_decision_accepts_matching_primary_seal_name_even_with_debug_differences(self) -> None:
        labels = [
            _doc_label(
                OcrRegion(role="seal", text="上海市第一人民医院"),
                OcrRegion(role="footer", text="第1页"),
            ),
            _doc_label(
                OcrRegion(role="seal", text="上海市第一人民医院 4541982082"),
            ),
            _doc_label(
                OcrRegion(role="seal", text="上海市第一人民医院"),
                OcrRegion(role="other", text="内部资料"),
            ),
        ]
        score_result = score_consensus("img-ocr-1", labels, [_output("m1"), _output("m2"), _output("m3")])
        validation_result = validate_labels("img-ocr-1", labels)
        consensus = decide_consensus(
            image_id="img-ocr-1",
            labels=labels,
            model_outputs=[_output("m1"), _output("m2"), _output("m3")],
            score_result=score_result,
            validation_result=validation_result,
            graph_fusion_result=None,
        )

        self.assertEqual(consensus.decision, "accepted")
        self.assertEqual(consensus.seal_agreement, 1.0)
        self.assertEqual(consensus.structure_agreement, 1.0)

    def test_decision_routes_seal_disagreement_to_review(self) -> None:
        labels = [
            _doc_label(OcrRegion(role="seal", text="上海市第一人民医院")),
            _doc_label(OcrRegion(role="seal", text="上海市第二人民医院")),
        ]
        score_result = score_consensus("img-ocr-1", labels, [_output("m1"), _output("m2")])
        validation_result = validate_labels("img-ocr-1", labels)
        consensus = decide_consensus(
            image_id="img-ocr-1",
            labels=labels,
            model_outputs=[_output("m1"), _output("m2")],
            score_result=score_result,
            validation_result=validation_result,
            graph_fusion_result=None,
        )

        self.assertEqual(consensus.decision, "review")
        self.assertIn("low_seal_agreement", consensus.escalation_reasons)
        self.assertEqual(consensus.seal_agreement, 0.0)
        self.assertIn("primary seal texts are inconsistent across model outputs", consensus.validation_errors)


if __name__ == "__main__":
    unittest.main()
