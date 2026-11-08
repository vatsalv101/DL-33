import sys
import importlib
import torch
import torch.nn as nn
import torch.optim as optim

def run_tests():
    # Allow testing reference solutions or student model
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.17_detr.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    box_cxcywh_to_xyxy = getattr(mod, "box_cxcywh_to_xyxy")
    box_xyxy_to_cxcywh = getattr(mod, "box_xyxy_to_cxcywh")
    box_iou = getattr(mod, "box_iou")
    generalized_box_iou = getattr(mod, "generalized_box_iou")
    PositionEmbeddingSine = getattr(mod, "PositionEmbeddingSine")
    HungarianMatcher = getattr(mod, "HungarianMatcher")
    SetCriterion = getattr(mod, "SetCriterion")
    DETR = getattr(mod, "DETR")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: Box conversions and GIoU
    # ---------------------------------------------------------
    print("Running Test 1: Box conversions & GIoU...", end=" ")
    try:
        # Box: center at (0.5, 0.5), width 0.4, height 0.6 -> [0.3, 0.2, 0.7, 0.8]
        cxcywh = torch.tensor([[0.5, 0.5, 0.4, 0.6], [0.2, 0.2, 0.2, 0.2]])
        xyxy = box_cxcywh_to_xyxy(cxcywh)
        expected_xyxy = torch.tensor([[0.3, 0.2, 0.7, 0.8], [0.1, 0.1, 0.3, 0.3]])
        assert torch.allclose(xyxy, expected_xyxy, atol=1e-5), f"Expected {expected_xyxy}, got {xyxy}"

        # Roundtrip
        cxcywh_rec = box_xyxy_to_cxcywh(xyxy)
        assert torch.allclose(cxcywh, cxcywh_rec, atol=1e-5), f"Roundtrip failed: {cxcywh} vs {cxcywh_rec}"

        # IoU identical boxes
        b1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        b2 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
        iou, _ = box_iou(b1, b2)
        assert abs(iou[0, 0].item() - 1.0) < 1e-4, f"Identical IoU should be 1.0, got {iou[0, 0].item()}"
        assert abs(iou[0, 1].item()) < 1e-4, f"Disjoint IoU should be 0.0, got {iou[0, 1].item()}"

        # GIoU non-overlapping boxes should be strictly negative
        giou = generalized_box_iou(b1, b2)
        assert abs(giou[0, 0].item() - 1.0) < 1e-4, f"GIoU identical should be 1.0, got {giou[0, 0].item()}"
        assert giou[0, 1].item() < 0.0, f"GIoU disjoint should be < 0, got {giou[0, 1].item()}"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: 2D Positional Embedding
    # ---------------------------------------------------------
    print("Running Test 2: 2D Sinusoidal Position Embedding...", end=" ")
    try:
        pos_enc = PositionEmbeddingSine(num_pos_feats=32)
        dummy_feat = torch.zeros(2, 64, 8, 8)
        pos = pos_enc(dummy_feat)
        assert pos.shape == (2, 64, 8, 8), f"Expected shape (2, 64, 8, 8), got {pos.shape}"
        assert not torch.allclose(pos[:, :, 0, 0], pos[:, :, 1, 1]), "Positional embeddings across different coords should differ"
        assert pos.abs().max() <= 1.0 + 1e-5, "Positional embeddings should be bounded in [-1, 1]"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: Hungarian Matcher
    # ---------------------------------------------------------
    print("Running Test 3: Hungarian Matcher Assignment...", end=" ")
    try:
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        # Synthetic outputs: 1 batch, 4 queries, 2 classes (+ bg)
        # Query 1 is close to target 0; Query 3 is close to target 1
        pred_logits = torch.tensor([
            [[5.0, -5.0, -5.0],   # query 0: class 0
             [-5.0, 5.0, -5.0],   # query 1: class 1
             [-5.0, -5.0, 5.0],   # query 2: bg
             [5.0, -5.0, -5.0]]   # query 3: class 0
        ])
        pred_boxes = torch.tensor([
            [[0.1, 0.1, 0.2, 0.2],  # query 0
             [0.8, 0.8, 0.2, 0.2],  # query 1
             [0.5, 0.5, 0.2, 0.2],  # query 2
             [0.2, 0.2, 0.2, 0.2]]  # query 3
        ])
        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        # Targets: target 0 is class 1 at (0.8, 0.8, 0.2, 0.2); target 1 is class 0 at (0.1, 0.1, 0.2, 0.2)
        targets = [{
            "labels": torch.tensor([1, 0], dtype=torch.int64),
            "boxes": torch.tensor([[0.8, 0.8, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2]])
        }]

        indices = matcher(outputs, targets)
        assert len(indices) == 1
        src_idx, tgt_idx = indices[0]
        # Query 1 should match target 0 (class 1), Query 0 should match target 1 (class 0)
        mapping = dict(zip(tgt_idx.tolist(), src_idx.tolist()))
        assert mapping[0] == 1, f"Expected target 0 to match query 1, got query {mapping[0]}"
        assert mapping[1] == 0, f"Expected target 1 to match query 0, got query {mapping[1]}"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: DETR Forward Pass and SetCriterion Loss
    # ---------------------------------------------------------
    print("Running Test 4: DETR Forward Pass & Loss...", end=" ")
    try:
        num_classes = 3
        num_queries = 6
        model = DETR(num_classes=num_classes, num_queries=num_queries, hidden_dim=64, nheads=2, num_encoder_layers=2, num_decoder_layers=2)
        images = torch.randn(2, 3, 32, 32)
        outputs = model(images)

        assert "pred_logits" in outputs and "pred_boxes" in outputs
        assert outputs["pred_logits"].shape == (2, num_queries, num_classes + 1), f"Unexpected logits shape: {outputs['pred_logits'].shape}"
        assert outputs["pred_boxes"].shape == (2, num_queries, 4), f"Unexpected boxes shape: {outputs['pred_boxes'].shape}"
        assert (outputs["pred_boxes"] >= 0.0).all() and (outputs["pred_boxes"] <= 1.0).all(), "Bounding box coordinates should be in [0, 1]"

        # Compute criterion
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0})

        targets = [
            {"labels": torch.tensor([0, 2]), "boxes": torch.tensor([[0.2, 0.2, 0.3, 0.3], [0.6, 0.6, 0.2, 0.2]])},
            {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}
        ]

        loss_dict = criterion(outputs, targets)
        assert "loss_ce" in loss_dict and "loss_bbox" in loss_dict and "loss_giou" in loss_dict
        total_loss = loss_dict["loss_ce"] + 5.0 * loss_dict["loss_bbox"] + 2.0 * loss_dict["loss_giou"]
        total_loss.backward()

        assert model.class_embed.weight.grad is not None, "class_embed weight grad should not be None"
        assert model.bbox_embed.layers[0].weight.grad is not None, "bbox_embed grad should not be None"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 5: Single-Batch Overfit
    # ---------------------------------------------------------
    print("Running Test 5: Single-Batch Overfit...", end=" ")
    try:
        torch.manual_seed(42)
        num_classes = 2
        num_queries = 4
        model = DETR(num_classes=num_classes, num_queries=num_queries, hidden_dim=64, nheads=2, num_encoder_layers=2, num_decoder_layers=2)
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0})

        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

        images = torch.randn(1, 3, 32, 32)
        targets = [{
            "labels": torch.tensor([0, 1], dtype=torch.int64),
            "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.3, 0.3]])
        }]

        initial_loss = None
        for step in range(80):
            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss_ce"] + 5.0 * loss_dict["loss_bbox"] + 2.0 * loss_dict["loss_giou"]
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.4 or final_loss < 0.5, f"Overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll DETR unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
