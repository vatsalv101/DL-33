import sys
import math
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def run_tests():
    module_name = sys.argv[1] if len(sys.argv) > 1 else "papers.22_dino.model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        sys.exit(1)

    DINOHead = getattr(mod, "DINOHead")
    DINOLoss = getattr(mod, "DINOLoss")
    DINO = getattr(mod, "DINO")

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Test 1: DINO Projection Head Contracts
    # ---------------------------------------------------------
    print("Running Test 1: DINO Head Contracts...", end=" ")
    try:
        B, in_dim, out_dim = 4, 64, 128
        head = DINOHead(in_dim=in_dim, out_dim=out_dim, hidden_dim=64, bottleneck_dim=32)
        x = torch.randn(B, in_dim, requires_grad=True)
        out = head(x)
        assert out.shape == (B, out_dim), f"Expected shape {(B, out_dim)}, got {out.shape}"

        loss = out.sum()
        loss.backward()
        assert head.mlp[0].weight.grad is not None, "DINO head first layer should receive grad"
        assert head.last_layer.weight.grad is not None, "DINO head last layer should receive grad"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 2: Centering & Sharpening Mechanics
    # ---------------------------------------------------------
    print("Running Test 2: Centering & Sharpening Mechanics...", end=" ")
    try:
        out_dim = 64
        loss_fn = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

        # Initial center is zero
        assert torch.allclose(loss_fn.center, torch.zeros(1, out_dim)), "Initial center must be 0"

        # Update center with batch
        teacher_batch = torch.full((8, out_dim), 2.0)
        loss_fn.update_center(teacher_batch)
        expected_center = torch.full((1, out_dim), 0.2)  # 0 * 0.9 + 2.0 * 0.1
        assert torch.allclose(loss_fn.center, expected_center, atol=1e-5), (
            f"Center update failed: expected {expected_center[0, 0]}, got {loss_fn.center[0, 0]}"
        )

        # Test Sharpening: Teacher entropy with small tau_t should be significantly lower than student entropy
        logits = torch.randn(4, out_dim)
        p_teacher = F.softmax((logits - loss_fn.center) / 0.04, dim=-1)
        p_student = F.softmax(logits / 0.1, dim=-1)

        ent_teacher = -(p_teacher * torch.log(p_teacher + 1e-9)).sum(dim=-1).mean()
        ent_student = -(p_student * torch.log(p_student + 1e-9)).sum(dim=-1).mean()
        assert ent_teacher < ent_student, (
            f"Teacher entropy ({ent_teacher:.3f}) should be lower than student ({ent_student:.3f}) due to sharpening"
        )
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 3: Forward Pass & Detached Teacher Audit
    # ---------------------------------------------------------
    print("Running Test 3: DINO Forward & Detached Teacher Audit...", end=" ")
    try:
        dino = DINO(embed_dim=32, out_dim=64, student_temp=0.1, teacher_temp=0.04)
        x1 = torch.randn(2, 3, 16, 16)
        x2 = torch.randn(2, 3, 16, 16)

        loss, s_out, t_out = dino(x1, x2)
        assert s_out.shape == (4, 64), f"Expected student output shape (4, 64), got {s_out.shape}"
        assert t_out.shape == (4, 64), f"Expected teacher output shape (4, 64), got {t_out.shape}"
        assert loss.ndim == 0 and loss.item() > 0.0

        loss.backward()
        # Verify student receives gradients
        student_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in dino.student.parameters())
        assert student_has_grad, "Student parameters must receive gradients"

        # Verify teacher DOES NOT receive gradients
        for p in dino.teacher.parameters():
            assert p.grad is None, "Detached teacher violation! Teacher must not receive gradients"
        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Test 4: Self-Distillation Optimization & Anti-Collapse Audit
    # ---------------------------------------------------------
    print("Running Test 4: Self-Distillation Optimization & Anti-Collapse...", end=" ")
    try:
        torch.manual_seed(42)
        N = 4
        base_images = torch.randn(N, 3, 16, 16)

        dino = DINO(embed_dim=32, out_dim=64, student_temp=0.1, teacher_temp=0.04, momentum_teacher=0.99)
        optimizer = optim.AdamW(dino.student.parameters(), lr=1e-3)

        initial_teacher_weight = next(dino.teacher.parameters()).clone()

        for step in range(30):
            optimizer.zero_grad()
            v1 = base_images + 0.04 * torch.randn_like(base_images)
            v2 = base_images + 0.04 * torch.randn_like(base_images)

            loss, s_out, t_out = dino(v1, v2)
            assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
            loss.backward()
            optimizer.step()

        # 1. Verify center updated from initial zeros
        assert not torch.allclose(dino.loss_fn.center, torch.zeros_like(dino.loss_fn.center)), (
            "Center buffer must update via momentum tracking teacher output"
        )

        # 2. Verify teacher weights updated via EMA
        current_teacher_weight = next(dino.teacher.parameters())
        assert not torch.allclose(initial_teacher_weight, current_teacher_weight), (
            "Teacher weights must update via student EMA"
        )

        # 3. Verify teacher probabilities avoid uniform collapse (entropy < log(K))
        p_t = F.softmax((t_out - dino.loss_fn.center) / dino.teacher_temp, dim=-1)
        mean_entropy = -(p_t * torch.log(p_t + 1e-9)).sum(dim=-1).mean().item()
        max_entropy = math.log(64)
        assert mean_entropy < max_entropy, (
            f"Uniform collapse detected! Entropy {mean_entropy:.3f} equals max entropy {max_entropy:.3f}"
        )
        assert mean_entropy > 0.01, "Dirac collapse detected! Entropy is zero."

        print(f"[PASS] (Loss {loss.item():.3f}, Teacher Entropy {mean_entropy:.3f} < {max_entropy:.3f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        sys.exit(1)

    print("\nAll DINO unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
