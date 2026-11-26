import sys
import math
import importlib
import torch
import torch.nn as nn
import torch.optim as optim


def get_model_module():
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
    else:
        module_name = "papers.31_cfg.model"
    return importlib.import_module(module_name)


def test_shapes_and_conditioning(model_mod):
    print("Running Test 1: Class-Conditioned U-Net & Tensor Shape Contracts...", end=" ")
    try:
        ClassConditionedUNet = model_mod.ClassConditionedUNet
        unet = ClassConditionedUNet(
            in_channels=1, out_channels=1, num_classes=4, hidden_dims=(16, 32, 64), cond_dim=32
        )
        unet.eval()

        x = torch.randn(4, 1, 28, 28)
        t = torch.tensor([10, 50, 100, 500])
        y = torch.tensor([0, 1, 2, 3])

        # Conditional forward pass
        out = unet(x, t, y)
        assert out.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {out.shape}"

        # Unconditional forward pass with y=None (null token)
        out_uncond = unet(x, t, None)
        assert out_uncond.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {out_uncond.shape}"

        # 3-channel 32x32 contract
        unet_rgb = ClassConditionedUNet(
            in_channels=3, out_channels=3, num_classes=10, hidden_dims=(16, 32, 64), cond_dim=32
        )
        x_rgb = torch.randn(2, 3, 32, 32)
        t_rgb = torch.tensor([20, 80])
        y_rgb = torch.tensor([4, 7])
        out_rgb = unet_rgb(x_rgb, t_rgb, y_rgb)
        assert out_rgb.shape == (2, 3, 32, 32), f"Expected RGB shape (2, 3, 32, 32), got {out_rgb.shape}"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_guidance_extrapolation_math(model_mod):
    print("Running Test 2: CFG Extrapolation & Negative Prompting Math...", end=" ")
    try:
        ClassConditionedUNet = model_mod.ClassConditionedUNet
        ClassifierFreeDiffusion = model_mod.ClassifierFreeDiffusion

        unet = ClassConditionedUNet(
            in_channels=1, out_channels=1, num_classes=4, hidden_dims=(16, 32, 64), cond_dim=32
        )
        diffusion = ClassifierFreeDiffusion(unet, timesteps=100)

        x_t = torch.randn(2, 1, 28, 28)
        t = torch.tensor([25, 75])
        y = torch.tensor([1, 2])

        # Test guidance_scale = 1.0 (pure conditional)
        eps_cond_only = diffusion.predict_guided_noise(x_t, t, y, guidance_scale=1.0)
        expected_cond = unet(x_t, t, y)
        assert torch.allclose(eps_cond_only, expected_cond, atol=1e-5), "At s=1.0, output must match conditional prediction"

        # Test guidance_scale = 0.0 (pure unconditional)
        eps_uncond_only = diffusion.predict_guided_noise(x_t, t, y, guidance_scale=0.0)
        expected_uncond = unet(x_t, t, torch.full_like(y, unet.null_token_id))
        assert torch.allclose(eps_uncond_only, expected_uncond, atol=1e-5), "At s=0.0, output must match unconditional prediction"

        # Test guidance_scale = 3.0 extrapolation formula: eps_uncond + 3.0 * (eps_cond - eps_uncond)
        eps_guided = diffusion.predict_guided_noise(x_t, t, y, guidance_scale=3.0)
        expected_guided = expected_uncond + 3.0 * (expected_cond - expected_uncond)
        assert torch.allclose(eps_guided, expected_guided, atol=1e-5), "CFG extrapolation formula mismatch"

        # Test negative prompt condition y_neg
        y_neg = torch.tensor([0, 3])
        eps_neg_guided = diffusion.predict_guided_noise(x_t, t, y, guidance_scale=2.5, y_neg=y_neg)
        eps_neg_base = unet(x_t, t, y_neg)
        expected_neg_guided = eps_neg_base + 2.5 * (expected_cond - eps_neg_base)
        assert torch.allclose(eps_neg_guided, expected_neg_guided, atol=1e-5), "Negative prompt guidance mismatch"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_condition_dropout_training(model_mod):
    print("Running Test 3: Conditioning Dropout & Gradient Flow...", end=" ")
    try:
        ClassConditionedUNet = model_mod.ClassConditionedUNet
        ClassifierFreeDiffusion = model_mod.ClassifierFreeDiffusion

        unet = ClassConditionedUNet(
            in_channels=1, out_channels=1, num_classes=4, hidden_dims=(16, 32, 64), cond_dim=32
        )
        diffusion = ClassifierFreeDiffusion(unet, timesteps=100)

        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 4, (8,))

        # Force dropout to train null token
        loss = diffusion.compute_loss(x, y, p_uncond=0.5)
        assert loss.ndim == 0, "Loss must be scalar"
        assert loss.item() > 0.0, "Loss must be positive"

        loss.backward()

        # Check gradients on class embeddings including the null token (index 4)
        assert unet.class_embedding.weight.grad is not None, "Class embedding must receive gradients"
        assert not torch.allclose(
            unet.class_embedding.weight.grad, torch.zeros_like(unet.class_embedding.weight.grad)
        ), "Class embedding gradients cannot be all zero"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_guided_sampling_gate(model_mod):
    print("Running Test 4: Class-Guided Sampling Gate...", end=" ")
    try:
        torch.manual_seed(42)
        ClassConditionedUNet = model_mod.ClassConditionedUNet
        ClassifierFreeDiffusion = model_mod.ClassifierFreeDiffusion

        unet = ClassConditionedUNet(
            in_channels=1, out_channels=1, num_classes=2, hidden_dims=(16, 32, 64), cond_dim=32
        )
        diffusion = ClassifierFreeDiffusion(unet, timesteps=40)
        optimizer = optim.Adam(unet.parameters(), lr=5e-3)

        # Class 0: Horizontal stripe, Class 1: Vertical stripe
        x_target = torch.full((4, 1, 28, 28), -1.0)
        x_target[0:2, 0, 12:16, :] = 1.0  # Horizontal (Class 0)
        x_target[2:4, 0, :, 12:16] = 1.0  # Vertical (Class 1)
        y_target = torch.tensor([0, 0, 1, 1])

        initial_loss = None
        for step in range(60):
            optimizer.zero_grad()
            loss = diffusion.compute_loss(x_target, y_target, p_uncond=0.2)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.45, f"Loss did not decrease sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"

        # Sample with high guidance scale s = 3.0
        with torch.no_grad():
            sample_labels = torch.tensor([0, 1])
            samples = diffusion.sample(y=sample_labels, num_inference_steps=10, guidance_scale=3.0)
            assert samples.shape == (2, 1, 28, 28), f"Expected shape (2, 1, 28, 28), got {samples.shape}"
            assert not torch.isnan(samples).any(), "Samples must not contain NaNs"

        print(f"[PASS] (Loss {initial_loss:.4f} -> {final_loss:.4f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def main():
    model_mod = get_model_module()
    test_shapes_and_conditioning(model_mod)
    test_guidance_extrapolation_math(model_mod)
    test_condition_dropout_training(model_mod)
    test_guided_sampling_gate(model_mod)
    print("\nAll CFG unit tests passed successfully!")


if __name__ == "__main__":
    main()
