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
        module_name = "papers.29_ddpm.model"
    return importlib.import_module(module_name)


def test_shapes_and_contracts(model_mod):
    print("Running Test 1: U-Net Denoiser & Tensor Shape Contracts...", end=" ")
    try:
        UNetDenoiser = model_mod.UNetDenoiser
        SinusoidalPosEmb = model_mod.SinusoidalPosEmb

        # Sinusoidal embedding check
        pos_emb = SinusoidalPosEmb(dim=32)
        t = torch.tensor([0, 10, 100, 999])
        emb = pos_emb(t)
        assert emb.shape == (4, 32), f"Expected pos_emb shape (4, 32), got {emb.shape}"

        # 28x28 1-channel U-Net
        unet = UNetDenoiser(in_channels=1, out_channels=1, hidden_dims=(16, 32, 64), time_emb_dim=32)
        unet.eval()
        x = torch.randn(4, 1, 28, 28)
        out = unet(x, t)
        assert out.shape == (4, 1, 28, 28), f"Expected U-Net output shape (4, 1, 28, 28), got {out.shape}"

        # 32x32 3-channel U-Net
        unet_rgb = UNetDenoiser(in_channels=3, out_channels=3, hidden_dims=(16, 32, 64), time_emb_dim=32)
        x_rgb = torch.randn(2, 3, 32, 32)
        t_rgb = torch.tensor([5, 50])
        out_rgb = unet_rgb(x_rgb, t_rgb)
        assert out_rgb.shape == (2, 3, 32, 32), f"Expected RGB U-Net output shape (2, 3, 32, 32), got {out_rgb.shape}"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_forward_diffusion_math(model_mod):
    print("Running Test 2: Closed-Form Forward Diffusion Math Contract...", end=" ")
    try:
        UNetDenoiser = model_mod.UNetDenoiser
        GaussianDiffusion = model_mod.GaussianDiffusion

        unet = UNetDenoiser(in_channels=1, out_channels=1, hidden_dims=(16, 32, 64), time_emb_dim=32)
        diffusion = GaussianDiffusion(unet, timesteps=1000)

        x_start = torch.ones(4, 1, 28, 28)
        fixed_noise = torch.randn(4, 1, 28, 28)

        # At t = 0: alpha_bar_0 is close to 1.0 -> x_t should be very close to x_start
        t_zero = torch.zeros(4, dtype=torch.long)
        x_0 = diffusion.q_sample(x_start, t_zero, fixed_noise)
        diff_0 = (x_0 - x_start).abs().mean().item()
        assert diff_0 < 0.1, f"At t=0, x_t should be very close to x_start, got mean diff {diff_0:.4f}"

        # At t = T - 1: alpha_bar_{T-1} is close to 0.0 -> x_t should be dominated by noise
        t_last = torch.full((4,), 999, dtype=torch.long)
        x_T = diffusion.q_sample(x_start, t_last, fixed_noise)
        diff_T = (x_T - fixed_noise).abs().mean().item()
        assert diff_T < 0.15, f"At t=T-1, x_t should be dominated by noise, got mean diff {diff_T:.4f}"

        # Exact formula test
        alpha_bar_500 = diffusion.alphas_cumprod[500].item()
        expected = math.sqrt(alpha_bar_500) * x_start[0:1] + math.sqrt(1.0 - alpha_bar_500) * fixed_noise[0:1]
        actual = diffusion.q_sample(x_start[0:1], torch.tensor([500]), fixed_noise[0:1])
        assert torch.allclose(expected, actual, atol=1e-5), "q_sample must match closed-form equation exactly"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_loss_and_gradients(model_mod):
    print("Running Test 3: Training Loss & Gradient Flow...", end=" ")
    try:
        UNetDenoiser = model_mod.UNetDenoiser
        GaussianDiffusion = model_mod.GaussianDiffusion

        unet = UNetDenoiser(in_channels=1, out_channels=1, hidden_dims=(16, 32, 64), time_emb_dim=32)
        diffusion = GaussianDiffusion(unet, timesteps=100)

        x = torch.randn(4, 1, 28, 28)
        loss = diffusion.compute_loss(x)
        assert loss.ndim == 0, "compute_loss must return a scalar tensor"
        assert loss.item() > 0.0, "MSE loss must be strictly positive"

        loss.backward()
        for name, param in unet.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} did not receive gradients"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_reverse_sampling_overfit_gate(model_mod):
    print("Running Test 4: Reverse Sampling & Overfitting Gate...", end=" ")
    try:
        torch.manual_seed(42)
        UNetDenoiser = model_mod.UNetDenoiser
        GaussianDiffusion = model_mod.GaussianDiffusion

        unet = UNetDenoiser(in_channels=1, out_channels=1, hidden_dims=(16, 32, 64), time_emb_dim=32)
        diffusion = GaussianDiffusion(unet, timesteps=50)
        optimizer = optim.Adam(unet.parameters(), lr=5e-3)

        # 4 distinct synthetic visual patterns
        x_target = torch.full((4, 1, 28, 28), -1.0)
        x_target[0, 0, :, 12:16] = 1.0  # Vertical stripe
        x_target[1, 0, 12:16, :] = 1.0  # Horizontal stripe
        x_target[2, 0, 8:20, 8:20] = 1.0  # Center block
        x_target[3, 0, :, :] = 0.5  # Uniform gray

        initial_loss = None
        for step in range(60):
            optimizer.zero_grad()
            loss = diffusion.compute_loss(x_target)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.4, f"Loss did not decrease sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"

        # Test reverse generation
        with torch.no_grad():
            samples = diffusion.sample(shape=(2, 1, 28, 28))
            assert samples.shape == (2, 1, 28, 28), f"Expected samples shape (2, 1, 28, 28), got {samples.shape}"
            assert not torch.isnan(samples).any(), "Sampled images must not contain NaNs"
            assert not torch.isinf(samples).any(), "Sampled images must not contain Infs"

        print(f"[PASS] (Loss {initial_loss:.4f} -> {final_loss:.4f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def main():
    model_mod = get_model_module()
    test_shapes_and_contracts(model_mod)
    test_forward_diffusion_math(model_mod)
    test_loss_and_gradients(model_mod)
    test_reverse_sampling_overfit_gate(model_mod)
    print("\nAll DDPM unit tests passed successfully!")


if __name__ == "__main__":
    main()
