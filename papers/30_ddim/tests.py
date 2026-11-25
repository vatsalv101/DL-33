import sys
import math
import importlib
import torch
import torch.nn as nn
from core.ddpm import UNetDenoiser


def get_model_module():
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
    else:
        module_name = "papers.30_ddim.model"
    return importlib.import_module(module_name)


def test_shapes_and_schedules(model_mod):
    print("Running Test 1: DDIM Sub-Sequence Scheduling & Tensor Shapes...", end=" ")
    try:
        DDIMScheduler = model_mod.DDIMScheduler
        DDIM = model_mod.DDIM

        scheduler = DDIMScheduler(num_train_timesteps=1000)
        timesteps = scheduler.set_timesteps(num_inference_steps=20)

        assert len(timesteps) == 20, f"Expected 20 timesteps, got {len(timesteps)}"
        assert timesteps[0] > timesteps[-1], "Timesteps must be in descending order for reverse diffusion"
        assert timesteps[-1] == 0, f"Last timestep must be 0, got {timesteps[-1].item()}"

        # Test DDIM wrapper sampling contract
        unet = UNetDenoiser(in_channels=1, out_channels=1, hidden_dims=(16, 32, 64), time_emb_dim=32)
        ddim = DDIM(unet, num_train_timesteps=1000)

        samples = ddim.sample(shape=(2, 1, 28, 28), num_inference_steps=5, eta=0.0)
        assert samples.shape == (2, 1, 28, 28), f"Expected samples shape (2, 1, 28, 28), got {samples.shape}"
        assert not torch.isnan(samples).any(), "Samples must not contain NaNs"
        assert (samples >= -1.0).all() and (samples <= 1.0).all(), "Samples must be clamped in [-1, 1]"

        # 3-channel 32x32 contract
        unet_rgb = UNetDenoiser(in_channels=3, out_channels=3, hidden_dims=(16, 32, 64), time_emb_dim=32)
        ddim_rgb = DDIM(unet_rgb, num_train_timesteps=1000)
        samples_rgb = ddim_rgb.sample(shape=(1, 3, 32, 32), num_inference_steps=5)
        assert samples_rgb.shape == (1, 3, 32, 32), f"Expected RGB shape (1, 3, 32, 32), got {samples_rgb.shape}"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_deterministic_consistency(model_mod):
    print("Running Test 2: Deterministic Sampling Consistency (eta = 0)...", end=" ")
    try:
        DDIM = model_mod.DDIM
        unet = UNetDenoiser(in_channels=1, out_channels=1, hidden_dims=(16, 32, 64), time_emb_dim=32)
        ddim = DDIM(unet, num_train_timesteps=100)

        # Fix starting noise
        fixed_noise = torch.randn(2, 1, 28, 28)

        # Generate twice from the exact same initial noise using eta = 0.0 (deterministic)
        timesteps = ddim.scheduler.set_timesteps(num_inference_steps=5)

        img1 = fixed_noise.clone()
        for i, t in enumerate(timesteps):
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((2,), t, dtype=torch.long)
            noise_pred = ddim.model(img1, t_batch)
            img1, _ = ddim.scheduler.step(noise_pred, t.item(), prev_t.item() if isinstance(prev_t, torch.Tensor) else prev_t, img1, eta=0.0)

        img2 = fixed_noise.clone()
        for i, t in enumerate(timesteps):
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((2,), t, dtype=torch.long)
            noise_pred = ddim.model(img2, t_batch)
            img2, _ = ddim.scheduler.step(noise_pred, t.item(), prev_t.item() if isinstance(prev_t, torch.Tensor) else prev_t, img2, eta=0.0)

        assert torch.allclose(img1, img2, atol=1e-6), "DDIM with eta=0 must be completely deterministic for identical noise"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_ddim_math_contract(model_mod):
    print("Running Test 3: DDIM Step Mathematical Contract...", end=" ")
    try:
        DDIMScheduler = model_mod.DDIMScheduler
        scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)

        x_t = torch.randn(2, 1, 14, 14)
        noise_pred = torch.randn(2, 1, 14, 14)
        t = 500
        t_prev = 250

        # Step calculation with eta = 0.0
        prev_sample, pred_x_0 = scheduler.step(noise_pred, t, t_prev, x_t, eta=0.0)

        alpha_bar_t = scheduler.alphas_cumprod[t].item()
        alpha_bar_prev = scheduler.alphas_cumprod[t_prev].item()

        # Expected manual calculation
        expected_pred_x_0 = (x_t - math.sqrt(1.0 - alpha_bar_t) * noise_pred) / math.sqrt(alpha_bar_t)
        expected_dir = math.sqrt(1.0 - alpha_bar_prev) * noise_pred
        expected_prev_sample = math.sqrt(alpha_bar_prev) * expected_pred_x_0 + expected_dir

        assert torch.allclose(pred_x_0, expected_pred_x_0, atol=1e-5), "pred_x_0 calculation mismatch"
        assert torch.allclose(prev_sample, expected_prev_sample, atol=1e-5), "prev_sample calculation mismatch"

        # Boundary check: when t_prev = -1, prev_sample must equal pred_x_0 exactly
        final_sample, final_pred_x_0 = scheduler.step(noise_pred, 0, -1, x_t, eta=0.0)
        assert torch.allclose(final_sample, final_pred_x_0, atol=1e-5), "At terminal step t_prev=-1, output must equal pred_x_0"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_inversion_and_reconstruction(model_mod):
    print("Running Test 4: Latent Inversion & Reconstruction Gate...", end=" ")
    try:
        DDIM = model_mod.DDIM
        unet = UNetDenoiser(in_channels=1, out_channels=1, hidden_dims=(16, 32, 64), time_emb_dim=32)
        ddim = DDIM(unet, num_train_timesteps=100)

        # Smooth synthetic visual pattern in [-1, 1]
        x_orig = torch.full((1, 1, 28, 28), -0.5)
        x_orig[0, 0, 10:18, 10:18] = 0.8

        # Invert x_0 -> x_T using 10 steps
        z_latent = ddim.invert(x_orig, num_inference_steps=10)
        assert z_latent.shape == (1, 1, 28, 28), f"Expected latent shape (1, 1, 28, 28), got {z_latent.shape}"
        assert not torch.isnan(z_latent).any(), "Inverted latent must not contain NaNs"

        # Reconstruct from z_latent -> x_recon
        timesteps = ddim.scheduler.set_timesteps(num_inference_steps=10)
        img = z_latent.clone()
        for i, t in enumerate(timesteps):
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((1,), t, dtype=torch.long)
            noise_pred = ddim.model(img, t_batch)
            img, _ = ddim.scheduler.step(noise_pred, t.item(), prev_t.item() if isinstance(prev_t, torch.Tensor) else prev_t, img, eta=0.0)

        recon_error = (x_orig - img).abs().mean().item()
        assert recon_error < 0.35, f"Reconstruction error too high after inversion: {recon_error:.4f}"

        # Interpolation test
        x_other = torch.full((1, 1, 28, 28), 0.5)
        interp = ddim.interpolate(x_orig, x_other, steps=4, num_inference_steps=5)
        assert interp.shape == (4, 1, 28, 28), f"Expected interpolation shape (4, 1, 28, 28), got {interp.shape}"

        print(f"[PASS] (Inversion error = {recon_error:.4f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def main():
    model_mod = get_model_module()
    test_shapes_and_schedules(model_mod)
    test_deterministic_consistency(model_mod)
    test_ddim_math_contract(model_mod)
    test_inversion_and_reconstruction(model_mod)
    print("\nAll DDIM unit tests passed successfully!")


if __name__ == "__main__":
    main()
