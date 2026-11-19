import sys
import importlib
import torch
import torch.nn as nn
import torch.optim as optim


def get_model_module():
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
    else:
        module_name = "papers.26_vae.model"
    return importlib.import_module(module_name)


def test_shapes_and_contracts(model_mod):
    print("Running Test 1: Encoder/Decoder Tensor Shape Contracts...", end=" ")
    try:
        VAE = model_mod.VAE
        vae = VAE(in_channels=1, latent_dim=16, img_size=28)
        vae.eval()

        # Batch of 4 28x28 images
        x = torch.rand(4, 1, 28, 28)
        recon_x, mu, logvar = vae(x)

        assert recon_x.shape == (4, 1, 28, 28), f"Expected recon shape (4, 1, 28, 28), got {recon_x.shape}"
        assert mu.shape == (4, 16), f"Expected mu shape (4, 16), got {mu.shape}"
        assert logvar.shape == (4, 16), f"Expected logvar shape (4, 16), got {logvar.shape}"
        assert (recon_x >= 0.0).all() and (recon_x <= 1.0).all(), "Reconstructed image must be in [0, 1] range"

        # Sampling contract
        samples = vae.sample(num_samples=8)
        assert samples.shape == (8, 1, 28, 28), f"Expected samples shape (8, 1, 28, 28), got {samples.shape}"

        # Interpolation contract
        z1 = torch.randn(16)
        z2 = torch.randn(16)
        interp = vae.interpolate(z1, z2, steps=10)
        assert interp.shape == (10, 1, 28, 28), f"Expected interp shape (10, 1, 28, 28), got {interp.shape}"

        # 3-channel 32x32 contract
        vae_cifar = VAE(in_channels=3, latent_dim=32, img_size=32)
        x_cifar = torch.rand(2, 3, 32, 32)
        recon_cifar, mu_c, logvar_c = vae_cifar(x_cifar)
        assert recon_cifar.shape == (2, 3, 32, 32), f"Expected CIFAR recon shape (2, 3, 32, 32), got {recon_cifar.shape}"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_reparameterization_gradient_flow(model_mod):
    print("Running Test 2: Reparameterization Trick & Gradient Flow...", end=" ")
    try:
        reparameterize = model_mod.reparameterize

        mu = torch.randn(8, 16, requires_grad=True)
        logvar = torch.randn(8, 16, requires_grad=True)

        z = reparameterize(mu, logvar)
        assert z.shape == (8, 16), f"Expected z shape (8, 16), got {z.shape}"

        # Gradient check
        loss = z.sum()
        loss.backward()

        assert mu.grad is not None, "mu must receive gradients through reparameterization"
        assert logvar.grad is not None, "logvar must receive gradients through reparameterization"
        assert not torch.allclose(mu.grad, torch.zeros_like(mu.grad)), "mu gradients cannot be all zero"
        assert not torch.allclose(logvar.grad, torch.zeros_like(logvar.grad)), "logvar gradients cannot be all zero"

        # Stochasticity check: two independent calls must yield different latent codes
        with torch.no_grad():
            z1 = reparameterize(mu, logvar)
            z2 = reparameterize(mu, logvar)
            assert not torch.allclose(z1, z2), "Stochastic reparameterization must produce distinct samples"

            # Low variance limit: when logvar -> -inf, z -> mu
            logvar_zero_var = torch.full_like(logvar, -100.0)
            z_det = reparameterize(mu, logvar_zero_var)
            assert torch.allclose(z_det, mu, atol=1e-4), "As sigma -> 0, z must converge to mu"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_kl_divergence_math_contract(model_mod):
    print("Running Test 3: Analytical KL Divergence Mathematical Contract...", end=" ")
    try:
        vae_loss = model_mod.vae_loss

        # Case 1: Standard normal prior match (mu = 0, logvar = 0) => D_KL must be exactly 0
        mu_zero = torch.zeros(4, 16)
        logvar_zero = torch.zeros(4, 16)
        dummy_x = torch.rand(4, 1, 28, 28)

        _, _, kl_zero = vae_loss(dummy_x, dummy_x, mu_zero, logvar_zero)
        assert torch.isclose(kl_zero, torch.tensor(0.0), atol=1e-6), f"D_KL(N(0, I) || N(0, I)) must be 0, got {kl_zero.item()}"

        # Case 2: Non-zero mean or non-unit variance => D_KL must be strictly positive
        mu_shifted = torch.ones(4, 16)
        logvar_scaled = torch.zeros(4, 16)
        _, _, kl_shifted = vae_loss(dummy_x, dummy_x, mu_shifted, logvar_scaled)
        # Expected D_KL = 0.5 * sum(mu^2) = 0.5 * 16 * 1.0 = 8.0
        assert torch.isclose(kl_shifted, torch.tensor(8.0), atol=1e-5), f"Expected KL of 8.0 for mu=1, got {kl_shifted.item()}"

        # Case 3: Verify beta weighting
        total_beta1, recon, kl = vae_loss(dummy_x, dummy_x, mu_shifted, logvar_scaled, beta=1.0)
        total_beta2, _, _ = vae_loss(dummy_x, dummy_x, mu_shifted, logvar_scaled, beta=2.0)
        assert torch.isclose(total_beta2, recon + 2.0 * kl, atol=1e-5), "Total loss must respect beta weighting"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_overfitting_gate(model_mod):
    print("Running Test 4: Generative Overfitting Gate...", end=" ")
    try:
        torch.manual_seed(42)
        VAE = model_mod.VAE
        vae_loss = model_mod.vae_loss

        vae = VAE(in_channels=1, latent_dim=8, img_size=28)
        optimizer = optim.Adam(vae.parameters(), lr=1e-2)

        # Create 4 distinct synthetic visual patterns (vertical line, horizontal line, dot, cross)
        x_target = torch.zeros(4, 1, 28, 28)
        x_target[0, 0, :, 12:16] = 1.0  # Vertical stripe
        x_target[1, 0, 12:16, :] = 1.0  # Horizontal stripe
        x_target[2, 0, 10:18, 10:18] = 1.0  # Center block
        x_target[3, 0, :, :] = 0.5  # Uniform gray

        initial_loss = None
        final_loss = None

        for step in range(60):
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x_target)
            total_loss, recon_loss, kl_loss = vae_loss(recon_x, x_target, mu, logvar, beta=0.1)

            if step == 0:
                initial_loss = total_loss.item()

            total_loss.backward()
            optimizer.step()

        final_loss = total_loss.item()
        assert final_loss < initial_loss * 0.35, f"Expected loss to decrease significantly: initial={initial_loss:.3f}, final={final_loss:.3f}"
        print(f"[PASS] (Loss {initial_loss:.3f} -> {final_loss:.3f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def main():
    model_mod = get_model_module()
    test_shapes_and_contracts(model_mod)
    test_reparameterization_gradient_flow(model_mod)
    test_kl_divergence_math_contract(model_mod)
    test_overfitting_gate(model_mod)
    print("\nAll VAE unit tests passed successfully!")


if __name__ == "__main__":
    main()
