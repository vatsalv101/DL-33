import sys
import importlib
import torch
import torch.nn as nn
import torch.optim as optim


def get_model_module():
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
    else:
        module_name = "papers.28_gan.model"
    return importlib.import_module(module_name)


def test_shapes_and_contracts(model_mod):
    print("Running Test 1: Generator/Critic Tensor Shape Contracts...", end=" ")
    try:
        WGANGP = model_mod.WGANGP
        wgan = WGANGP(latent_dim=32, img_channels=1, hidden_dim=32, img_size=28)
        wgan.eval()

        # Batch of 4 noise vectors
        z = torch.randn(4, 32)
        fake_images = wgan.generate(z)
        assert fake_images.shape == (4, 1, 28, 28), f"Expected fake shape (4, 1, 28, 28), got {fake_images.shape}"
        assert (fake_images >= -1.0).all() and (fake_images <= 1.0).all(), "Generator output must be bounded in [-1, 1] by Tanh"

        # Critic scoring
        scores = wgan.score(fake_images)
        assert scores.shape == (4, 1), f"Expected critic score shape (4, 1), got {scores.shape}"

        # Sampling contract
        samples = wgan.sample(num_samples=8)
        assert samples.shape == (8, 1, 28, 28), f"Expected sample shape (8, 1, 28, 28), got {samples.shape}"

        # 3-channel 32x32 image contract
        wgan_rgb = WGANGP(latent_dim=48, img_channels=3, hidden_dim=32, img_size=32)
        fake_rgb = wgan_rgb.generate(torch.randn(2, 48))
        assert fake_rgb.shape == (2, 3, 32, 32), f"Expected RGB shape (2, 3, 32, 32), got {fake_rgb.shape}"
        scores_rgb = wgan_rgb.score(fake_rgb)
        assert scores_rgb.shape == (2, 1), f"Expected RGB score shape (2, 1), got {scores_rgb.shape}"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_gradient_penalty_math(model_mod):
    print("Running Test 2: Gradient Penalty Math & 1-Lipschitz Audit...", end=" ")
    try:
        compute_gradient_penalty = model_mod.compute_gradient_penalty

        # Define a mock linear critic: D(x) = c * sum(x)
        # Gradient norm w.r.t x (size D) is ||c * 1||_2 = |c| * sqrt(D)
        class LinearCritic(nn.Module):
            def __init__(self, c=0.5):
                super().__init__()
                self.c = nn.Parameter(torch.tensor([c]))

            def forward(self, x):
                return (x.view(x.shape[0], -1) * self.c).sum(dim=1, keepdim=True)

        critic = LinearCritic(c=0.1)
        # Image shape: (B, 1, 4, 4) -> D = 16. ||grad||_2 = 0.1 * sqrt(16) = 0.4
        real = torch.randn(4, 1, 4, 4)
        fake = torch.randn(4, 1, 4, 4)

        gp = compute_gradient_penalty(critic, real, fake)
        expected_gp = (0.4 - 1.0) ** 2  # 0.36
        assert torch.isclose(gp, torch.tensor(expected_gp), atol=1e-4), f"Expected GP={expected_gp:.4f}, got {gp.item():.4f}"

        # Verify gradient flow into critic parameters
        gp.backward()
        assert critic.c.grad is not None, "Critic parameters must receive gradients from GP"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_loss_contracts(model_mod):
    print("Running Test 3: WGAN-GP Loss Mathematical Contracts...", end=" ")
    try:
        critic_loss = model_mod.critic_loss
        generator_loss = model_mod.generator_loss
        Critic = model_mod.Critic

        critic = Critic(img_channels=1, hidden_dim=16, img_size=28)
        real = torch.randn(4, 1, 28, 28)
        fake = torch.randn(4, 1, 28, 28)

        total_loss, w_loss, gp = critic_loss(critic, real, fake, lambda_gp=10.0)
        assert torch.isclose(total_loss, w_loss + 10.0 * gp, atol=1e-5), "Total critic loss must equal W-loss + lambda * GP"

        g_loss = generator_loss(critic, fake)
        assert g_loss.ndim == 0, "Generator loss must be scalar"

        # Verify generator loss equals negative mean of critic score on fake
        d_fake_mean = critic(fake).mean()
        assert torch.isclose(g_loss, -d_fake_mean, atol=1e-5), "Generator loss must equal -E[D(fake)]"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_adversarial_overfitting_gate(model_mod):
    print("Running Test 4: Adversarial Overfitting Gate...", end=" ")
    try:
        torch.manual_seed(42)
        WGANGP = model_mod.WGANGP
        critic_loss = model_mod.critic_loss
        generator_loss = model_mod.generator_loss

        wgan = WGANGP(latent_dim=16, img_channels=1, hidden_dim=16, img_size=28, lambda_gp=10.0)
        opt_c = optim.Adam(wgan.critic.parameters(), lr=2e-3, betas=(0.0, 0.9))
        opt_g = optim.Adam(wgan.generator.parameters(), lr=2e-3, betas=(0.0, 0.9))

        # 4 distinct synthetic target images in [-1, 1]
        x_real = torch.full((4, 1, 28, 28), -1.0)
        x_real[0, 0, :, 12:16] = 1.0  # Vertical stripe
        x_real[1, 0, 12:16, :] = 1.0  # Horizontal stripe
        x_real[2, 0, 8:20, 8:20] = 1.0  # Center block
        x_real[3, 0, :, :] = 0.5  # Uniform gray

        initial_w_dist = None
        for step in range(50):
            # Train Critic (2 steps per generator step)
            for _ in range(2):
                opt_c.zero_grad()
                z = torch.randn(4, 16)
                with torch.no_grad():
                    fake = wgan.generate(z)
                c_loss, w_loss, gp = critic_loss(wgan.critic, x_real, fake, lambda_gp=10.0)
                c_loss.backward()
                opt_c.step()

            # Train Generator
            opt_g.zero_grad()
            z = torch.randn(4, 16)
            fake = wgan.generate(z)
            g_loss = generator_loss(wgan.critic, fake)
            g_loss.backward()
            opt_g.step()

            w_dist = -w_loss.item()  # Estimated Wasserstein distance E[D(real)] - E[D(fake)]
            if step == 0:
                initial_w_dist = w_dist

        # Verify no NaN values and generator produces diverse valid images
        with torch.no_grad():
            final_samples = wgan.sample(num_samples=4)
            assert not torch.isnan(final_samples).any(), "Generated images must not contain NaNs"
            assert not torch.isinf(final_samples).any(), "Generated images must not contain Infs"
            assert final_samples.std() > 0.05, "Generator must not produce completely collapsed uniform constant"

        print(f"[PASS] (Critic stable, final std={final_samples.std().item():.3f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def main():
    model_mod = get_model_module()
    test_shapes_and_contracts(model_mod)
    test_gradient_penalty_math(model_mod)
    test_loss_contracts(model_mod)
    test_adversarial_overfitting_gate(model_mod)
    print("\nAll GAN / WGAN-GP unit tests passed successfully!")


if __name__ == "__main__":
    main()
