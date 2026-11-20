import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_model_module():
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
    else:
        module_name = "papers.27_vqvae.model"
    return importlib.import_module(module_name)


def test_shapes_and_contracts(model_mod):
    print("Running Test 1: Encoder/Decoder Tensor Shape Contracts...", end=" ")
    try:
        VQVAE = model_mod.VQVAE
        vqvae = VQVAE(in_channels=1, hidden_dim=32, embedding_dim=16, num_embeddings=64)
        vqvae.eval()

        # Batch of 4 28x28 images
        x = torch.rand(4, 1, 28, 28)
        x_recon, vq_loss, perplexity, indices = vqvae(x)

        assert x_recon.shape == (4, 1, 28, 28), f"Expected x_recon shape (4, 1, 28, 28), got {x_recon.shape}"
        assert indices.shape == (4, 7, 7), f"Expected indices shape (4, 7, 7), got {indices.shape}"
        assert vq_loss.ndim == 0, f"Expected scalar vq_loss, got shape {vq_loss.shape}"
        assert perplexity.ndim == 0, f"Expected scalar perplexity, got shape {perplexity.shape}"
        assert (indices >= 0).all() and (indices < 64).all(), "Indices must fall within [0, num_embeddings - 1]"
        assert (x_recon >= 0.0).all() and (x_recon <= 1.0).all(), "Reconstructed image must be in [0, 1] range"

        # Verify decode_code equivalence
        with torch.no_grad():
            x_from_code = vqvae.decode_code(indices)
            assert torch.allclose(x_recon, x_from_code, atol=1e-5), "decode_code must produce output matching forward pass"

        # 3-channel 32x32 image contract
        vqvae_rgb = VQVAE(in_channels=3, hidden_dim=32, embedding_dim=16, num_embeddings=64)
        x_rgb = torch.rand(2, 3, 32, 32)
        x_recon_rgb, _, _, indices_rgb = vqvae_rgb(x_rgb)
        assert x_recon_rgb.shape == (2, 3, 32, 32), f"Expected RGB shape (2, 3, 32, 32), got {x_recon_rgb.shape}"
        assert indices_rgb.shape == (2, 8, 8), f"Expected RGB indices shape (2, 8, 8), got {indices_rgb.shape}"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_straight_through_gradient_flow(model_mod):
    print("Running Test 2: Straight-Through Estimator & Gradient Audit...", end=" ")
    try:
        VectorQuantizer = model_mod.VectorQuantizer
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16, commitment_cost=0.25)

        z = torch.randn(2, 16, 7, 7, requires_grad=True)
        quantized, loss, perplexity, indices = quantizer(z)

        # Straight-through gradient check from downstream decoder loss
        dummy_decoder_loss = (quantized * 2.0).sum()
        dummy_decoder_loss.backward(retain_graph=True)

        assert z.grad is not None, "z must receive straight-through gradients from downstream decoder"
        # Gradient of quantized * 2.0 with respect to z through STE is identically 2.0
        assert torch.allclose(z.grad, torch.full_like(z.grad, 2.0)), "STE must pass downstream gradients directly (grad=2.0)"

        # Verify codebook embeddings receive gradients from vq_loss
        quantizer.zero_grad()
        loss.backward()
        assert quantizer.embedding.weight.grad is not None, "Codebook embeddings must receive gradients from vq_loss"
        assert not torch.allclose(
            quantizer.embedding.weight.grad, torch.zeros_like(quantizer.embedding.weight.grad)
        ), "Codebook gradients cannot be all zero"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_perplexity_and_quantization_contracts(model_mod):
    print("Running Test 3: Perplexity & Quantization Metric Contracts...", end=" ")
    try:
        VectorQuantizer = model_mod.VectorQuantizer
        quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=8)

        # Case 1: When all spatial vectors are identical to codebook vector 0
        code_0 = quantizer.embedding.weight[0:1].unsqueeze(-1).unsqueeze(-1)  # (1, 8, 1, 1)
        z_collapsed = code_0.repeat(4, 1, 4, 4)  # (4, 8, 4, 4)

        _, loss, perplexity, indices = quantizer(z_collapsed)
        assert torch.all(indices == 0), "All indices must map to code 0"
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), f"Loss on exact codebook match must be 0, got {loss.item()}"
        assert torch.isclose(perplexity, torch.tensor(1.0), atol=1e-3), f"Perplexity on single code must be 1.0, got {perplexity.item()}"

        # Case 2: Perplexity bounded by num_embeddings
        z_random = torch.randn(8, 8, 16, 16)
        _, _, perplexity_rand, _ = quantizer(z_random)
        assert 1.0 <= perplexity_rand.item() <= 16.05, f"Perplexity must be in [1, K], got {perplexity_rand.item()}"

        print("[PASS]")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def test_reconstruction_overfit_gate(model_mod):
    print("Running Test 4: Discrete Reconstruction Overfit Gate...", end=" ")
    try:
        torch.manual_seed(42)
        VQVAE = model_mod.VQVAE

        vqvae = VQVAE(in_channels=1, hidden_dim=32, embedding_dim=16, num_embeddings=32, commitment_cost=0.25)
        optimizer = optim.Adam(vqvae.parameters(), lr=1e-2)

        # 4 distinct synthetic visual patterns
        x_target = torch.zeros(4, 1, 28, 28)
        x_target[0, 0, :, 12:16] = 1.0  # Vertical stripe
        x_target[1, 0, 12:16, :] = 1.0  # Horizontal stripe
        x_target[2, 0, 8:20, 8:20] = 1.0  # Center block
        x_target[3, 0, :, :] = 0.5  # Uniform gray

        initial_loss = None
        final_loss = None

        for step in range(60):
            optimizer.zero_grad()
            x_recon, vq_loss, perplexity, indices = vqvae(x_target)
            recon_loss = F.mse_loss(x_recon, x_target)
            total_loss = recon_loss + vq_loss

            if step == 0:
                initial_loss = total_loss.item()

            total_loss.backward()
            optimizer.step()

        final_loss = total_loss.item()
        assert final_loss < initial_loss * 0.35, f"Expected loss to decrease significantly: initial={initial_loss:.4f}, final={final_loss:.4f}"
        print(f"[PASS] (Loss {initial_loss:.4f} -> {final_loss:.4f})")
    except Exception as e:
        print(f"[FAIL]: {e}")
        raise e


def main():
    model_mod = get_model_module()
    test_shapes_and_contracts(model_mod)
    test_straight_through_gradient_flow(model_mod)
    test_perplexity_and_quantization_contracts(model_mod)
    test_reconstruction_overfit_gate(model_mod)
    print("\nAll VQ-VAE unit tests passed successfully!")


if __name__ == "__main__":
    main()
