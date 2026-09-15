import time
import torch


def main():
    print("PyTorch :", torch.__version__)
    print("CUDA :", torch.version.cuda)
    print("HIP :", torch.version.hip)

    if not torch.cuda.is_available():
        print("Aucun GPU compatible détecté.")
        return

    device = torch.device("cuda")

    print("GPU :", torch.cuda.get_device_name(0))

    # Matrices suffisamment grosses pour solliciter le GPU
    size = 4096

    print(f"\nCréation de matrices {size}x{size}...")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    print("Warmup...")
    _ = a @ b
    torch.cuda.synchronize()

    print("Benchmark multiplication matricielle...")

    start = time.perf_counter()

    c = a @ b

    # Les opérations GPU sont asynchrones.
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"Temps : {elapsed:.4f} s")
    print("Résultat :", c.shape)
    print("Device :", c.device)

    print("\nMémoire GPU :")
    print(
        f"Allouée : "
        f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    print(
        f"Réservée : "
        f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    )


if __name__ == "__main__":
    main()