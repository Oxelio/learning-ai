import torch


def main():
    print("PyTorch :", torch.__version__)

    # ------------------------------------------------------------
    # 1. Tensor avec suivi des gradients
    # ------------------------------------------------------------

    x = torch.tensor(
        3.0,
        requires_grad=True,
    )

    print("\n--- Simple derivative ---")
    print("x :", x)

    y = x**2

    print("y = x² :", y)

    # Calcule dy/dx
    y.backward()

    print("dy/dx :", x.grad)

    # Mathématiquement :
    #
    # y = x²
    # dy/dx = 2x
    #
    # pour x = 3 :
    #
    # dy/dx = 6

    # ------------------------------------------------------------
    # 2. Remise à zéro du gradient
    # ------------------------------------------------------------

    x.grad.zero_()

    print("\nGradient reset :", x.grad)

    # ------------------------------------------------------------
    # 3. Fonction un peu plus complexe
    # ------------------------------------------------------------

    # y = x³ + 2x² + 5x + 1
    y = x**3 + 2 * x**2 + 5 * x + 1

    y.backward()

    print("\n--- Polynomial derivative ---")
    print("x :", x.item())
    print("y :", y.item())
    print("dy/dx :", x.grad.item())

    # Mathématiquement :
    #
    # y = x³ + 2x² + 5x + 1
    #
    # dy/dx = 3x² + 4x + 5
    #
    # pour x = 3 :
    #
    # 3 * 9 + 12 + 5 = 44

    # ------------------------------------------------------------
    # 4. Plusieurs variables
    # ------------------------------------------------------------

    a = torch.tensor(
        2.0,
        requires_grad=True,
    )

    b = torch.tensor(
        4.0,
        requires_grad=True,
    )

    z = a**2 + a * b + b**2

    z.backward()

    print("\n--- Multiple variables ---")
    print("z :", z.item())

    print("dz/da :", a.grad.item())
    print("dz/db :", b.grad.item())

    # Mathématiquement :
    #
    # z = a² + ab + b²
    #
    # dz/da = 2a + b
    # dz/db = a + 2b
    #
    # a = 2
    # b = 4
    #
    # dz/da = 8
    # dz/db = 10

    # ------------------------------------------------------------
    # 5. Gradient sur un vecteur
    # ------------------------------------------------------------

    values = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    # Somme des carrés
    result = (values**2).sum()

    result.backward()

    print("\n--- Vector gradient ---")
    print("values :", values)
    print("result :", result.item())
    print("gradient :", values.grad)

    # result = x₁² + x₂² + x₃²
    #
    # gradient :
    #
    # [
    #   2x₁,
    #   2x₂,
    #   2x₃
    # ]
    #
    # donc :
    #
    # [2, 4, 6]

    # ------------------------------------------------------------
    # 6. Désactiver Autograd
    # ------------------------------------------------------------

    print("\n--- torch.no_grad() ---")

    with torch.no_grad():
        prediction = values * 10

    print("prediction :", prediction)
    print("requires_grad :", prediction.requires_grad)


if __name__ == "__main__":
    main()