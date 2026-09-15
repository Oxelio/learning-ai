import torch
import torch.nn as nn


def main():
    # Permet d'obtenir les mêmes valeurs initiales à chaque exécution.
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("PyTorch :", torch.__version__)
    print("Device :", device)

    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
        print("HIP :", torch.version.hip)

    # ------------------------------------------------------------
    # Données
    #
    # On veut apprendre la fonction :
    #
    # y = 2x + 1
    # ------------------------------------------------------------

    x = torch.arange(
        0,
        10,
        dtype=torch.float32
    ).reshape(-1, 1)

    y = 2 * x + 1

    x = x.to(device)
    y = y.to(device)

    # ------------------------------------------------------------
    # Modèle
    #
    # nn.Linear(1, 1) représente :
    #
    # y = weight * x + bias
    # ------------------------------------------------------------

    model = nn.Linear(1, 1).to(device)

    # ------------------------------------------------------------
    # Fonction de coût
    #
    # Mean Squared Error :
    #
    # moyenne((prediction - vraie_valeur)²)
    # ------------------------------------------------------------

    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------
    # Optimiseur
    # ------------------------------------------------------------

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
    )

    # ------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------

    epochs = 1000

    for epoch in range(epochs):
        # 1. Forward pass
        prediction = model(x)

        # 2. Calcul de l'erreur
        loss = loss_fn(prediction, y)

        # 3. Suppression des anciens gradients
        optimizer.zero_grad()

        # 4. Calcul des gradients
        loss.backward()

        # 5. Mise à jour des paramètres
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss.item():.6f}"
            )

    # ------------------------------------------------------------
    # Résultat
    # ------------------------------------------------------------

    print()
    print("Entraînement terminé")
    print("--------------------")

    print("Poids appris :", model.weight.item())
    print("Biais appris :", model.bias.item())

    # ------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------

    model.eval()

    with torch.no_grad():
        input_value = torch.tensor(
            [[20.0]],
            device=device
        )

        prediction = model(input_value)

    print()
    print("x = 20")
    print("Prédiction :", prediction.item())
    print("Valeur attendue :", 41.0)


if __name__ == "__main__":
    main()