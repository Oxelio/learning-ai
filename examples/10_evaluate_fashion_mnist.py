import torch
import torch.nn as nn

from fashion_mnist import (
    CLASS_NAMES,
    FashionCNN,
    create_dataloaders,
    create_datasets,
    evaluate_classifier,
    most_common_confusions,
    train_one_epoch,
)
from fashion_mnist.evaluation import (
    class_accuracies,
)


def print_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> None:
    print()
    print("Confusion matrix")
    print("----------------")

    print(
        " " * 14,
        end="",
    )

    for index in range(len(CLASS_NAMES)):
        print(
            f"{index:>6}",
            end="",
        )

    print()

    for row_index, row in enumerate(
        confusion_matrix
    ):
        print(
            f"{row_index:2d} "
            f"{CLASS_NAMES[row_index][:10]:10s}",
            end="",
        )

        for value in row:
            print(
                f"{value.item():6d}",
                end="",
            )

        print()


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("PyTorch :", torch.__version__)
    print("Device :", device)

    if torch.cuda.is_available():
        print(
            "GPU :",
            torch.cuda.get_device_name(0),
        )
        print(
            "HIP :",
            torch.version.hip,
        )

    train_dataset, test_dataset = (
        create_datasets()
    )

    train_loader, test_loader = (
        create_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=128,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    )

    model = FashionCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

    epochs = 10

    print()
    print("Training")
    print("--------")

    for epoch in range(epochs):
        loss, accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    result = evaluate_classifier(
        model,
        test_loader,
        device,
    )

    print()
    print("Evaluation")
    print("----------")
    print(
        f"Test accuracy : "
        f"{result.accuracy * 100:.2f}%"
    )
    print(
        "Correct predictions :",
        int(
            result.confusion_matrix
            .diag()
            .sum()
            .item()
        ),
    )
    print(
        "Incorrect predictions :",
        len(result.errors),
    )

    print_confusion_matrix(
        result.confusion_matrix
    )

    accuracies = class_accuracies(
        result.confusion_matrix
    )

    print()
    print("Accuracy by class")
    print("-----------------")

    for class_index, accuracy in enumerate(
        accuracies
    ):
        correct = int(
            result.confusion_matrix[
                class_index,
                class_index,
            ].item()
        )

        total = int(
            result.confusion_matrix[
                class_index
            ].sum().item()
        )

        print(
            f"{class_index:2d} | "
            f"{CLASS_NAMES[class_index]:12s} | "
            f"{accuracy * 100:6.2f}% | "
            f"{correct:4d}/{total:4d}"
        )

    print()
    print("Most common confusions")
    print("----------------------")

    for (
        occurrences,
        expected,
        predicted,
    ) in most_common_confusions(
        result.confusion_matrix,
        count=10,
    ):
        print(
            f"{CLASS_NAMES[expected]:12s} "
            f"-> "
            f"{CLASS_NAMES[predicted]:12s} "
            f": {occurrences}"
        )

    print()
    print("Most confident errors")
    print("---------------------")

    errors = sorted(
        result.errors,
        key=lambda error: error.confidence,
        reverse=True,
    )

    for error in errors[:10]:
        print(
            f"expected="
            f"{CLASS_NAMES[error.expected]:12s} | "
            f"predicted="
            f"{CLASS_NAMES[error.predicted]:12s} | "
            f"confidence="
            f"{error.confidence * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()
