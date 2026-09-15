# Fashion-MNIST mini-project

This layer turns the learning examples into a small reusable ML workflow.

## Structure

```text
configs/
└── fashion_mnist.json

scripts/
├── train.py
├── evaluate.py
└── predict.py

src/
└── fashion_mnist/
    ├── checkpoint.py
    ├── config.py
    ├── inference.py
    └── visualization.py
```

The existing `data.py`, `models.py`, `training.py`, `evaluation.py`,
`constants.py`, and examples remain in place.

## Install local package

```powershell
python -m pip install -e .
```

## Train

```powershell
python .\scripts\train.py
```

Outputs:

```text
models/fashion_cnn_best.pt
outputs/training_history.json
```

## Evaluate

```powershell
python .\scripts\evaluate.py
```

Outputs:

```text
outputs/evaluation.json
outputs/confusion_matrix.png
```

## Predict an external image

```powershell
python .\scripts\predict.py .\my_image.png
```

For a white background / dark object image:

```powershell
python .\scripts\predict.py .\my_image.png --invert
```

Fashion-MNIST uses centered 28x28 grayscale clothing images. Ordinary photos
can perform poorly because they are outside the training distribution.

## Suggested commit

```powershell
git add configs scripts src
git commit -m "feat: add Fashion-MNIST training pipeline"
```
