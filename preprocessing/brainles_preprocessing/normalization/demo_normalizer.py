import numpy as np
import matplotlib.pyplot as plt
from .windowing_normalizer import WindowingNormalizer
from .percentile_normalizer import PercentileNormalizer


if __name__ == "__main__":
    image = np.random.uniform(-100, 100, size=(512, 512))

    # Windowing method
    window_normalizer = WindowingNormalizer(center=50, width=200)
    windowed_image = window_normalizer.normalize(image)

    # Percentile method
    percentile_normalizer = PercentileNormalizer(
        lower_percentile=10, upper_percentile=90, lower_limit=0, upper_limit=1
    )
    percentile_normalized_image = percentile_normalizer.normalize(image)

    # Visualization (requires matplotlib)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray", vmin=-1000, vmax=1000)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(windowed_image, cmap="gray", vmin=-1000, vmax=1000)
    plt.title("Windowed Normalization")

    plt.subplot(1, 3, 3)
    plt.imshow(percentile_normalized_image, cmap="gray")
    plt.title("Percentile Normalization")

    plt.show()
