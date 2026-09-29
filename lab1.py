import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

N = 10
predictions = probability_model(x_test[:N])
predicted_labels = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(N, 3, figsize=(6, 1.5 * N))
fig.subplots_adjust(hspace=0.5)

axes[0, 0].set_title("Изображение", fontsize=12)
axes[0, 1].set_title("Распознано", fontsize=12)
axes[0, 2].set_title("Реально", fontsize=12)

for i in range(N):
    ax_img = axes[i, 0]
    ax_img.imshow(x_test[i], cmap='gray')
    ax_img.axis('off')

    ax_pred = axes[i, 1]
    ax_pred.text(0.5, 0.5, str(predicted_labels[i]),
                 ha='center', va='center', fontsize=14, fontweight='bold')
    ax_pred.axis('off')

    ax_true = axes[i, 2]
    ax_true.text(0.5, 0.5, str(y_test[i]),
                 ha='center', va='center', fontsize=14, fontweight='bold')
    ax_true.axis('off')

for ax in axes[0]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('table_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nТаблица сохранена в файл table_predictions.png")
