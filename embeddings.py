from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.tabpfn import TabPFNClassifier, TabPFNEncoder

# X = np.random.rand(10, 5)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# clf = TabPFNClassifier()
# clf.fit(X_train, y_train)

# # prediction_probabilities = clf.predict_proba(X_test)
# # print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# predictions = clf.predict(X_test)
# print("Accuracy (fitted)", accuracy_score(y_test, predictions))


encoder = TabPFNEncoder()
encoder.init_model()

# print(encoder.model_)

embeddings = encoder.get_raw_embeddings(X)
print(embeddings.shape) # (569, 192)

y_pred = encoder.predict_in_context(X_train, y_train, X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy (raw):", acc)

exit()


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)

tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=0)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

def plot_2d(embeddings_2d, y, title):
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=y,
        cmap="Set1",
        alpha=0.75,
    )
    plt.colorbar(scatter, label="Class label")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    filename = "pca" if "PCA" in title else "tsne"
    plt.savefig(f"results/{filename}.png")

# plot_2d(embeddings_2d_pca, y, title="TabPFN Embeddings (PCA 2D)")
# plot_2d(embeddings_2d_tsne, y, title="TabPFN Embeddings (t-SNE 2D)")


pca = PCA(n_components=3)
embeddings_3d_pca = pca.fit_transform(embeddings)

tsne = TSNE(n_components=3, init="pca", perplexity=30, random_state=0)
embeddings_3d_tsne = tsne.fit_transform(embeddings)

def plot_3d(embeddings_3d, y, title):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=y,
        cmap="Set1",
        alpha=0.75,
    )
    plt.colorbar(scatter, label="Class label")
    plt.title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")

    filename = "pca" if "PCA" in title else "tsne"
    plt.savefig(f"results/{filename}_3d.png")

plot_3d(embeddings_3d_pca, y, title="TabPFN Embeddings (PCA 3D)")
plot_3d(embeddings_3d_tsne, y, title="TabPFN Embeddings (t-SNE 3D)")
