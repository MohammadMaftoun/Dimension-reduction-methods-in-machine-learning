import numpy as np
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
from sklearn.neighbors import NearestNeighbors
class UMAP:
    def __init__(self, n_neighbors=15, n_components=2, min_dist=0.1, n_epochs=500, learning_rate=1.0):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def fit_transform(self, X):
        # Step 1: Find k-nearest neighbors
        print("Finding nearest neighbors...")
        knn_indices, knn_dists = self._compute_knn(X)

        # Step 2: Compute fuzzy graph (probability distribution over edges)
        print("Constructing fuzzy simplicial set...")
        graph = self._construct_fuzzy_graph(knn_indices, knn_dists)

        # Step 3: Compute spectral embedding for initialization
        print("Computing spectral embedding...")
        embedding = self._spectral_embedding(graph)

        # Step 4: Optimize the embedding layout
        print("Optimizing low-dimensional representation...")
        embedding = self._optimize_embedding(graph, embedding)

        return embedding

    def _compute_knn(self, X):
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        return indices, distances

    def _construct_fuzzy_graph(self, knn_indices, knn_dists):
        n_samples = knn_indices.shape[0]
        rows, cols, values = [], [], []

        for i in range(n_samples):
            for j in range(1, self.n_neighbors):  # Ignore self-loop
                neighbor_idx = knn_indices[i, j]
                distance = knn_dists[i, j]
                weight = np.exp(-distance)  # Exponential decay
                rows.append(i)
                cols.append(neighbor_idx)
                values.append(weight)

        graph = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(n_samples, n_samples))
        return graph.maximum(graph.T)  # Ensure symmetry

    def _spectral_embedding(self, graph):
        # Compute normalized Laplacian
        laplacian = scipy.sparse.csgraph.laplacian(graph, normed=True)
        _, eigenvectors = scipy.sparse.linalg.eigsh(laplacian, k=self.n_components + 1, which='SM')
        return eigenvectors[:, 1:]  # Ignore the first trivial eigenvector

    def _optimize_embedding(self, graph, embedding):
        n_samples = embedding.shape[0]
        for epoch in range(self.n_epochs):
            # Compute forces based on attractive and repulsive terms
            for i in range(n_samples):
                for j in graph[i].indices:
                    # Attractive force
                    diff = embedding[i] - embedding[j]
                    dist_sq = np.dot(diff, diff)
                    attraction = -2 * (dist_sq - self.min_dist**2) * diff
                    embedding[i] += self.learning_rate * attraction

                    # Repulsive force (avoid collapsing)
                    if dist_sq > 0:
                        repulsion = 2 * (self.min_dist**2 - dist_sq) * diff
                        embedding[i] -= self.learning_rate * repulsion

            # Decay learning rate
            self.learning_rate *= 0.99

            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{self.n_epochs} complete")

        return embedding

# Another alternative is the UMAP package
# It starts with installing via the following code:
!pip install umap-learn
import umap
# building Embedding 
embedding = umap.UMAP(n_neighbors=5).fit_transform(data)
# plot it using matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*embedding.T, s=0.3, c=target, cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(10))
cbar.set_ticklabels(classes)
plt.title('Embedded Data via UMAP');
