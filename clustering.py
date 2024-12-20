import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import matplotlib.colors as mcolors
import plotly.io as pio
import joblib

pio.renderers.default = 'browser'

def hex_to_rgb(color_hex):
    rgb = mcolors.hex2color(color_hex)  
    return tuple(int(255 * val) for val in rgb)

fixed_colors = px.colors.qualitative.Plotly

def plot_clusters_with_curved_boundaries(X, labels, title):
    unique_labels = np.unique(labels)

    colors = [fixed_colors[i % len(fixed_colors)] for i in range(len(unique_labels))]
    contour_colors = []

    fig = go.Figure()

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'rgb(0, 0, 0)'  # Шум для DBSCAN
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            marker=dict(color=color, size=7, line=dict(width=1)),
            name=f"Cluster {label}" if label != -1 else "Noise"
        ))

        if color.startswith('#'):
            r, g, b = hex_to_rgb(color)
        else:
            r, g, b = [int(c) for c in color[4:-1].split(',')]

        contour_colors.append(f'rgba({r},{g},{b},0.2)')

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, labels)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    for idx, color in enumerate(contour_colors):
        mask = (Z == unique_labels[idx])
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=mask.astype(int),
            colorscale=[[0, 'rgba(255,255,255,0)'], [1, color]],
            opacity=0.3,
            line_width=0,
            showscale=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True
    )

    fig.show()

X_circles = joblib.load('X_circles.pkl')

kmeans_circles = KMeans(n_clusters=2, random_state=42)
kmeans_labels_circles = kmeans_circles.fit_predict(X_circles)
plot_clusters_with_curved_boundaries(X_circles, kmeans_labels_circles, "KMeans Clustering with Circles")

dbscan_circles = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels_circles = dbscan_circles.fit_predict(X_circles)
plot_clusters_with_curved_boundaries(X_circles, dbscan_labels_circles, "DBSCAN Clustering with Circles")

X_blobs = joblib.load('X_blobs.pkl')

kmeans_blobs = KMeans(n_clusters=3, random_state=42)
kmeans_labels_blobs = kmeans_blobs.fit_predict(X_blobs)
plot_clusters_with_curved_boundaries(X_blobs, kmeans_labels_blobs, "KMeans Clustering with Blobs")

dbscan_blobs = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels_blobs = dbscan_blobs.fit_predict(X_blobs)
plot_clusters_with_curved_boundaries(X_blobs, dbscan_labels_blobs, "DBSCAN Clustering with Blobs")

X_moons = joblib.load('X_moons.pkl')

kmeans_circles = KMeans(n_clusters=2, random_state=42)
kmeans_labels_circles = kmeans_circles.fit_predict(X_moons)
plot_clusters_with_curved_boundaries(X_moons, kmeans_labels_circles, "KMeans Clustering with Moons")

dbscan_circles = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels_circles = dbscan_circles.fit_predict(X_moons)
plot_clusters_with_curved_boundaries(X_moons, dbscan_labels_circles, "DBSCAN Clustering with Moons")