from flask import Flask, request, jsonify, render_template
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import matplotlib.colors as mcolors
import plotly.utils
import joblib
import json
import redis

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, db=0)

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
            color = 'rgb(0, 0, 0)'
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
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_dbscan_parameters', methods=['POST'])
def update_dbscan_parameters():
    data = request.get_json()
    eps_value = data['eps_value']
    minpts_value = data['minpts_value']
    dataset = data['dbscan_dataset']

    redis_key = f"dbscan:{dataset}:{eps_value}:{minpts_value}"

    dbscan_graph = r.get(redis_key)
    if not dbscan_graph:
        if dataset == "1":
            X = joblib.load('X_circles.pkl')
        elif dataset == "2":
            X = joblib.load('X_blobs.pkl')
        else:
            X = joblib.load('X_moons.pkl')

        dbscan= DBSCAN(eps=float(eps_value), min_samples=int(minpts_value))
        dbscan_labels = dbscan.fit_predict(X)
        dbscan_graph = plot_clusters_with_curved_boundaries(X, dbscan_labels, f"DBSCAN Clustering")
        r.set(redis_key, dbscan_graph)
    else:
        dbscan_graph = dbscan_graph.decode('utf-8')
    return jsonify({'dbscan_graph': dbscan_graph})

@app.route('/update_kmeans_parameters', methods=['POST'])
def update_kmeans_parameters():
    data = request.get_json()
    k_value = data['k_value']
    dataset = data['kmeans_dataset']

    redis_key = f"kmeans:{dataset}:{k_value}"

    kmeans_graph = r.get(redis_key)
    if not kmeans_graph:
        if dataset == "1":
            X = joblib.load('X_circles.pkl')
        elif dataset == "2":
            X = joblib.load('X_blobs.pkl')
        else:
            X = joblib.load('X_moons.pkl')

        kmeans = KMeans(n_clusters=int(k_value), random_state=42)
        kmeans_labels = kmeans.fit_predict(X)

        kmeans_graph = plot_clusters_with_curved_boundaries(X, kmeans_labels, f"KMeans Clustering")
        r.set(redis_key, kmeans_graph)
    else:
        kmeans_graph = kmeans_graph.decode('utf-8')
    return jsonify({'kmeans_graph': kmeans_graph})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)