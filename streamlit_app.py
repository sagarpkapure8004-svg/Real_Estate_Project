import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go

sns.set_style("whitegrid")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['sale_price_numeric'] = (
        df['sale_price'].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    df['country'] = df['country'].astype(str).str.strip()
    df['unit_category'] = df['unit_category'].astype(str).str.strip()
    df['acquisition'] = df['acquisition'].astype(str).str.strip()
    return df

@st.cache_data
def prepare_features(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, np.ndarray]:
    df_clean = df[features].dropna().copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean)
    return df_clean, scaled

@st.cache_data
def run_kmeans(scaled: np.ndarray, n_clusters: int) -> tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    return labels, model

@st.cache_data
def run_hierarchical(scaled: np.ndarray, n_clusters: int, linkage_method: str) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(scaled)
    return labels

@st.cache_data
def compute_elbow(scaled: np.ndarray, max_k: int = 10) -> pd.DataFrame:
    inertia, silhouette, davies = [], [], []
    ks = list(range(2, max_k + 1))
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        inertia.append(model.inertia_)
        silhouette.append(silhouette_score(scaled, labels))
        davies.append(davies_bouldin_score(scaled, labels))
    return pd.DataFrame({
        'k': ks,
        'inertia': inertia,
        'silhouette': silhouette,
        'davies_bouldin': davies,
    })

@st.cache_data
def compute_pca(scaled: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2)
    return pca.fit_transform(scaled), pca

def plot_dendrogram(scaled: np.ndarray, sample_size: int = 60):
    idx = np.random.choice(range(len(scaled)), size=min(sample_size, len(scaled)), replace=False)
    sample = scaled[idx]
    linkage_matrix = linkage(sample, method='ward')
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage_matrix, ax=ax, color_threshold=0)
    ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    return fig

def plot_pca_scatter(pca_features: np.ndarray, labels: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        pca_features[:, 0], pca_features[:, 1],
        c=labels, cmap='tab10', edgecolor='k', alpha=0.7, s=70
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    legend1 = ax.legend(*scatter.legend_elements(), title='Cluster')
    ax.add_artist(legend1)
    return fig

def plot_profile_bar(df: pd.DataFrame, label_column: str, features: list, title_prefix: str):
    summary = df.groupby(label_column)[features].mean().round(2)
    fig, axes = plt.subplots(1, len(features), figsize=(16, 4))
    colors = ['#440154', '#31688e', '#35b779']
    for i, feature in enumerate(features):
        axes[i].bar(summary.index.astype(str), summary[feature], color=colors[: len(summary)] )
        axes[i].set_title(f'{title_prefix}: {feature.replace("_", " ").title()}')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel('Mean Value')
        axes[i].grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig

def plot_3d_scatter(df: pd.DataFrame, labels: np.ndarray, title: str):
    fig = px.scatter_3d(
        df,
        x='floor_area_sqft',
        y='sale_price_numeric',
        z='satisfaction',
        color=labels.astype(str),
        labels={
            'floor_area_sqft': 'Floor Area (sqft)',
            'sale_price_numeric': 'Sale Price',
            'satisfaction': 'Satisfaction',
        },
        title=title,
        width=900,
        height=700,
    )
    fig.update_traces(marker=dict(size=4))
    return fig

@st.cache_data
def plot_country_cluster_map(df: pd.DataFrame) -> go.Figure:
    country_summary = (
        df.groupby('country')
          .agg(count=('country', 'size'), top_regions=('region', lambda x: ', '.join(pd.Series(x).value_counts().head(3).index.tolist())))
          .reset_index()
    )
    if country_summary.empty:
        return go.Figure()

    fig = px.choropleth(
        country_summary,
        locations='country',
        locationmode='country names',
        color='count',
        hover_name='country',
        hover_data={'top_regions': True, 'count': True},
        projection='natural earth',
        title='Country Clustering Distribution',
        color_continuous_scale='Plasma',
        labels={'count': 'Property Count'},
        height=650,
    )
    fig.update_traces(marker_line_color='black', marker_line_width=0.8, selector=dict(type='choropleth'))
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='LightGray',
            landcolor='rgb(245, 245, 245)',
            lakecolor='LightBlue',
            bgcolor='rgba(255,255,255,0.98)',
            projection_type='natural earth',
        ),
        coloraxis_colorbar=dict(
            title='Property Count',
            thickness=18,
            ticklen=3,
            ticks='outside',
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig

st.set_page_config(page_title='Real Estate Clustering App', layout='wide')

st.markdown(
    '''
    <style>
    .hero {
        padding: 3rem 1rem 2rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 48%, #334155 100%);
        color: #f8fafc;
        border-radius: 18px;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.35);
        margin-bottom: 1.5rem;
        animation: fadeInUp 1s ease-out;
    }
    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.02em;
    }
    .hero p {
        font-size: 1.15rem;
        color: #cbd5e1;
        max-width: 760px;
        line-height: 1.7;
    }
    .scroll-down {
        margin: 2rem auto 0;
        width: 32px;
        height: 48px;
        border: 2px solid #cbd5e1;
        border-radius: 24px;
        position: relative;
    }
    .scroll-down::before {
        content: '';
        position: absolute;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #cbd5e1;
        animation: scroll 1.8s infinite;
    }
    @keyframes scroll {
        0% { opacity: 0; transform: translate(-50%, -8px); }
        50% { opacity: 1; transform: translate(-50%, 8px); }
        100% { opacity: 0; transform: translate(-50%, 24px); }
    }
    .section-title {
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e2e8f0;
        animation: fadeInUp 0.8s ease-out;
    }
    .section-card,
    .chart-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.18);
        animation: fadeInScale 0.9s ease-out;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 16px 35px rgba(15, 23, 42, 0.1);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 22px 40px rgba(15, 23, 42, 0.16);
    }
    .custom-card {
        animation: fadeInScale 1s ease-out;
    }
    html, body {
        scroll-behavior: smooth;
    }
    .floating-anchor {
        position: fixed;
        right: 28px;
        bottom: 26px;
        width: 48px;
        height: 48px;
        background: rgba(15, 23, 42, 0.9);
        color: #f8fafc;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.24);
        z-index: 9999;
        transition: transform 0.25s ease, background 0.25s ease;
    }
    .floating-anchor:hover {
        transform: translateY(-3px);
        background: #334155;
    }
    .floating-anchor span {
        font-size: 1.2rem;
        animation: floatUp 1.8s infinite ease-in-out;
    }
    @keyframes floatUp {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: translateY(16px) scale(0.98);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    </style>
    <div id="top" class="hero">
        <h1>Real Estate Clustering Explorer</h1>
        <p>Analyze property sales using real estate clustering models, live filters, and interactive dashboards for faster market insight.</p>
        <div class="scroll-down"></div>
    </div>
    ''',
    unsafe_allow_html=True,
)

DATA_PATH = 'Real_Estate_Project.csv'
FEATURES = ['floor_area_sqft', 'sale_price_numeric', 'satisfaction']

with st.spinner('Loading property dataset...'):
    df = load_data(DATA_PATH)

available_countries = sorted(df['country'].dropna().unique())
available_types = sorted(df['unit_category'].dropna().unique())
available_acquisitions = sorted(df['acquisition'].dropna().unique())

with st.sidebar.expander('Data Filters', expanded=True):
    selected_countries = st.multiselect('Countries', available_countries, default=available_countries)
    selected_types = st.multiselect('Property Types', available_types, default=available_types)
    selected_acquisition = st.multiselect('Acquisition', available_acquisitions, default=available_acquisitions)
    price_min = int(df['sale_price_numeric'].min())
    price_max = int(df['sale_price_numeric'].max())
    selected_range = st.slider('Sale price range', price_min, price_max, (price_min, price_max), step=1000)
    show_raw = st.checkbox('Show filtered raw dataset')

filters = (
    df['country'].isin(selected_countries)
    & df['unit_category'].isin(selected_types)
    & df['acquisition'].isin(selected_acquisition)
    & df['sale_price_numeric'].between(selected_range[0], selected_range[1])
)

df_filtered = df.loc[filters].copy()

st.sidebar.header('Clustering Controls')
algorithm = st.sidebar.selectbox('Algorithm', ['K-means', 'Hierarchical', 'Both'])
num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=8, value=3)
linkage_method = st.sidebar.selectbox('Hierarchical Linkage', ['ward', 'complete', 'average', 'single'])
show_elbow = st.sidebar.checkbox('Show K-means elbow / metric charts', value=True)
show_3d = st.sidebar.checkbox('Show interactive 3D scatter plots', value=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Properties', len(df_filtered), delta=f'{len(df_filtered) - len(df):+d}')
col2.metric('Median Sale Price', f"${int(df_filtered['sale_price_numeric'].median()):,}")
col3.metric('Avg Floor Area', f"{int(df_filtered['floor_area_sqft'].mean()):,} sqft")
col4.metric('Avg Satisfaction', f"{df_filtered['satisfaction'].mean():.2f}")

st.markdown('---')

st.markdown('<div class="section-title"><h2>Market Snapshot</h2></div>', unsafe_allow_html=True)
row1, row2 = st.columns([2, 1])
with row1:
    st.markdown('<div class="section-card"><h3>Filtered Details</h3></div>', unsafe_allow_html=True)
    st.write(df_filtered[['unit_category', 'country', 'acquisition', 'floor_area_sqft', 'sale_price_numeric', 'satisfaction']].head(12))
with row2:
    st.markdown('<div class="section-card"><h3>Quick insight</h3></div>', unsafe_allow_html=True)
    st.write(f'*Filtered dataset contains {len(df_filtered)} records from {len(selected_countries)} countries.*')
    st.write(f'*Average sale price is ${df_filtered["sale_price_numeric"].mean():,.0f}.*')
    st.write(f'*Average satisfaction is {df_filtered["satisfaction"].mean():.2f}.*')

if show_raw:
    st.markdown('<div class="section-card"><h3>Raw Filtered Dataset</h3></div>', unsafe_allow_html=True)
    st.dataframe(df_filtered.reset_index(drop=True))

country_map_fig = plot_country_cluster_map(df_filtered)
if not df_filtered.empty and country_map_fig.data:
    st.markdown('<div class="section-title"><h2>Country Cluster Map</h2></div>', unsafe_allow_html=True)
    st.plotly_chart(country_map_fig, width='stretch')

if df_filtered.empty:
    st.warning('The filtered dataset is empty. Please expand your filter selection to view clustering results.')
else:
    df_clean, scaled_features = prepare_features(df_filtered, FEATURES)
    if len(df_clean) < 10:
        st.warning('Not enough rows in the filtered dataset to perform clustering reliably. Adjust filters to use the clustering dashboard.')
    else:
        if show_elbow:
            st.markdown('<div class="section-title"><h2>K-means Model Calibration</h2></div>', unsafe_allow_html=True)
            elbow_df = compute_elbow(scaled_features, max_k=10)
            fig, ax = plt.subplots(1, 3, figsize=(18, 4))
            ax[0].plot(elbow_df['k'], elbow_df['inertia'], marker='o')
            ax[0].set_title('Inertia')
            ax[0].set_xlabel('k')
            ax[0].grid(True, alpha=0.3)
            ax[1].plot(elbow_df['k'], elbow_df['silhouette'], marker='o', color='green')
            ax[1].set_title('Silhouette Score')
            ax[1].set_xlabel('k')
            ax[1].grid(True, alpha=0.3)
            ax[2].plot(elbow_df['k'], elbow_df['davies_bouldin'], marker='o', color='red')
            ax[2].set_title('Davies-Bouldin Score')
            ax[2].set_xlabel('k')
            ax[2].grid(True, alpha=0.3)
            st.markdown('<div class="chart-card"></div>', unsafe_allow_html=True)
            st.pyplot(fig)

        pca_features, pca_model = compute_pca(scaled_features)

        if algorithm in ['K-means', 'Both']:
            st.markdown('<div class="section-title"><h2>K-means Clustering Results</h2></div>', unsafe_allow_html=True)
            labels, kmeans_model = run_kmeans(scaled_features, num_clusters)
            df_clean['KMeans_Cluster'] = labels
            st.write('Silhouette Score:', f'{silhouette_score(scaled_features, labels):.4f}')
            st.write('Davies-Bouldin Score:', f'{davies_bouldin_score(scaled_features, labels):.4f}')

            st.markdown('<div class="chart-card"></div>', unsafe_allow_html=True)
            st.pyplot(plot_pca_scatter(pca_features, labels, f'K-means Clusters (k={num_clusters})'))

            if show_3d:
                st.plotly_chart(plot_3d_scatter(df_clean, labels, 'K-means 3D Cluster View'), width='stretch')

            st.markdown('<div class="chart-card"></div>', unsafe_allow_html=True)
            st.pyplot(plot_profile_bar(df_clean, 'KMeans_Cluster', FEATURES, 'K-means'))

        if algorithm in ['Hierarchical', 'Both']:
            st.markdown('<div class="section-title"><h2>Hierarchical Clustering Results</h2></div>', unsafe_allow_html=True)
            labels_hier = run_hierarchical(scaled_features, num_clusters, linkage_method)
            df_clean['Hierarchical_Cluster'] = labels_hier
            st.write('Silhouette Score:', f'{silhouette_score(scaled_features, labels_hier):.4f}')
            st.write('Davies-Bouldin Score:', f'{davies_bouldin_score(scaled_features, labels_hier):.4f}')

            st.markdown('<div class="chart-card"></div>', unsafe_allow_html=True)
            st.pyplot(plot_pca_scatter(pca_features, labels_hier, f'Hierarchical Clusters (k={num_clusters}, {linkage_method})'))

            if show_3d:
                st.plotly_chart(plot_3d_scatter(df_clean, labels_hier, 'Hierarchical 3D Cluster View'), width='stretch')

            st.markdown('<div class="chart-card"></div>', unsafe_allow_html=True)
            st.pyplot(plot_profile_bar(df_clean, 'Hierarchical_Cluster', FEATURES, 'Hierarchical'))

            if algorithm != 'K-means':
                st.markdown('<div class="section-title"><h2>Hierarchical Dendrogram</h2></div>', unsafe_allow_html=True)
                st.pyplot(plot_dendrogram(scaled_features))

    st.sidebar.markdown('### Data Details')
    st.sidebar.write(f'Rows: {len(df_filtered)}')
    st.sidebar.write('Features used for clustering:')
    st.sidebar.write(FEATURES)
    st.sidebar.write('Countries selected:')
    st.sidebar.write(', '.join(selected_countries[:10]) + ('...' if len(selected_countries) > 10 else ''))

    st.markdown('---')
    st.markdown('**Usage:** Run this app with `streamlit run streamlit_app.py` from the project folder.')
    st.markdown('<a class="floating-anchor" href="#top" title="Back to top"><span>↑</span></a>', unsafe_allow_html=True)
