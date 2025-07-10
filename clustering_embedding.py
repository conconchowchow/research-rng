"""
Experiment to test repulsive clustering on text embeddings generated from LLMs.

This experiment uses embeddings of model responses instead of numerical scores.
The data format is: [(embedding_vector, model_name), ...]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
import json
from collections import Counter
import google.generativeai as genai
from dotenv import load_dotenv
import random

# Import clustering functions from the main clustering module
from clustering import (
    create_trial_folder, 
    repulsive_sampling,
    assign_to_clusters,
    dpp_sampling,
    random_sampling,
    nearest_sampling,
    get_embedding,
    configure_gemini_api,
    cosine_similarity
)

# Import model providers
from model_provider import ModelProvider, GeminiProvider, MockProvider

load_dotenv()

def embedding_distance(p1: Tuple[np.ndarray, str], p2: Tuple[np.ndarray, str]) -> float:
    """
    Distance between two points using cosine similarity of their embedding vectors.
    
    Args:
        p1, p2: Tuples of (embedding_vector, model_name)
        
    Returns:
        Distance value where 0 means identical and 2 means completely opposite
    """
    # Extract embeddings (first element of tuple)
    embedding1 = p1[0]
    embedding2 = p2[0]
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    
    # Convert similarity to distance: 1 - similarity gives us a distance
    # where 0 means identical (similarity = 1) and 2 means completely opposite (similarity = -1)
    distance_value = 1 - similarity
    
    return distance_value

def repulsive_sampling_embedding(ensemble_data: List[Tuple[np.ndarray, str]], min_distance: float = 0.5) -> List[Tuple[np.ndarray, str]]:
    """
    Repulsive sampling for embedding data: Select cluster centers that are at least min_distance apart.
    
    Args:
        ensemble_data: List of (embedding_vector, model_name) tuples
        min_distance: Minimum distance required between cluster centers
    
    Returns:
        List of cluster centers [(embedding_vector, model_name), ...]
    """
    centers = []
    remaining_points = ensemble_data.copy()
    
    while len(remaining_points) > 0:
        # Pick a random point from remaining_points
        center_idx = random.randint(0, len(remaining_points) - 1)
        center = remaining_points[center_idx]
        centers.append(center)
        
        # Remove the selected center from remaining points
        remaining_points.pop(center_idx)
        
        # Remove all points within min_distance of the selected center
        remaining_points = [p for p in remaining_points 
                          if embedding_distance(p, center) >= min_distance]
    
    return centers

def assign_to_clusters_embedding(ensemble_data: List[Tuple[np.ndarray, str]], 
                                centers: List[Tuple[np.ndarray, str]]) -> List[Tuple[Tuple[np.ndarray, str], Tuple[Tuple[np.ndarray, str], ...]]]:
    """
    Assign each data point to the nearest cluster center using embedding distance.
    
    Args:
        ensemble_data: All data points
        centers: Cluster centers
    
    Returns:
        List of (center, (assigned_points, ...)) tuples
    """
    clusters = [(center, ()) for center in centers]
    
    # Create a set of center model names for efficient lookup
    center_names = {center[1] for center in centers}
    
    for point in ensemble_data:
        if point[1] not in center_names:  # Don't assign centers to themselves (check by model name)
            # Find nearest center
            nearest_center_idx = 0
            min_dist = embedding_distance(point, centers[0])
            
            for i, center in enumerate(centers):
                dist = embedding_distance(point, center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_center_idx = i
            
            # Add point to nearest cluster
            center, existing_points = clusters[nearest_center_idx]
            clusters[nearest_center_idx] = (center, existing_points + (point,))
    
    return clusters

def random_sampling_embedding(ensemble_data: List[Tuple[np.ndarray, str]], k: int) -> List[Tuple[np.ndarray, str]]:
    """
    Random sampling for embedding data: Select k random points from the ensemble data.
    
    Args:
        ensemble_data: List of (embedding_vector, model_name) tuples
        k: Number of centers to select
    
    Returns:
        List of randomly selected cluster centers
    """
    if len(ensemble_data) == 0:
        return []
    
    if k >= len(ensemble_data):
        return ensemble_data.copy()
    
    return random.sample(ensemble_data, k)

def nearest_sampling_embedding(ensemble_data: List[Tuple[np.ndarray, str]], k: int) -> List[Tuple[np.ndarray, str]]:
    """
    Nearest sampling for embedding data: Select k cluster centers by picking the first point
    and then selecting the k-1 points that are closest to the first point.
    
    Args:
        ensemble_data: List of (embedding_vector, model_name) tuples
        k: Number of centers to select
    
    Returns:
        List of selected cluster centers
    """
    if len(ensemble_data) == 0:
        return []
    
    if k >= len(ensemble_data):
        return ensemble_data.copy()
    
    # Pick the first point as the initial center
    first_point = ensemble_data[0]
    centers = [first_point]
    
    # Sort remaining points by distance to the first point
    remaining_points = ensemble_data[1:]  # Exclude the first point
    sorted_by_distance = sorted(remaining_points, key=lambda x: embedding_distance(x, first_point))
    
    # Select the k-1 closest points to the first point
    nearest_points = sorted_by_distance[:k-1]
    centers.extend(nearest_points)
    
    return centers

def dpp_sampling_embedding(ensemble_data: List[Tuple[np.ndarray, str]], k: int = None, sigma: float = 0.5) -> List[Tuple[np.ndarray, str]]:
    """
    Determinantal Point Process (DPP) sampling for embedding data.
    
    Args:
        ensemble_data: List of (embedding_vector, model_name) tuples
        k: Number of points to sample (if None, uses variable-size DPP)
        sigma: Bandwidth parameter for the RBF kernel
    
    Returns:
        List of selected points that are diverse according to the DPP
    """
    n = len(ensemble_data)
    if n == 0:
        return []
    
    if k is None:
        k = max(1, n // 4)  # Default to selecting about 1/4 of the points
    
    if k >= n:
        return ensemble_data.copy()
    
    # Create kernel matrix using RBF kernel based on embedding similarity
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Use cosine similarity as the basis for the kernel
            similarity = cosine_similarity(ensemble_data[i][0], ensemble_data[j][0])
            # Convert to distance and apply RBF kernel
            distance_val = 1 - similarity
            K[i, j] = np.exp(-distance_val ** 2 / (2 * sigma ** 2))
    
    # Use the DPP sampling from the original clustering module
    from clustering import sample_dpp
    from scipy.linalg import eigh
    
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = eigh(K)
    eigenvals = np.real(eigenvals)
    eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
    
    # Rescale eigenvalues to get expected size k
    if np.sum(eigenvals) > 0:
        scale_factor = k / np.sum(eigenvals / (1 + eigenvals))
        eigenvals = eigenvals * scale_factor
    
    # Sample from the DPP
    selected_indices = sample_dpp(eigenvals, eigenvecs, k)
    
    return [ensemble_data[i] for i in selected_indices]

def generate_test_responses(provider: ModelProvider, use_api: bool = True) -> Tuple[List[Tuple[np.ndarray, str]], Dict[str, str], Dict[str, str]]:
    """
    Generate test responses on different topics and convert to embedding format.
    
    Args:
        provider: Model provider to use for generating responses
        use_api: Whether to use the API or mock responses
    
    Returns:
        Tuple of (List of (embedding_vector, descriptive_model_name) tuples, Dict mapping model_name to prompt, Dict mapping model_name to response text)
    """
    # Define prompts for different topics
    architecture_prompt = "Describe the key characteristics of Gothic architecture in 2-3 sentences."
    constitution_prompt = "Explain the main purpose and structure of the US Constitution in 2-3 sentences."
    
    test_data = []
    prompt_mapping = {}  # Maps model_name to the prompt used
    response_mapping = {}  # Maps model_name to the actual response text
    
    # Generate architecture responses
    for i in range(4):  # Generate 4 architecture responses
        if use_api:
            response = provider.generate_response(architecture_prompt)
        else:
            # Use mock responses for testing
            mock_responses = [
                "Gothic architecture is characterized by pointed arches, ribbed vaults, and flying buttresses. These elements create soaring vertical spaces and allow for large windows with intricate tracery.",
                "Baroque architecture features dramatic use of light and shadow, ornate decoration, and curved forms. It emphasizes grandeur and emotional intensity through elaborate details.",
                "Modernist architecture embraces clean lines, minimal ornamentation, and functional design. It often uses industrial materials like steel, glass, and concrete.",
                "Art Deco architecture combines geometric patterns with luxurious materials. It features bold angular forms, stylized decorative elements, and rich colors."
            ]
            response = mock_responses[i % len(mock_responses)]
        
        # Get embedding for the response
        embedding = get_embedding(response)
        
        # Create descriptive model name
        model_name = f"architecture_response{i+1}_model"
        
        test_data.append((embedding, model_name))
        prompt_mapping[model_name] = architecture_prompt
        response_mapping[model_name] = response
        print(f"Generated {model_name}: {response[:100]}...")
    
    # Generate constitution responses
    for i in range(3):  # Generate 3 constitution responses
        if use_api:
            response = provider.generate_response(constitution_prompt)
        else:
            # Use mock responses for testing
            mock_responses = [
                "The United States Constitution establishes the framework of federal government with three branches: legislative, executive, and judicial. It defines the separation of powers and system of checks and balances.",
                "The Bill of Rights comprises the first ten amendments to the Constitution, protecting fundamental freedoms like speech, religion, and due process of law.",
                "The Constitution can be amended through a specific process requiring broad consensus, demonstrating both its stability and ability to evolve with changing times."
            ]
            response = mock_responses[i % len(mock_responses)]
        
        # Get embedding for the response
        embedding = get_embedding(response)
        
        # Create descriptive model name
        model_name = f"constitution_response{i+1}_model"
        
        test_data.append((embedding, model_name))
        prompt_mapping[model_name] = constitution_prompt
        response_mapping[model_name] = response
        print(f"Generated {model_name}: {response[:100]}...")
    
    return test_data, prompt_mapping, response_mapping

def plot_clustering_results_embedding(ensemble_data: List[Tuple[np.ndarray, str]], 
                                    clusters: List[Tuple[Tuple[np.ndarray, str], Tuple[Tuple[np.ndarray, str], ...]]], 
                                    min_distance: float, title: str, trial_folder: str):
    """
    Plot the clustering results for embedding data.
    
    Args:
        ensemble_data: Original data points
        clusters: Clustered data with centers and assigned points
        min_distance: The minimum distance used for clustering
        title: Title for the clustering method
        trial_folder: Folder to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Before clustering - show topics
    # Count responses by topic (based on model name)
    topic_counts = {"architecture": 0, "constitution": 0, "other": 0}
    
    for _, model_name in ensemble_data:
        if "architecture" in model_name.lower():
            topic_counts["architecture"] += 1
        elif "constitution" in model_name.lower():
            topic_counts["constitution"] += 1
        else:
            topic_counts["other"] += 1
    
    # Create bar chart for topics
    topics = list(topic_counts.keys())
    counts = list(topic_counts.values())
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    ax1.bar(topics, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Responses')
    ax1.set_title(f'Before Clustering\n{len(ensemble_data)} Total Responses')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: After clustering - show cluster composition
    cluster_labels = []
    architecture_counts = []
    constitution_counts = []
    other_counts = []
    
    for i, (center, assigned_points) in enumerate(clusters):
        cluster_size = len(assigned_points) + 1  # +1 for the center itself
        cluster_labels.append(f'Cluster {i+1}\n(n={cluster_size})')
        
        # Count topics in this cluster
        arch_count = 0
        const_count = 0
        other_count = 0
        
        # Count the center
        center_name = center[1]
        if "architecture" in center_name.lower():
            arch_count += 1
        elif "constitution" in center_name.lower():
            const_count += 1
        else:
            other_count += 1
        
        # Count assigned points
        for point in assigned_points:
            point_name = point[1]
            if "architecture" in point_name.lower():
                arch_count += 1
            elif "constitution" in point_name.lower():
                const_count += 1
            else:
                other_count += 1
        
        architecture_counts.append(arch_count)
        constitution_counts.append(const_count)
        other_counts.append(other_count)
    
    # Create stacked bar chart
    x = np.arange(len(cluster_labels))
    width = 0.6
    
    p1 = ax2.bar(x, architecture_counts, width, label='Architecture', color='skyblue', alpha=0.7)
    p2 = ax2.bar(x, constitution_counts, width, bottom=architecture_counts, 
                 label='Constitution', color='lightcoral', alpha=0.7)
    p3 = ax2.bar(x, other_counts, width, 
                 bottom=np.array(architecture_counts) + np.array(constitution_counts),
                 label='Other', color='lightgreen', alpha=0.7)
    
    ax2.set_ylabel('Number of Responses')
    ax2.set_title(f'After {title}\n{len(clusters)} Clusters (min_distance={min_distance:.2f})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cluster_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{title.lower().replace(' ', '_')}_embedding_clustering.png"
    save_path = os.path.join(trial_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Print clustering summary
    print(f"\nClustering Summary ({title}):")
    print(f"Original data points: {len(ensemble_data)}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Minimum distance: {min_distance:.3f}")
    print(f"\nCluster details:")
    
    for i, (center, assigned_points) in enumerate(clusters):
        print(f"  Cluster {i+1}: Center = {center[1]}")
        print(f"    Assigned points: {len(assigned_points)}")
        for point in assigned_points:
            print(f"      - {point[1]}")
        print()

def plot_clustering_results_by_response(ensemble_data: List[Tuple[np.ndarray, str]], 
                                     clusters: List[Tuple[Tuple[np.ndarray, str], Tuple[Tuple[np.ndarray, str], ...]]], 
                                     min_distance: float, title: str, trial_folder: str, 
                                     response_mapping: Dict[str, str]):
    """
    Plot the clustering results for embedding data, showing individual responses with unique colors.
    Each response gets its own color and is shown in a legend with the actual response text.
    
    Args:
        ensemble_data: Original data points
        clusters: Clustered data with centers and assigned points
        min_distance: The minimum distance used for clustering
        title: Title for the clustering method
        trial_folder: Folder to save the plot
        response_mapping: Dict mapping model_name to actual response text
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get unique model names and create color mapping
    model_names = [model_name for _, model_name in ensemble_data]
    unique_models = list(set(model_names))
    
    # Create distinct colors for each individual response
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_models)))
    model_colors = {model: colors[i] for i, model in enumerate(unique_models)}
    
    # Truncate response text for legend (keep first 80 characters)
    truncated_responses = {}
    for model_name in unique_models:
        response_text = response_mapping.get(model_name, f"Response for {model_name}")
        truncated_responses[model_name] = response_text[:80] + "..." if len(response_text) > 80 else response_text
    
    # Plot 1: Before clustering - show topics
    # Count responses by topic (based on model name)
    topic_counts = {"architecture": 0, "constitution": 0, "other": 0}
    topic_models = {"architecture": [], "constitution": [], "other": []}
    
    for _, model_name in ensemble_data:
        if "architecture" in model_name.lower():
            topic_counts["architecture"] += 1
            topic_models["architecture"].append(model_name)
        elif "constitution" in model_name.lower():
            topic_counts["constitution"] += 1
            topic_models["constitution"].append(model_name)
        else:
            topic_counts["other"] += 1
            topic_models["other"].append(model_name)
    
    # Create stacked bar chart for topics with individual colors
    topics = list(topic_counts.keys())
    x_pos = np.arange(len(topics))
    topic_colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    # For each topic, create individual bars for each model
    bottom_positions = [0] * len(topics)
    
    for topic_idx, topic in enumerate(topics):
        models_in_topic = topic_models[topic]
        for model_idx, model_name in enumerate(models_in_topic):
            ax1.bar(topic_idx, 1, bottom=bottom_positions[topic_idx], 
                   color=model_colors[model_name], alpha=0.7, 
                   edgecolor='black', linewidth=0.5, width=0.6)
            bottom_positions[topic_idx] += 1
    
    ax1.set_ylabel('Number of Responses')
    ax1.set_title(f'Before Clustering\n{len(ensemble_data)} Individual Responses')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(topics)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on top of each topic
    for i, count in enumerate(topic_counts.values()):
        ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: After clustering - show cluster composition with individual responses
    cluster_labels = []
    cluster_positions = []
    
    x_offset = 0
    for i, (center, assigned_points) in enumerate(clusters):
        cluster_size = len(assigned_points) + 1  # +1 for the center itself
        cluster_labels.append(f'Cluster {i+1}\n(n={cluster_size})')
        
        # Plot center (marked with star)
        center_name = center[1]
        ax2.bar(x_offset, 1, color=model_colors[center_name], alpha=0.7, 
               edgecolor='black', linewidth=2, width=0.8)
        ax2.text(x_offset, 1.1, '★', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Plot assigned points
        for j, point in enumerate(assigned_points):
            point_name = point[1]
            ax2.bar(x_offset + j + 1, 1, color=model_colors[point_name], alpha=0.7, 
                   edgecolor='black', linewidth=0.5, width=0.8)
        
        # Add cluster label above the cluster
        cluster_center_x = x_offset + (cluster_size - 1) / 2
        ax2.text(cluster_center_x, 1.3, f'Cluster {i+1}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add cluster separator (except for the last cluster)
        if i < len(clusters) - 1:
            separator_x = x_offset + cluster_size + 0.5
            ax2.axvline(x=separator_x, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        x_offset += cluster_size + 1  # +1 for spacing between clusters
        cluster_positions.append(cluster_center_x)
    
    ax2.set_ylabel('Response Count')
    ax2.set_title(f'After {title}\n{len(clusters)} Clusters (min_distance={min_distance:.2f})')
    ax2.set_xlim(-0.5, x_offset - 0.5)
    ax2.set_ylim(0, 1.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Remove x-axis ticks and labels for cleaner look
    ax2.set_xticks([])
    
    plt.tight_layout()
    
    # Create a comprehensive legend below the plots showing actual responses
    legend_elements = []
    legend_labels = []
    
    for model_name in sorted(unique_models):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=model_colors[model_name], alpha=0.7, 
                                           edgecolor='black', linewidth=0.5))
        legend_labels.append(f"{model_name}: {truncated_responses[model_name]}")
    
    # Add legend below the plots
    fig.legend(legend_elements, legend_labels, 
              loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=30,
              title="Individual Responses:", title_fontsize=24, frameon=True,
              markerscale=1.5)
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.45)
    
    # Save the plot
    filename = f"{title.lower().replace(' ', '_')}_response_clustering.png"
    save_path = os.path.join(trial_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Print clustering summary with actual responses
    print(f"\nResponse-based Clustering Summary ({title}):")
    print(f"Original data points: {len(ensemble_data)}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Minimum distance: {min_distance:.3f}")
    
    print(f"\nCluster details:")
    for i, (center, assigned_points) in enumerate(clusters):
        center_response = truncated_responses[center[1]]
        print(f"  Cluster {i+1}: Center = {center[1]}")
        print(f"    Center Response: {center_response}")
        print(f"    Assigned points: {len(assigned_points)}")
        for point in assigned_points:
            point_response = truncated_responses[point[1]]
            print(f"      - {point[1]}: {point_response}")
        print()

# def plot_clustering_results_with_groups(ensemble_data: List[Tuple[np.ndarray, str]], 
#                                        clusters: List[Tuple[Tuple[np.ndarray, str], Tuple[Tuple[np.ndarray, str], ...]]], 
#                                        min_distance: float, title: str, trial_folder: str, 
#                                        response_mapping: Dict[str, str]):
#     """
#     Plot the clustering results showing individual responses with unique colors AND cluster groupings.
#     Each response gets its own color and text in the legend, plus visual indicators for cluster membership.
    
#     Args:
#         ensemble_data: Original data points
#         clusters: Clustered data with centers and assigned points
#         min_distance: The minimum distance used for clustering
#         title: Title for the clustering method
#         trial_folder: Folder to save the plot
#         response_mapping: Dict mapping model_name to actual response text
#     """
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
#     # Get unique model names and create color mapping
#     model_names = [model_name for _, model_name in ensemble_data]
#     unique_models = list(set(model_names))
    
#     # Create distinct colors for each individual response
#     colors = plt.cm.tab20(np.linspace(0, 1, len(unique_models)))
#     model_colors = {model: colors[i] for i, model in enumerate(unique_models)}
    
#     # Create cluster grouping patterns/styles
#     cluster_patterns = ['', '///', '...', '+++', 'xxx', 'ooo', '***']  # Different hatch patterns
#     cluster_edge_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
    
#     # Map each model to its cluster
#     model_to_cluster = {}
#     for i, (center, assigned_points) in enumerate(clusters):
#         model_to_cluster[center[1]] = i
#         for point in assigned_points:
#             model_to_cluster[point[1]] = i
    
#     # Truncate response text for legend (keep first 60 characters for better readability with larger fonts)
#     truncated_responses = {}
#     for model_name in unique_models:
#         response_text = response_mapping.get(model_name, f"Response for {model_name}")
#         truncated_responses[model_name] = response_text[:60] + "..." if len(response_text) > 60 else response_text
    
#     # Plot 1: Before clustering - show topics with individual colors and cluster borders
#     topic_counts = {"architecture": 0, "constitution": 0, "other": 0}
#     topic_models = {"architecture": [], "constitution": [], "other": []}
    
#     for _, model_name in ensemble_data:
#         if "architecture" in model_name.lower():
#             topic_counts["architecture"] += 1
#             topic_models["architecture"].append(model_name)
#         elif "constitution" in model_name.lower():
#             topic_counts["constitution"] += 1
#             topic_models["constitution"].append(model_name)
#     else:
#             topic_counts["other"] += 1
#             topic_models["other"].append(model_name)
    
#     # Create stacked bar chart for topics with individual colors and cluster indicators
#     topics = list(topic_counts.keys())
#     x_pos = np.arange(len(topics))
    
#     # For each topic, create individual bars for each model with cluster indicators
#     bottom_positions = [0] * len(topics)
    
#     for topic_idx, topic in enumerate(topics):
#         models_in_topic = topic_models[topic]
#         for model_idx, model_name in enumerate(models_in_topic):
#             cluster_id = model_to_cluster.get(model_name, 0)
            
#             # Use cluster-specific edge styling
#             edge_color = cluster_edge_colors[cluster_id % len(cluster_edge_colors)]
#             pattern = cluster_patterns[cluster_id % len(cluster_patterns)]
#             edge_width = 3 if cluster_id < len(cluster_edge_colors) else 2
            
#             ax1.bar(topic_idx, 1, bottom=bottom_positions[topic_idx], 
#                    color=model_colors[model_name], alpha=0.7, 
#                    edgecolor=edge_color, linewidth=edge_width, width=0.6,
#                    hatch=pattern)
#             bottom_positions[topic_idx] += 1
    
#     ax1.set_ylabel('Number of Responses')
#     ax1.set_title(f'Before Clustering\n{len(ensemble_data)} Individual Responses\n(Border colors indicate future cluster groupings)')
#     ax1.set_xticks(x_pos)
#     ax1.set_xticklabels(topics)
#     ax1.grid(True, alpha=0.3, axis='y')
    
#     # Add count labels on top of each topic
#     for i, count in enumerate(topic_counts.values()):
#         ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    
#     # Plot 2: After clustering - show cluster composition with individual responses
#     x_offset = 0
#     cluster_info = []
    
#     for i, (center, assigned_points) in enumerate(clusters):
#         cluster_size = len(assigned_points) + 1  # +1 for the center itself
#         cluster_start_x = x_offset
        
#         # Get cluster styling
#         edge_color = cluster_edge_colors[i % len(cluster_edge_colors)]
#         pattern = cluster_patterns[i % len(cluster_patterns)]
        
#         # Plot center (marked with star)
#         center_name = center[1]
#         ax2.bar(x_offset, 1, color=model_colors[center_name], alpha=0.7, 
#                edgecolor=edge_color, linewidth=4, width=0.8, hatch=pattern)
#         ax2.text(x_offset, 1.15, '★', ha='center', va='bottom', fontsize=20, fontweight='bold', color='gold')
        
#         # Plot assigned points
#         for j, point in enumerate(assigned_points):
#             point_name = point[1]
#             ax2.bar(x_offset + j + 1, 1, color=model_colors[point_name], alpha=0.7, 
#                    edgecolor=edge_color, linewidth=3, width=0.8, hatch=pattern)
        
#         # Add cluster label above the cluster
#         cluster_center_x = x_offset + (cluster_size - 1) / 2
#         ax2.text(cluster_center_x, 1.4, f'Cluster {i+1}', ha='center', va='bottom', 
#                 fontweight='bold', fontsize=12, 
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor=edge_color, alpha=0.3))
        
#         # Store cluster info for legend
#         cluster_info.append({
#             'id': i+1,
#             'color': edge_color,
#             'pattern': pattern,
#             'size': cluster_size
#         })
        
#         # Add cluster separator (except for the last cluster)
#         if i < len(clusters) - 1:
#             separator_x = x_offset + cluster_size + 0.5
#             ax2.axvline(x=separator_x, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
#         x_offset += cluster_size + 1  # +1 for spacing between clusters
    
#     ax2.set_ylabel('Response Count')
#     ax2.set_title(f'After {title}\n{len(clusters)} Clusters (min_distance={min_distance:.2f})\n(Border colors/patterns show cluster membership)')
#     ax2.set_xlim(-0.5, x_offset - 0.5)
#     ax2.set_ylim(0, 1.6)
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     # Remove x-axis ticks and labels for cleaner look
#     ax2.set_xticks([])
    
#     plt.tight_layout()
    
#     # Create two legends: one for individual responses, one for cluster groups
    
#     # Legend 1: Individual Responses (left side)
#     response_legend_elements = []
#     response_legend_labels = []
    
#     for model_name in sorted(unique_models):
#         response_legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=model_colors[model_name], alpha=0.7, 
#                                                     edgecolor='black', linewidth=0.5))
#         response_legend_labels.append(f"{model_name}: {truncated_responses[model_name]}")
    
#     # Legend 2: Cluster Groups (right side)
#     cluster_legend_elements = []
#     cluster_legend_labels = []
    
#     for cluster in cluster_info:
#         cluster_legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='lightgray', alpha=0.7, 
#                                                     edgecolor=cluster['color'], linewidth=3,
#                                                     hatch=cluster['pattern']))
#         cluster_legend_labels.append(f"Cluster {cluster['id']} (n={cluster['size']})")
    
#     # Position legends side by side below the plots
#     legend1 = fig.legend(response_legend_elements, response_legend_labels, 
#                         loc='lower left', bbox_to_anchor=(0.02, -0.35), ncol=1, fontsize=20,
#                         title="Individual Responses:", title_fontsize=20, frameon=True,
#                         markerscale=1.5)
    
#     legend2 = fig.legend(cluster_legend_elements, cluster_legend_labels, 
#                         loc='lower right', bbox_to_anchor=(0.98, -0.35), ncol=1, fontsize=20,
#                         title="Cluster Groups:", title_fontsize=20, frameon=True,
#                         markerscale=1.5)
    
#     # Adjust layout to make room for legends
#     plt.subplots_adjust(bottom=0.5)
    
#     # Save the plot
#     filename = f"{title.lower().replace(' ', '_')}_grouped_response_clustering.png"
#     save_path = os.path.join(trial_folder, filename)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved: {save_path}")
    
#     # Print detailed clustering summary
#     print(f"\nGrouped Response-based Clustering Summary ({title}):")
#     print(f"Original data points: {len(ensemble_data)}")
#     print(f"Number of clusters: {len(clusters)}")
#     print(f"Minimum distance: {min_distance:.3f}")
    
#     print(f"\nDetailed cluster composition:")
#     for i, (center, assigned_points) in enumerate(clusters):
#         cluster_color = cluster_edge_colors[i % len(cluster_edge_colors)]
#         cluster_pattern = cluster_patterns[i % len(cluster_patterns)]
#         center_response = truncated_responses[center[1]]
        
#         print(f"  Cluster {i+1} (Border: {cluster_color}, Pattern: {cluster_pattern}):")
#         print(f"    ★ CENTER: {center[1]}")
#         print(f"      Response: {center_response}")
#         print(f"    Assigned points: {len(assigned_points)}")
#         for point in assigned_points:
#             point_response = truncated_responses[point[1]]
#             print(f"      - {point[1]}: {point_response}")
#         print()

def save_test_data(test_data: List[Tuple[np.ndarray, str]], trial_folder: str):
    """
    Save test data to a JSON file for later analysis.
    
    Args:
        test_data: List of (embedding_vector, model_name) tuples
        trial_folder: Folder to save the data
    """
    # Convert embeddings to lists for JSON serialization
    serializable_data = []
    for embedding, model_name in test_data:
        serializable_data.append({
            "embedding": embedding.tolist(),
            "model_name": model_name
        })
    
    # Save to file
    save_path = os.path.join(trial_folder, "embedding_test_data.json")
    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Test data saved to: {save_path}")

def main():
    """Main experiment function."""
    USE_API = False

    print("=" * 60)
    print("CLUSTERING EMBEDDING EXPERIMENT")
    print("=" * 60)
    
    # Configure Gemini API
    try:
        configure_gemini_api()
        print("✓ Gemini API configured successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not configure Gemini API: {e}")
        print("Using mock responses instead.")
    
    # Create trial folder
    trial_folder = create_trial_folder()
    
    # Initialize model provider
    if USE_API:
        try:
            provider = GeminiProvider()
            print("✓ Using Gemini API for response generation")
        except Exception as e:
            print(f"⚠ Falling back to mock provider: {e}")
            provider = MockProvider()
    else:
        provider = MockProvider()
        print("✓ Using mock provider for response generation")
    
    # Generate test data
    print("\nGenerating test responses...")
    test_data, prompt_mapping, response_mapping = generate_test_responses(provider, use_api=USE_API)
    
    # Save test data
    save_test_data(test_data, trial_folder)
    
    # Set parameters for clustering
    MIN_DISTANCE = 0.3  # Embedding distance is typically 0-2, so we use a smaller value
    
    print(f"\nUsing min_distance={MIN_DISTANCE} for embedding clustering")
    print(f"Total test data points: {len(test_data)}")
    
    # Perform repulsive sampling
    print("\nPerforming repulsive sampling clustering...")
    centers = repulsive_sampling_embedding(test_data, min_distance=MIN_DISTANCE)
    clusters = assign_to_clusters_embedding(test_data, centers)
    plot_clustering_results_embedding(test_data, clusters, MIN_DISTANCE, "Repulsive Clustering", trial_folder)
    plot_clustering_results_by_response(test_data, clusters, MIN_DISTANCE, "Repulsive Clustering", trial_folder, response_mapping)
    # plot_clustering_results_with_groups(test_data, clusters, min_distance, "Repulsive Clustering", trial_folder, response_mapping)
    
    # Perform DPP sampling
    print("\nPerforming DPP sampling clustering...")
    dpp_centers = dpp_sampling_embedding(test_data, k=len(centers), sigma=0.5)
    dpp_clusters = assign_to_clusters_embedding(test_data, dpp_centers)
    plot_clustering_results_embedding(test_data, dpp_clusters, MIN_DISTANCE, "DPP Clustering", trial_folder)
    plot_clustering_results_by_response(test_data, dpp_clusters, MIN_DISTANCE, "DPP Clustering", trial_folder, response_mapping)
    # plot_clustering_results_with_groups(test_data, dpp_clusters, min_distance, "DPP Clustering", trial_folder, response_mapping)
    
    # Perform random sampling
    print("\nPerforming random sampling clustering...")
    random_centers = random_sampling_embedding(test_data, k=len(centers))
    random_clusters = assign_to_clusters_embedding(test_data, random_centers)
    plot_clustering_results_embedding(test_data, random_clusters, MIN_DISTANCE, "Random Clustering", trial_folder)
    plot_clustering_results_by_response(test_data, random_clusters, MIN_DISTANCE, "Random Clustering", trial_folder, response_mapping)
    # plot_clustering_results_with_groups(test_data, random_clusters, min_distance, "Random Clustering", trial_folder, response_mapping)
    
    # Perform nearest sampling (based on similarity to first point)
    print("\nPerforming nearest sampling clustering...")
    nearest_centers = nearest_sampling_embedding(test_data, k=len(centers))
    nearest_clusters = assign_to_clusters_embedding(test_data, nearest_centers)
    plot_clustering_results_embedding(test_data, nearest_clusters, MIN_DISTANCE, "Nearest Clustering", trial_folder)
    plot_clustering_results_by_response(test_data, nearest_clusters, MIN_DISTANCE, "Nearest Clustering", trial_folder, response_mapping)
    # plot_clustering_results_with_groups(test_data, nearest_clusters, min_distance, "Nearest Clustering", trial_folder, response_mapping)
    
    print(f"\n✓ All results saved to: {trial_folder}")
    print("=" * 60)

if __name__ == "__main__":
    main() 