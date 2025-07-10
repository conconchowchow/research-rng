"""
Experiment to test how well LLMs can generate unique jokes using embedding clustering.

This experiment generates N jokes from an LLM, embeds them, and then uses clustering
strategies from clustering_embedding.py to analyze joke diversity and uniqueness.
"""

import numpy as np
import json
import os
import sys
from typing import List, Tuple, Dict
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Add parent directory to path to import clustering functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import clustering functions and utilities
from clustering_embedding import (
    repulsive_sampling_embedding,
    assign_to_clusters_embedding,
    dpp_sampling_embedding,
    random_sampling_embedding,
    nearest_sampling_embedding,
    embedding_distance,
    get_embedding,
    configure_gemini_api
)

# Import model providers
from model_provider import ModelProvider, GeminiProvider, MockProvider

class JokeDiversityExperiment:
    """
    Experiment class to generate and analyze joke diversity using embedding clustering.
    """
    
    def __init__(self, use_api: bool = True, num_jokes: int = 20):
        """
        Initialize the joke diversity experiment.
        
        Args:
            use_api: Whether to use real API or mock responses
            num_jokes: Number of jokes to generate
        """
        self.use_api = use_api
        self.num_jokes = num_jokes
        self.provider = None
        self.jokes_data = []  # List of (embedding, joke_text) tuples
        
        # Initialize model provider
        self._setup_provider()
    
    def _setup_provider(self):
        """Set up the model provider for joke generation."""
        if self.use_api:
            try:
                configure_gemini_api()
                self.provider = GeminiProvider()
                print("✓ Using Gemini API for joke generation")
            except Exception as e:
                print(f"⚠ Falling back to mock provider: {e}")
                self.provider = MockProvider()
                self.use_api = False
        else:
            self.provider = MockProvider()
            print("✓ Using mock provider for joke generation")
    
    def generate_jokes(self) -> List[Tuple[np.ndarray, str]]:
        """
        Generate N jokes and convert them to embedding format.
        
        Returns:
            List of (embedding_vector, joke_text) tuples
        """
        print(f"\nGenerating {self.num_jokes} jokes...")
        
        # Different joke prompts for variety
        joke_prompts = [
            "Tell me a funny one-liner joke.",
            "Give me a clever pun or wordplay joke.",
            # "Share a funny observational humor joke.",
            # "Tell me a joke about everyday life.",
            # "Give me a witty joke that makes people think.",
            # "Share a clean, family-friendly joke.",
            # "Tell me a joke about technology or computers.",
            # "Give me a food-related joke.",
            # "Share a joke about animals.",
            # "Tell me a travel or vacation joke."
        ]
        
        jokes_data = []
        
        # Mock jokes for when not using API
        mock_jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my wife she was drawing her eyebrows too high. She seemed surprised.",
            "Why don't skeletons fight each other? They don't have the guts.",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
            "Why did the math book look so sad? Because it had too many problems!",
            "What do you call a sleeping bull? A bulldozer!",
            "Why don't some couples go to the gym? Because some relationships don't work out!",
            "What did the ocean say to the beach? Nothing, it just waved!",
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fish wearing a crown? A king fish!",
            "Why did the bicycle fall over? Because it was two-tired!",
            "What do you call a dinosaur that crashes his car? Tyrannosaurus Wrecks!",
            "Why don't programmers like nature? It has too many bugs!",
            "What's orange and sounds like a parrot? A carrot!",
            "Why did the coffee file a police report? It got mugged!",
            "What do you call a belt made of watches? A waist of time!"
        ]
        
        for i in range(self.num_jokes):
            if self.use_api:
                # Use a variety of prompts
                prompt = joke_prompts[i % len(joke_prompts)]
                joke_response = self.provider.generate_response(prompt)
            else:
                # Use mock jokes with some variation
                joke_response = mock_jokes[i % len(mock_jokes)]
                if i >= len(mock_jokes):
                    # Add some variation for repeated jokes
                    joke_response += f" (variation {i // len(mock_jokes) + 1})"
            
            # Get embedding for the joke
            embedding = get_embedding(joke_response)
            
            # Store as tuple (embedding, joke_text)
            jokes_data.append((embedding, joke_response.strip()))
            
            print(f"Generated joke {i+1}: {joke_response[:50]}...")
        
        self.jokes_data = jokes_data
        print(f"✓ Generated {len(jokes_data)} jokes with embeddings")
        return jokes_data
    
    def analyze_clustering(self, method: str = "repulsive", min_distance: float = 0.3, k: int = None) -> Dict:
        """
        Analyze joke clustering using the specified method.
        
        Args:
            method: Clustering method ("repulsive", "dpp", "random", "nearest")
            min_distance: Minimum distance for repulsive clustering
            k: Number of clusters for methods that require it
            
        Returns:
            Dictionary with clustering results
        """
        if not self.jokes_data:
            raise ValueError("No jokes data available. Run generate_jokes() first.")
        
        print(f"\nAnalyzing clustering using {method} method...")
        
        # Select clustering method
        if method == "repulsive":
            logger.info(f"Using repulsive clustering with min_distance={min_distance}")
            centers = repulsive_sampling_embedding(self.jokes_data, min_distance=min_distance)
            logger.info(f"Repulsive clustering centers: {centers}")
        elif method == "dpp":
            centers = dpp_sampling_embedding(self.jokes_data, k=k)
        elif method == "random":
            centers = random_sampling_embedding(self.jokes_data, k=k if k else max(1, len(self.jokes_data) // 4))
        elif method == "nearest":
            centers = nearest_sampling_embedding(self.jokes_data, k=k if k else max(1, len(self.jokes_data) // 4))
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Assign points to clusters
        clusters = assign_to_clusters_embedding(self.jokes_data, centers)
        
        # Format results
        results = {
            "method": method,
            "parameters": {
                "min_distance": min_distance if method == "repulsive" else None,
                "k": k,
                "total_jokes": len(self.jokes_data),
                "num_clusters": len(clusters)
            },
            "clusters": {}
        }
        
        # Convert clusters to the requested format
        for i, (center, assigned_points) in enumerate(clusters):
            cluster_id = f"cluster_{i+1}"
            
            # Extract joke texts
            center_joke = center[1]  # center is (embedding, joke_text)
            cluster_jokes = [point[1] for point in assigned_points]  # each point is (embedding, joke_text)
            
            results["clusters"][cluster_id] = {
                "cluster_center_response": center_joke,
                "cluster_responses": cluster_jokes
            }
        
        print(f"✓ Created {len(clusters)} clusters using {method} method")
        return results
    
    def calculate_diversity_metrics(self, clusters_result: Dict) -> Dict:
        """
        Calculate diversity metrics for the clustering results.
        
        Args:
            clusters_result: Results from analyze_clustering
            
        Returns:
            Dictionary with diversity metrics
        """
        clusters = clusters_result["clusters"]
        total_jokes = clusters_result["parameters"]["total_jokes"]
        num_clusters = len(clusters)
        
        # Calculate cluster sizes
        cluster_sizes = []
        for cluster_data in clusters.values():
            cluster_size = len(cluster_data["cluster_responses"]) + 1  # +1 for center
            cluster_sizes.append(cluster_size)
        
        # Calculate average intra-cluster distances
        intra_cluster_distances = []
        for cluster_id, cluster_data in clusters.items():
            center_joke = cluster_data["cluster_center_response"]
            cluster_jokes = cluster_data["cluster_responses"]
            
            if len(cluster_jokes) > 0:
                # Find center embedding
                center_embedding = None
                for emb, joke in self.jokes_data:
                    if joke == center_joke:
                        center_embedding = emb
                        break
                
                if center_embedding is not None:
                    # Calculate distances from center to all points in cluster
                    distances = []
                    for joke in cluster_jokes:
                        for emb, j in self.jokes_data:
                            if j == joke:
                                dist = embedding_distance((center_embedding, center_joke), (emb, joke))
                                distances.append(dist)
                                break
                    
                    if distances:
                        intra_cluster_distances.extend(distances)
        
        # Calculate inter-cluster distances (between cluster centers)
        inter_cluster_distances = []
        cluster_centers = []
        
        for cluster_data in clusters.values():
            center_joke = cluster_data["cluster_center_response"]
            for emb, joke in self.jokes_data:
                if joke == center_joke:
                    cluster_centers.append((emb, joke))
                    break
        
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                dist = embedding_distance(cluster_centers[i], cluster_centers[j])
                inter_cluster_distances.append(dist)
        
        metrics = {
            "total_jokes": total_jokes,
            "num_clusters": num_clusters,
            "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
            "cluster_size_std": np.std(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "avg_intra_cluster_distance": np.mean(intra_cluster_distances) if intra_cluster_distances else 0,
            "avg_inter_cluster_distance": np.mean(inter_cluster_distances) if inter_cluster_distances else 0,
            "diversity_ratio": (np.mean(inter_cluster_distances) / np.mean(intra_cluster_distances)) if intra_cluster_distances and inter_cluster_distances else 0
        }
        
        return metrics
    
    def run_full_experiment(self, output_dir: str = "results") -> str:
        """
        Run the complete joke diversity experiment.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to the saved results file
        """
        print("=" * 60)
        print("JOKE DIVERSITY EXPERIMENT")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate jokes
        self.generate_jokes()
        
        # Run different clustering methods
        methods = ["repulsive", "dpp", "random", "nearest"]
        all_results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "num_jokes": self.num_jokes,
                "use_api": self.use_api,
                "provider": type(self.provider).__name__
            },
            "methods": {}
        }
        
        for method in methods:
            print(f"\n{'-' * 40}")
            print(f"Running {method.upper()} clustering...")
            print(f"{'-' * 40}")
            
            try:
                if method == "repulsive":
                    clusters_result = self.analyze_clustering(method=method, min_distance=0.3)
                else:
                    # For other methods, use about 1/4 of the jokes as cluster centers
                    k = max(1, self.num_jokes // 4)
                    clusters_result = self.analyze_clustering(method=method, k=k)
                
                # Calculate diversity metrics
                diversity_metrics = self.calculate_diversity_metrics(clusters_result)
                
                # Store results
                all_results["methods"][method] = {
                    "clustering_results": clusters_result,
                    "diversity_metrics": diversity_metrics
                }
                
                print(f"✓ {method.upper()} clustering completed:")
                print(f"  - {diversity_metrics['num_clusters']} clusters")
                print(f"  - Avg cluster size: {diversity_metrics['avg_cluster_size']:.1f}")
                print(f"  - Diversity ratio: {diversity_metrics['diversity_ratio']:.3f}")
                
            except Exception as e:
                print(f"Error in {method} clustering: {e}")
                all_results["methods"][method] = {"error": str(e)}
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"joke_diversity_experiment_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Print summary
        self._print_experiment_summary(all_results)
        
        return output_file
    
    def _print_experiment_summary(self, results: Dict):
        """Print a summary of the experiment results."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        info = results["experiment_info"]
        print(f"Jokes generated: {info['num_jokes']}")
        print(f"Provider used: {info['provider']}")
        print(f"Timestamp: {info['timestamp']}")
        
        print(f"\nClustering Method Comparison:")
        print(f"{'Method':<12} {'Clusters':<10} {'Avg Size':<10} {'Diversity':<12}")
        print("-" * 50)
        
        for method, data in results["methods"].items():
            if "error" not in data:
                metrics = data["diversity_metrics"]
                print(f"{method.capitalize():<12} {metrics['num_clusters']:<10} "
                      f"{metrics['avg_cluster_size']:<10.1f} {metrics['diversity_ratio']:<12.3f}")
            else:
                print(f"{method.capitalize():<12} {'ERROR':<10} {'-':<10} {'-':<12}")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run the joke diversity experiment."""
    
    # Configuration
    USE_API = True  # Set to True to use real API
    NUM_JOKES = 25   # Number of jokes to generate
    OUTPUT_DIR = "results"
    
    # Create and run experiment
    experiment = JokeDiversityExperiment(use_api=USE_API, num_jokes=NUM_JOKES)
    results_file = experiment.run_full_experiment(output_dir=OUTPUT_DIR)
    
    print(f"\nExperiment completed! Results saved to: {results_file}")


if __name__ == "__main__":
    main()
