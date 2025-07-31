import json
import os
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from clustering_embedding import get_embedding, embedding_distance, repulsive_sampling_embedding, assign_to_clusters_embedding, plot_clustering_results_embedding, plot_clustering_results_by_response
import numpy as np
import matplotlib.pyplot as plt
import umap

class Generator:
    def __init__(self, config: List[Dict[str, Any]], 
                 temperature: float = 0.7, 
                 max_tokens: int = 1000,
                 logs_dir: str = "logs"):
        """
        Initialize the Generator with model configurations.
        
        Args:
            config: List of dicts with format [{"model_name": "name", "number_of_models": x}, ...]
            temperature: Default temperature for generation
            max_tokens: Default max tokens for generation
            logs_dir: Directory to save logs
        """
        self.config = config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logs_dir = logs_dir
        
        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare model instances with unique names
        self.model_instances = []
        for model_config in config:
            model_name = model_config["model_name"]
            num_models = model_config.get("number_of_models", 1)
            
            for i in range(num_models):
                unique_name = f"{model_name}_{i+1}" if num_models > 1 else model_name
                self.model_instances.append({
                    "unique_name": unique_name,
                    "model_name": model_name,
                    "temperature": model_config.get("temperature", self.temperature),
                    "max_tokens": model_config.get("max_tokens", self.max_tokens),
                    # "top_p": model_config.get("top_p", 1.0),
                })
    
    def _get_prompt_prefix(self, prompt: str, max_words: int = 5) -> str:
        """
        Extract first few words from prompt for filename.
        
        Args:
            prompt: Input prompt
            max_words: Maximum number of words to use
            
        Returns:
            Sanitized string for use in filename
        """
        # Extract first few words and sanitize for filename
        words = prompt.split()[:max_words]
        prefix = "_".join(words)
        # Remove special characters and limit length
        prefix = re.sub(r'[^a-zA-Z0-9_]', '', prefix)
        return prefix[:50]  # Limit length
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert an object to be JSON serializable.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._make_json_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__, convert to dict
            return self._make_json_serializable(obj.__dict__)
        else:
            # For anything else, convert to string representation
            return str(obj)

    def _generate_single_response(self, prompt: str, model_instance: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate a single response using LiteLLM.
        
        Args:
            prompt: Input prompt
            model_instance: Model configuration dict
            
        Returns:
            Tuple of (response, unique_model_name, metadata)
        """
        try:
            response = litellm.completion(
                model=model_instance["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=model_instance["temperature"],
                max_tokens=model_instance["max_tokens"],
                # top_p=model_instance["top_p"]
            )
            
            response_text = response.choices[0].message.content
            
            # Collect metadata with proper JSON serialization
            usage_data = {}
            if hasattr(response, 'usage') and response.usage:
                try:
                    # Try to convert usage to dict first
                    if hasattr(response.usage, '__dict__'):
                        usage_data = self._make_json_serializable(response.usage.__dict__)
                    else:
                        usage_data = self._make_json_serializable(dict(response.usage))
                except Exception as e:
                    # Fallback: just get basic usage info as strings
                    usage_data = {"error_converting_usage": str(e), "usage_str": str(response.usage)}
            
            metadata = {
                "model_config": model_instance,
                "usage": usage_data,
                "response_id": str(response.id) if hasattr(response, 'id') else None,
                "created": int(response.created) if hasattr(response, 'created') else None,
                "timestamp": datetime.now().isoformat()
            }
            
            return response_text, model_instance["unique_name"], metadata
            
        except Exception as e:
            error_response = f"Error generating response with {model_instance['unique_name']}: {str(e)}"
            metadata = {
                "model_config": model_instance,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return error_response, model_instance["unique_name"], metadata
    
    def generate_response(self, prompt: str, max_workers: int = 5) -> List[Tuple[str, str]]:
        """
        Generate ensemble responses from all configured models.
        
        Args:
            prompt: Input prompt
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of (response, unique_model_name) tuples
        """
        results = []
        all_metadata = []
        
        # Generate responses concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(self._generate_single_response, prompt, model_instance): model_instance
                for model_instance in self.model_instances
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                response, unique_name, metadata = future.result()
                results.append((response, unique_name))
                all_metadata.append(metadata)
        
        # Generate clustering and UMAP data
        clustering_data = self._generate_clustering_data(results)
        
        # Log the results with clustering data
        self._log_responses(prompt, results, all_metadata, clustering_data)
        
        # Return just the response and model name tuples
        return results
    
    def _generate_clustering_data(self, responses: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Generate clustering and UMAP visualization data.
        
        Args:
            responses: List of (response, unique_model_name) tuples
            
        Returns:
            Dictionary containing clustering and UMAP data
        """
        try:
            # Generate clusters
            clusters, min_distance = self.cluster_responses(responses)
            
            # Extract all embeddings and create data for UMAP
            all_embeddings = []
            cluster_labels = []
            model_names = []
            is_center = []
            response_texts = []
            
            # Build mapping from model names to responses for easy lookup
            response_map = {model_name: response for response, model_name in responses}
            
            # Process each cluster
            for cluster_idx, (center, assigned_points) in enumerate(clusters):
                # Add center point
                center_embedding, center_model = center
                all_embeddings.append(center_embedding.tolist())  # Convert to list for JSON
                cluster_labels.append(cluster_idx)
                model_names.append(center_model)
                is_center.append(True)
                response_texts.append(response_map.get(center_model, ""))
                
                # Add assigned points
                for point_embedding, point_model in assigned_points:
                    all_embeddings.append(point_embedding.tolist())  # Convert to list for JSON
                    cluster_labels.append(cluster_idx)
                    model_names.append(point_model)
                    is_center.append(False)
                    response_texts.append(response_map.get(point_model, ""))
            
            # Generate UMAP coordinates if we have enough points
            umap_coordinates = []
            if len(all_embeddings) > 1:
                try:
                    embeddings_array = np.array(all_embeddings)
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_embeddings)-1))
                    embedding_2d = reducer.fit_transform(embeddings_array)
                    umap_coordinates = embedding_2d.tolist()  # Convert to list for JSON
                except Exception as e:
                    print(f"Warning: UMAP generation failed: {e}")
                    # Fallback to random coordinates
                    umap_coordinates = [[i*0.1, j*0.1] for i, j in enumerate(range(len(all_embeddings)))]
            
            return {
                "num_clusters": len(clusters),
                "min_distance": float(min_distance),
                "points": [
                    {
                        "umap_x": umap_coordinates[i][0] if i < len(umap_coordinates) else 0,
                        "umap_y": umap_coordinates[i][1] if i < len(umap_coordinates) else 0,
                        "cluster_id": int(cluster_labels[i]),
                        "model_name": model_names[i],
                        "is_center": bool(is_center[i]),
                        "response_text": response_texts[i]
                    }
                    for i in range(len(model_names))
                ]
            }
            
        except Exception as e:
            print(f"Warning: Failed to generate clustering data: {e}")
            return {
                "num_clusters": 0,
                "min_distance": 0.0,
                "points": [],
                "error": str(e)
            }

    def _log_responses(self, prompt: str, responses: List[Tuple[str, str]], metadata_list: List[Dict[str, Any]], clustering_data: Dict[str, Any] = None):
        """
        Log all responses to a JSON file.
        
        Args:
            prompt: Original prompt
            responses: List of (response, unique_model_name) tuples
            metadata_list: List of metadata dicts for each response
            clustering_data: Optional clustering and UMAP data
        """
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "generator_config": {
                "model_config": self.config,
                "default_temperature": self.temperature,
                "default_max_tokens": self.max_tokens
            },
            "responses": [],
            "clustering_data": clustering_data if clustering_data else None
        }
        
        # Add each response with its metadata, ensuring JSON serializability
        for (response, unique_name), metadata in zip(responses, metadata_list):
            try:
                # Ensure all data is JSON serializable
                serializable_metadata = self._make_json_serializable(metadata)
                log_entry["responses"].append({
                    "response": str(response),  # Ensure response is string
                    "unique_model_name": str(unique_name),
                    "metadata": serializable_metadata
                })
            except Exception as e:
                # Fallback: create a minimal entry if serialization fails
                print(f"Warning: Failed to serialize metadata for {unique_name}: {e}")
                log_entry["responses"].append({
                    "response": str(response),
                    "unique_model_name": str(unique_name),
                    "metadata": {
                        "error": f"Failed to serialize metadata: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                })
        
        # Generate filename
        prompt_prefix = self._get_prompt_prefix(prompt)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prompt_prefix}_{timestamp}.json"
        filepath = os.path.join(self.logs_dir, filename)
        
        # Write to file with better error handling
        try:
            # First, test if the log_entry is JSON serializable
            test_json = json.dumps(log_entry, indent=2, ensure_ascii=False)
            
            # If that succeeds, write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(test_json)
                f.flush()  # Ensure all data is written
                
            print(f"Successfully logged responses to {filepath}")
            
        except (TypeError, ValueError) as e:
            print(f"JSON serialization error for {filepath}: {e}")
            # Try to save a fallback version with minimal data
            fallback_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": str(prompt),
                "error": f"Original log failed JSON serialization: {str(e)}",
                "responses_count": len(responses),
                "model_names": [unique_name for _, unique_name in responses]
            }
            try:
                with open(filepath.replace('.json', '_fallback.json'), 'w', encoding='utf-8') as f:
                    json.dump(fallback_entry, f, indent=2, ensure_ascii=False)
                    f.flush()
                print(f"Saved fallback log to {filepath.replace('.json', '_fallback.json')}")
            except Exception as fallback_error:
                print(f"Even fallback logging failed: {fallback_error}")
                
        except Exception as e:
            print(f"File I/O error writing log file {filepath}: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about configured models.
        
        Returns:
            Dictionary with model configuration information
        """
        return {
            "total_models": len(self.model_instances),
            "unique_model_names": [instance["unique_name"] for instance in self.model_instances],
            "base_models": list(set(instance["model_name"] for instance in self.model_instances)),
            "config": self.config,
            "default_settings": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        }
    
    def cluster_responses(self, responses: List[Tuple[str, str]]) -> List[str]:
        """
        Cluster responses using embedding clustering

        Input: List[Tuple[str, str]] - (response, model_name)
        Output: List[Tuple[Tuple[np.ndarray, str], Tuple[Tuple[np.ndarray, str], ...]]] - (center, (assigned_points, ...))
        """

        # Convert each response to a tuple of (embedding, model_name)
        ensemble_data = [(get_embedding(response), model_name) for response, model_name in responses]

        # Get the centers:
        centers, min_distance = repulsive_sampling_embedding(ensemble_data, divisor_coefficient=2)

        # Cluster the embeddings:
        clusters = assign_to_clusters_embedding(ensemble_data, centers)

        # Return the clusters:
        return clusters, min_distance
    
    def plot_clusters(self, responses: List[Tuple[str, str]], clusters: List[Tuple[Tuple[np.ndarray, str], Tuple[Tuple[np.ndarray, str], ...]]], min_distance: float):
        """
        Plot the clusters using matplotlib
        """
        plot_clustering_results_embedding(responses, clusters, min_distance, "Repulsive Clustering", self.logs_dir)
        # plot_clustering_results_by_response(responses, clusters, min_distance, "Repulsive Clustering", trial_folder, response_mapping)
    
    def plot_umap_clustering(self, responses: List[Tuple[str, str]], clusters: List[Tuple[Tuple[np.ndarray, str], Tuple[Tuple[np.ndarray, str], ...]]], min_distance: float):
        """
        Plot the clusters using UMAP dimensionality reduction with colors representing clusters.
        
        Args:
            responses: List of (response_text, model_name) tuples
            clusters: List of clusters from cluster_responses method
            min_distance: Minimum distance from clustering
        """
        try:
            # Extract all embeddings and create cluster labels
            all_embeddings = []
            cluster_labels = []
            model_names = []
            
            # Process each cluster
            for cluster_idx, (center, assigned_points) in enumerate(clusters):
                # Add center point
                center_embedding, center_model = center
                all_embeddings.append(center_embedding)
                cluster_labels.append(cluster_idx)
                model_names.append(f"{center_model} (center)")
                
                # Add assigned points
                for point_embedding, point_model in assigned_points:
                    all_embeddings.append(point_embedding)
                    cluster_labels.append(cluster_idx)
                    model_names.append(point_model)
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings)
            
            # Apply UMAP dimensionality reduction
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_embeddings)-1))
            embedding_2d = reducer.fit_transform(embeddings_array)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Generate very distinct colors for each cluster
            # Use a combination of distinct colors from different colormaps for maximum differentiation
            distinct_colors = [
                '#FF0000',  # Red
                '#0000FF',  # Blue
                '#00FF00',  # Green
                '#FF8000',  # Orange
                '#8000FF',  # Purple
                '#00FFFF',  # Cyan
                '#FF00FF',  # Magenta
                '#FFFF00',  # Yellow
                '#800000',  # Dark Red
                '#000080',  # Dark Blue
                '#008000',  # Dark Green
                '#800080',  # Dark Purple
                '#808000',  # Olive
                '#008080',  # Teal
                '#FFA500',  # Orange
                '#FF69B4'   # Hot Pink
            ]
            
            # If we have more clusters than distinct colors, fall back to tab20
            if len(clusters) > len(distinct_colors):
                colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
            else:
                colors = [distinct_colors[i] for i in range(len(clusters))]
            
            # Keep track of which clusters we've added to legend to ensure all are included
            legend_added = set()
            
            # Plot each cluster with a different color
            for cluster_idx in range(len(clusters)):
                cluster_mask = np.array(cluster_labels) == cluster_idx
                cluster_points = embedding_2d[cluster_mask]
                cluster_models = [model_names[i] for i in range(len(model_names)) if cluster_labels[i] == cluster_idx]
                
                # Separate centers and regular points
                center_points = []
                regular_points = []
                center_indices = []
                regular_indices = []
                
                for i, model in enumerate(cluster_models):
                    if "(center)" in model:
                        center_points.append(cluster_points[i])
                        center_indices.append(i)
                    else:
                        regular_points.append(cluster_points[i])
                        regular_indices.append(i)
                
                # Plot regular points with larger size and no border
                if regular_points:
                    regular_points = np.array(regular_points)
                    plt.scatter(regular_points[:, 0], regular_points[:, 1], 
                              c=[colors[cluster_idx]], alpha=0.8, s=100,
                              label=f'Cluster {cluster_idx + 1}')
                    legend_added.add(cluster_idx)
                
                # Plot center points with larger size and minimal border
                if center_points:
                    center_points = np.array(center_points)
                    plt.scatter(center_points[:, 0], center_points[:, 1], 
                              c=[colors[cluster_idx]], alpha=1.0, s=150,
                              marker='*', edgecolors='black', linewidth=0.5)
                    
                    # Add to legend if not already added (for clusters with only centers)
                    if cluster_idx not in legend_added:
                        plt.scatter([], [], c=[colors[cluster_idx]], alpha=0.8, s=100,
                                  label=f'Cluster {cluster_idx + 1}')
                        legend_added.add(cluster_idx)
            
            plt.title(f'UMAP Visualization of Response Clusters\n{len(responses)} responses, {len(clusters)} clusters, centers are marked with *', fontsize=14)
            plt.xlabel('UMAP Dimension 1', fontsize=12)
            plt.ylabel('UMAP Dimension 2', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"umap_clustering_{timestamp}.png"
            filepath = os.path.join(self.logs_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"UMAP clustering plot saved to {filepath}")
            
        except Exception as e:
            print(f"Error creating UMAP plot: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    config = [
        {"model_name": "gpt-3.5-turbo", "number_of_models": 3},
        {"model_name": "claude-3-sonnet-20240229", "number_of_models": 2, "temperature": 0.5},
        {"model_name": "gpt-4", "number_of_models": 1}
    ]
    
    generator = Generator(config)
    print("Model info:", generator.get_model_info())
    
    # Generate ensemble responses
    prompt = "Explain a type of science and what it is about"
    responses = generator.generate_response(prompt)
    
    print(f"\nGenerated {len(responses)} responses:")
    for response, model_name in responses:
        print(f"\n{model_name}:")
        print(f"{response[:200]}..." if len(response) > 200 else response)

    clusters, min_distance = generator.cluster_responses(responses)
    generator.plot_clusters(responses, clusters, min_distance)
    generator.plot_umap_clustering(responses, clusters, min_distance)