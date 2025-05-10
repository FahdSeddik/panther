import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_benchmark_results(file_paths, model_names=None):
    """
    Load multiple benchmark result CSV files into a single DataFrame.
    
    Args:
        file_paths: List of paths to CSV files
        model_names: Optional list of names to use instead of filenames
        
    Returns:
        Combined DataFrame with all results
    """
    all_data = []
    
    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)
        
        # If model names provided, use them instead
        if model_names and i < len(model_names):
            df['source'] = model_names[i]
        else:
            # Use filename without extension as model source
            filename = os.path.splitext(os.path.basename(file_path))[0]
            df['source'] = filename
            
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def create_param_label(row):
    """Create a concise parameter label for plots"""
    return f"{row['in_features']}x{row['out_features']}, l={row['num_terms']}, k={row['low_rank']}"

def visualize_model_performance(df, output_dir=".", max_params_per_plot=40):
    """Generate a focused visualization of performance by model across all parameter combinations
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save output plots
        max_params_per_plot: Maximum number of parameter combinations to show in a single plot
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create separate directories for forward and backward passes
    forward_dir = os.path.join(output_dir, "forward")
    backward_dir = os.path.join(output_dir, "backward")
    total_dir = os.path.join(output_dir, "total")
    
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(backward_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)
    
    # Filter out invalid combinations and errors
    valid_df = df[df['is_valid'] == True].copy()
    if 'error' in df.columns:
        valid_df = valid_df[valid_df['error'].isna() | (valid_df['error'] == '')]
    valid_df = valid_df.dropna(subset=['forward_mean_ms', 'backward_mean_ms'])
    
    # Add a parameter label column for easier identification
    valid_df['param_label'] = valid_df.apply(create_param_label, axis=1)
    
    # Add a size group column for grouping by input/output dimensions
    valid_df['size_group'] = valid_df.apply(lambda row: f"{row['in_features']}x{row['out_features']}", axis=1)
    
    # Get unique size groups and sort them
    size_groups = sorted(valid_df['size_group'].unique())
    
    # For each group, get its parameter labels
    grouped_param_labels = []
    for group in size_groups:
        group_labels = valid_df[valid_df['size_group'] == group]['param_label'].unique()
        grouped_param_labels.extend(group_labels)
    
    # Get sources
    sources = valid_df['source'].unique()
    
    # Calculate how many plots we need
    num_params = len(grouped_param_labels)
    num_plots = (num_params + max_params_per_plot - 1) // max_params_per_plot  # Ceiling division
    
    # Create partitioned forward pass plots
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_params_per_plot
        end_idx = min((plot_idx + 1) * max_params_per_plot, num_params)
        current_labels = grouped_param_labels[start_idx:end_idx]
        
        plt.figure(figsize=(16, 10))
        
        for source in sources:
            source_data = valid_df[(valid_df['source'] == source) & 
                                  (valid_df['param_label'].isin(current_labels))]
            source_data = source_data.sort_values('param_label')
            
            if not source_data.empty:
                plt.plot(source_data['param_label'], source_data['forward_mean_ms'], 
                         marker='o', linewidth=2, label=source)
        
        plt.title(f'Forward Pass Performance (Part {plot_idx+1}/{num_plots})', fontsize=16)
        plt.ylabel('Forward Time (ms) - lower is better', fontsize=14)
        plt.xlabel('Parameter Combinations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=90)  # Vertical x-axis labels
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{forward_dir}/performance_part{plot_idx+1}.png", dpi=300)
        plt.close()
    
    # Create partitioned backward pass plots
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_params_per_plot
        end_idx = min((plot_idx + 1) * max_params_per_plot, num_params)
        current_labels = grouped_param_labels[start_idx:end_idx]
        
        plt.figure(figsize=(16, 10))
        
        for source in sources:
            source_data = valid_df[(valid_df['source'] == source) & 
                                  (valid_df['param_label'].isin(current_labels))]
            source_data = source_data.sort_values('param_label')
            
            if not source_data.empty:
                plt.plot(source_data['param_label'], source_data['backward_mean_ms'], 
                         marker='o', linewidth=2, label=source)
        
        plt.title(f'Backward Pass Performance (Part {plot_idx+1}/{num_plots})', fontsize=16)
        plt.ylabel('Backward Time (ms) - lower is better', fontsize=14)
        plt.xlabel('Parameter Combinations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=90)  # Vertical x-axis labels
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{backward_dir}/performance_part{plot_idx+1}.png", dpi=300)
        plt.close()
    
    # Create partitioned total (forward+backward) plots
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_params_per_plot
        end_idx = min((plot_idx + 1) * max_params_per_plot, num_params)
        current_labels = grouped_param_labels[start_idx:end_idx]
        
        plt.figure(figsize=(16, 10))
        
        for source in sources:
            source_data = valid_df[(valid_df['source'] == source) & 
                                  (valid_df['param_label'].isin(current_labels))]
            source_data = source_data.sort_values('param_label')
            
            if not source_data.empty:
                total_time = source_data['forward_mean_ms'] + source_data['backward_mean_ms']
                plt.plot(source_data['param_label'], total_time, 
                        marker='o', linewidth=2, label=f"{source} (Total)")
        
        plt.title(f'Total (Forward + Backward) Performance (Part {plot_idx+1}/{num_plots})', fontsize=16)
        plt.ylabel('Total Time (ms) - lower is better', fontsize=14)
        plt.xlabel('Parameter Combinations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=90)  # Vertical x-axis labels
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{total_dir}/performance_part{plot_idx+1}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    # Specify the paths to your benchmark result CSV files
    result_files = [
        "/kaggle/input/linear-times/torch_linear.csv",
        "/kaggle/input/linear-times/sk_linear.csv",
        # "sk_linear_tr.csv",
        # "sk_linear_cuda.csv",
    ]
    
    # Optional: Provide custom names for each result file
    model_names = [
        "torch_linear",
        "sk_linear",
        # "sk_linear_tr",
        # "sk_linear_cuda",
    ]
    
    # Load and combine all results
    combined_df = load_benchmark_results(result_files, model_names)
    
    # Generate focused model performance visualization
    visualize_model_performance(combined_df, output_dir="benchmark_visualizations")
    
    print(f"Visualizations generated in the 'benchmark_visualizations' directory")