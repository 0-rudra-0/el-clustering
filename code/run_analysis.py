
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df_adult = pd.read_csv('el_clustering_logs_adult.csv')
    df_diabetes = pd.read_csv('el_clustering_log_diabetes.csv')
    df_bd = pd.read_csv('el_clustering_log_bank.csv')

    datasets = {
        "Adult": df_adult,
        "Diabetes": df_diabetes,
        "Bank": df_bd
    }
    print("Datasets loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: One of the log files was not found. Please ensure the following files are in the same directory: 'el_clustering_logs_adult.csv', 'el_clustering_log_diabetes.csv', 'el_clustering_log_bd.csv'")
except Exception as e:
    print(f"An error occurred during data loading: {e}")

import os

# Define the directory to save the plots
plot_directory = './plots'

# Create the directory if it doesn't exist
os.makedirs(plot_directory, exist_ok=True)

print(f"Plots will be saved to: {plot_directory}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


true_k_values = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 500, 750, 1000]

try:

    df_adult['dataset'] = 'Adult'
    df_diabetes['dataset'] = 'Diabetes'
    df_bd['dataset'] = 'Bank'

    df_combined = pd.concat([df_adult, df_diabetes, df_bd], ignore_index=True)
    df_combined = df_combined[df_combined['k'].isin(true_k_values)].copy()

    df_combined['normalized_gap'] = (df_combined['U'] - df_combined['L']) / df_combined['L']

    datasets = {
        'Adult': df_combined[df_combined['dataset'] == 'Adult'],
        'Diabetes': df_combined[df_combined['dataset'] == 'Diabetes'],
        'Bank': df_combined[df_combined['dataset'] == 'Bank']
    }

    for name, df in datasets.items():
        print(f"\n Generating plots for {name} dataset ")

        print(f"Generating: Impact of 'k' on Cost and Violations for {name}...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Impact of 'k' on Performance ({name} Dataset)", fontsize=16)

        sns.lineplot(data=df, x='k', y='final_cost_S_I', ax=axes[0], marker='o')
        axes[0].set_title('Final Clustering Cost vs. Number of Clusters (k)')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Final Clustering Cost')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True)

        sns.lineplot(data=df, x='k', y='upper_bound_violation_factor', ax=axes[1], marker='o')
        axes[1].set_title('Upper Bound Violation vs. Number of Clusters (k)')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Upper Bound Violation Factor')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{plot_directory}/plot1_k_impact_{name.lower()}.png')
        plt.close(fig)

        print(f"Generating: Effect of Normalized Gap for {name}...")
        fig2 = plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='normalized_gap', y='upper_bound_violation_factor', size='k', alpha=0.7, sizes=(50, 500))
        plt.title(f'Violation Factor vs. Normalized Gap ((U - L) / L) ({name} Dataset)', fontsize=16)
        plt.xlabel('Normalized Gap ((U - L) / L)')
        plt.ylabel('Upper Bound Violation Factor')
        plt.legend(title='k', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{plot_directory}/plot2_normalized_gap_with_k_{name.lower()}.png')
        plt.close(fig2)


except NameError:
    print("DataFrames not found. Please run the data loading cell first.")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_worst_case_costs(df, dataset_name, ax):


    true_k_values = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 500, 750, 1000]


    df_filtered = df[df['k'].isin(true_k_values)].copy()

    if df_filtered.empty:
        print(f"No data found for the specified k values in the {dataset_name} dataset.")
        ax.text(0.5, 0.5, 'No data for specified k-values', ha='center', va='center')
        ax.set_title(f"Worst-Case Cost Analysis for {dataset_name} Dataset")
        return


    df_filtered['max_heuristic_cost'] = df_filtered[['cost_S_L', 'cost_S_U']].max(axis=1)
    df_filtered['cost_increase'] = df_filtered['final_cost_S_I'] - df_filtered['max_heuristic_cost']


    worst_indices = df_filtered.groupby('k')['cost_increase'].idxmax()
    worst_cases_df = df_filtered.loc[worst_indices]


    sns.lineplot(data=worst_cases_df, x='k', y='cost_S_L', ax=ax, marker='o', label='Cost S_L (Lower Bound Heuristic)')
    sns.lineplot(data=worst_cases_df, x='k', y='cost_S_U', ax=ax, marker='o', label='Cost S_U (Upper Bound Heuristic)')
    sns.lineplot(data=worst_cases_df, x='k', y='final_cost_S_I', ax=ax, marker='^', markersize=8, label='Final Cost S_I (Combined)')

    ax.set_title(f"Worst-Case Cost Analysis for {dataset_name} Dataset")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Clustering Cost")
    ax.set_xscale('linear')
    ax.legend()
    ax.grid(True, which="both", ls="--")


try:


    fig, axes = plt.subplots(3, 1, figsize=(15, 24))
    fig.suptitle("Comparison of Heuristic vs. Final Costs in Worst-Case Scenarios", fontsize=20, y=0.95)


    for i, (name, df) in enumerate(datasets.items()):
        print(f"Analyzing and plotting worst cases for {name} dataset...")
        plot_worst_case_costs(df, name, axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(f'{plot_directory}/worst_case_cost_analysis.png')
    plt.close(fig)

except NameError:
    print("DataFrames not found. Please run the data loading cell first.")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_worst_case_costs(df, dataset_name, ax):


    true_k_values = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 500, 750, 1000]


    df_filtered = df[df['k'].isin(true_k_values)].copy()

    if df_filtered.empty:
        print(f"No data found for the specified k values in the {dataset_name} dataset.")
        ax.text(0.5, 0.5, 'No data for specified k-values', ha='center', va='center')
        ax.set_title(f"Worst-Case Cost Analysis for {dataset_name} Dataset")
        return


    df_filtered['max_heuristic_cost'] = df_filtered[['cost_S_L', 'cost_S_U']].max(axis=1)
    df_filtered['cost_increase'] = df_filtered['final_cost_S_I'] - df_filtered['max_heuristic_cost']


    worst_indices = df_filtered.groupby('k')['cost_increase'].idxmax()
    worst_cases_df = df_filtered.loc[worst_indices]


    sns.lineplot(data=worst_cases_df, x='k', y='cost_S_L', ax=ax, marker='o', label='Cost S_L (Lower Bound Heuristic)')
    sns.lineplot(data=worst_cases_df, x='k', y='cost_S_U', ax=ax, marker='o', label='Cost S_U (Upper Bound Heuristic)')
    sns.lineplot(data=worst_cases_df, x='k', y='final_cost_S_I', ax=ax, marker='^', markersize=8, label='Final Cost S_I (Combined)')

    ax.set_title(f"Worst-Case Cost Analysis for {dataset_name} Dataset")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Clustering Cost")
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")


try:


    fig, axes = plt.subplots(3, 1, figsize=(15, 24))
    fig.suptitle("Comparison of Heuristic vs. Final Costs in Worst-Case Scenarios", fontsize=20, y=0.95)


    for i, (name, df) in enumerate(datasets.items()):
        print(f"Analyzing and plotting worst cases for {name} dataset...")
        plot_worst_case_costs(df, name, axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(f'{plot_directory}/worst_case_cost_analysis_log_scale.png')
    plt.close(fig)

except NameError:
    print("DataFrames not found. Please run the data loading cell first.")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_worst_case_absolute_costs(df, dataset_name, linear_ax):

    true_k_values = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 500, 750, 1000]
    df_filtered = df[df['k'].isin(true_k_values)].copy()

    if df_filtered.empty:
        for ax in [linear_ax]:
            ax.text(0.5, 0.5, 'No data for specified k-values', ha='center', va='center')
            ax.set_title(f"Analysis for {dataset_name} Dataset")
        return


    df_filtered['max_heuristic_cost'] = df_filtered[['cost_S_L', 'cost_S_U']].max(axis=1)
    df_filtered['cost_increase'] = df_filtered['final_cost_S_I'] - df_filtered['max_heuristic_cost']
    worst_indices = df_filtered.groupby('k')['cost_increase'].idxmax()
    worst_cases_df = df_filtered.loc[worst_indices]


    def create_plot(ax):

        sns.lineplot(data=worst_cases_df, x='k', y='cost_S_L', ax=ax, marker='o', label='Cost S_L (Heuristic)')
        sns.lineplot(data=worst_cases_df, x='k', y='cost_S_U', ax=ax, marker='o', label='Cost S_U (Heuristic)')
        sns.lineplot(data=worst_cases_df, x='k', y='final_cost_S_I', ax=ax, marker='^', markersize=8, label='Final Cost S_I (Combined)')
        sns.lineplot(data=worst_cases_df, x='k', y='theoretical_bound', ax=ax, color='red', linestyle='--', label='Theoretical Guarantee')
        ax.set_xlabel("Number of Clusters (k)")
        ax.grid(True, which="both", ls="--")
        ax.legend()

    create_plot(linear_ax)
    linear_ax.set_title(f"Absolute Costs for {dataset_name} ")
    linear_ax.set_ylabel("Clustering Cost ")



try:

    fig_linear, axes_linear = plt.subplots(3, 1, figsize=(15, 24))

    fig_linear.suptitle("Worst-Case Scenarios: Absolute Costs (Linear Scale)", fontsize=20, y=0.95)

    for i, (name, df) in enumerate(datasets.items()):
        print(f"Analyzing and plotting absolute worst cases for {name} dataset...")
        plot_worst_case_absolute_costs(df, name, axes_linear[i])

    fig_linear.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig_linear.savefig(f'{plot_directory}/worst_case_absolute_cost_analysis_linear.png')
    plt.close(fig_linear)


except NameError:
    print("DataFrames not found. Please run the data loading cell first.")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_worst_case_normalized_costs_linear(df, dataset_name, ax):

    true_k_values = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 500, 750, 1000]

    df_filtered = df[df['k'].isin(true_k_values)].copy()

    if df_filtered.empty:
        print(f"No data found for the specified k values in the {dataset_name} dataset.")
        ax.text(0.5, 0.5, 'No data for specified k-values', ha='center', va='center')
        ax.set_title(f"Normalized Worst-Case Cost Analysis for {dataset_name} Dataset")
        return

    df_filtered['max_heuristic_cost'] = df_filtered[['cost_S_L', 'cost_S_U']].max(axis=1)
    df_filtered['cost_increase'] = df_filtered['final_cost_S_I'] - df_filtered['max_heuristic_cost']
    worst_indices = df_filtered.groupby('k')['cost_increase'].idxmax()
    worst_cases_df = df_filtered.loc[worst_indices]


    worst_cases_df['cost_S_L_norm'] = (worst_cases_df['cost_S_L'] / worst_cases_df['theoretical_bound']) * 100
    worst_cases_df['cost_S_U_norm'] = (worst_cases_df['cost_S_U'] / worst_cases_df['theoretical_bound']) * 100
    worst_cases_df['final_cost_S_I_norm'] = (worst_cases_df['final_cost_S_I'] / worst_cases_df['theoretical_bound']) * 100


    ax.axhline(100, color='red', linestyle='--', label='Theoretical Bound (100%)')

    ax.set_title(f" Worst-Case Cost comparison with Theoretical Guarantee for {dataset_name} Dataset")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Cost as % of Theoretical Bound")
    ax.set_xscale('linear')
    ax.legend()
    ax.grid(True, which="both", ls="--")


try:

    fig, axes = plt.subplots(3, 1, figsize=(15, 24))
    fig.suptitle("Theoretical Guarantee vs. Final Costs in Worst-Case Scenarios", fontsize=20, y=0.95)

    for i, (name, df) in enumerate(datasets.items()):
        print(f"Analyzing and plotting normalized worst cases for {name} dataset...")
        plot_worst_case_normalized_costs_linear(df, name, axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(f'{plot_directory}/worst_case_normalized_cost_analysis_linear.png')
    plt.close(fig)

except NameError:
    print("DataFrames not found. Please run the data loading cell first.")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def distribution_analysis(df, dataset_name):

    df['max_heuristic_cost'] = df[['cost_S_L', 'cost_S_U']].max(axis=1)

    df['cost_increase_ratio'] = df.apply(
        lambda row: row['final_cost_S_I'] / row['max_heuristic_cost'] if row['max_heuristic_cost'] > 0 else 1,
        axis=1
    )
    df['constraint_gap'] = df['U'] - df['L']

    fig1 = plt.figure(figsize=(12, 7))
    sns.histplot(df['cost_increase_ratio'], kde=True, bins=50)
    p50 = df['cost_increase_ratio'].quantile(0.50)
    p80 = df['cost_increase_ratio'].quantile(0.80)
    p95 = df['cost_increase_ratio'].quantile(0.95)

    plt.axvline(p50, color='orange', linestyle='--', label=f'50th Percentile (Median): {p50:.3f}')
    plt.axvline(p80, color='red', linestyle='--', label=f'80th Percentile: {p80:.3f}')
    plt.axvline(p95, color='black', linestyle='--', label=f'95th Percentile: {p95:.3f}')

    title_str = (
        f"Distribution of Cost Increase Ratio for {dataset_name} Dataset\n"
        f"Statement: 80% of the time, Final Cost <= {p80:.3f} * Max Heuristic Cost"
    )
    plt.title(title_str, fontsize=16)
    plt.xlabel("Cost Increase Ratio (Final Cost / Max Heuristic Cost)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plot_directory}/distribution_cost_increase_ratio_{dataset_name.lower()}.png')
    plt.close(fig1)


    fig2 = plt.figure(figsize=(14, 8))
    k_order = sorted(df['k'].unique())
    ax = sns.boxplot(data=df, x='k', y='cost_increase_ratio', order=k_order)
    plt.title(f"Cost Increase Ratio vs. Number of Clusters (k) for {dataset_name}", fontsize=16)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Cost Increase Ratio")
    plt.xticks(rotation=45)
    plt.grid(True)

    for i, k_val in enumerate(k_order):
        subset = df[df['k'] == k_val]['cost_increase_ratio']
        if not subset.empty:
            median = subset.median()
            mean = subset.mean()
            q1 = subset.quantile(0.25)
            q3 = subset.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
            ax.text(i + 0.05, median, f'{median:.2f}', ha='left', va='center', color='red', fontsize=9, weight='bold')
            ax.text(i - 0.05, mean, f'{mean:.2f}', ha='right', va='center', color='blue', fontsize=9, weight='bold')
            outlier_y_offset = 0.005
            for j, outlier in enumerate(outliers):
                 ax.text(i + 0.05, outlier + (j * outlier_y_offset), f'{outlier:.2f}', ha='left', va='center', color='purple', fontsize=8)

    plt.savefig(f'{plot_directory}/boxplot_cost_increase_ratio_vs_k_{dataset_name.lower()}.png')
    plt.close(fig2)




try:

    for name, df in datasets.items():
        distribution_analysis(df, name)


except NameError:
    print("DataFrames not found. Please run the data loading cell first.")
except Exception as e:
    print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_heatmap_analysis(df_combined):


    df_combined['max_heuristic_cost'] = df_combined[['cost_S_L', 'cost_S_U']].max(axis=1)
    df_combined['cost_increase_ratio'] = df_combined.apply(
        lambda row: row['final_cost_S_I'] / row['max_heuristic_cost'] if row['max_heuristic_cost'] > 0 else 1,
        axis=1
    )
    df_combined['constraint_gap'] = df_combined['U'] - df_combined['L']
    df_combined['normalized_gap'] = (df_combined['U'] - df_combined['L']) / df_combined['L']


    print("Generating: Avg. Violation Heatmaps with Normalized Gap...")
    for dataset_name in df_combined['dataset'].unique():
        plt.figure(figsize=(14, 10))
        df_dataset = df_combined[df_combined['dataset'] == dataset_name]
        pivot_table = df_dataset.pivot_table(
            values='upper_bound_violation_factor',
            index='normalized_gap',
            columns='k',
            aggfunc='mean'
        )

        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Reds", linewidths=.5)
        plt.title(f"Avg. Violation Heatmap for {dataset_name} Dataset\n(Average Upper Bound Violation)", fontsize=16)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Normalized Gap ((U - L) / L)")
        plt.savefig(f'{plot_directory}/plot_adv_2_heatmap_{dataset_name.lower().replace(" ", "_")}.png')
        plt.close()



try:

    df_combined = pd.concat([
        df_adult.assign(dataset='Adult'),
        df_diabetes.assign(dataset='Diabetes'),
        df_bd.assign(dataset='Bank')
    ], ignore_index=True)

    run_heatmap_analysis(df_combined)

except NameError:
    print("DataFrames not found. Please run the data loading cell first.")
except Exception as e:
    print(f"An error occurred: {e}")