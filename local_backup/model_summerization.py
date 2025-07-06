import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import itertools

def plot_all_metrics_vs_dataset_size(df: pd.DataFrame, output_path=None):
    df.columns = df.columns.str.strip()

    numeric_metrics = [
        col for col in df.select_dtypes(include='number').columns
        if col != 'dataset_size'
    ]

    model_names = df['model_name'].unique()
    colors = cm.get_cmap('tab20', len(numeric_metrics))
    linestyles = itertools.cycle(['-', '--', '-.', ':'])
    markers = itertools.cycle(('o', 'v', 's', 'D', '^', '<', '>', '*', 'x', '+'))

    plt.figure(figsize=(14, 8))

    for idx, metric in enumerate(numeric_metrics):
        for model in model_names:
            model_df = df[df['model_name'] == model]
            sorted_df = model_df.sort_values(by='dataset_size')
            label = f"{metric} | {model}"
            plt.plot(
                sorted_df['dataset_size'],
                sorted_df[metric],
                label=label,
                linestyle=next(linestyles),
                marker=next(markers),
                linewidth=2
            )

    plt.title("Metrics vs Dataset Size per Model")
    plt.xlabel("Dataset Size")
    plt.ylabel("Metric Value")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize="small")
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
