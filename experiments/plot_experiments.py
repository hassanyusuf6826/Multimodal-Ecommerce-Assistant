"""
Visualize your experiments with beautiful charts
Usage: python plot_experiments.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_experiments():
    """Create visualization dashboard of all experiments"""

    if not os.path.exists('experiments.csv'):
        print("❌ No experiments.csv found. Run training first!")
        return

    # Load data
    df = pd.read_csv('experiments.csv')

    # Convert to numeric
    numeric_cols = ['dropout_rate', 'learning_rate', 'final_train_acc',
                    'final_val_acc', 'test_acc', 'train_val_gap', 'epochs_trained']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing test_acc
    df_complete = df[df['test_acc'].notna()].copy()

    if len(df_complete) == 0:
        print("⚠️ No experiments with test accuracy yet. Run evaluate_model.py!")
        return

    print(f"📊 Visualizing {len(df_complete)} experiments...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Create subplot layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('🔬 Experiment Analysis Dashboard', fontsize=20, fontweight='bold', y=0.995)

    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    x = range(len(df_complete))
    ax1.plot(x, df_complete['final_train_acc'], 'o-', label='Train', linewidth=2, markersize=8)
    ax1.plot(x, df_complete['final_val_acc'], 's-', label='Validation', linewidth=2, markersize=8)
    ax1.plot(x, df_complete['test_acc'], '^-', label='Test', linewidth=2, markersize=8)
    ax1.set_xlabel('Experiment Number', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy Across Experiments', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # 2. Overfitting Analysis
    ax2 = axes[0, 1]
    colors = ['green' if gap < 0.10 else 'orange' if gap < 0.20 else 'red'
              for gap in df_complete['train_val_gap']]
    ax2.bar(range(len(df_complete)), df_complete['train_val_gap'], color=colors, alpha=0.7)
    ax2.axhline(y=0.15, color='red', linestyle='--', linewidth=2, label='15% threshold')
    ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=2, label='10% threshold')
    ax2.set_xlabel('Experiment Number', fontsize=11)
    ax2.set_ylabel('Train-Val Gap (Overfitting)', fontsize=11)
    ax2.set_title('Overfitting Score by Experiment', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Dropout vs Performance
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df_complete['dropout_rate'], df_complete['test_acc'],
                         c=df_complete['train_val_gap'], cmap='RdYlGn_r',
                         s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Dropout Rate', fontsize=11)
    ax3.set_ylabel('Test Accuracy', fontsize=11)
    ax3.set_title('Dropout Rate vs Test Accuracy', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Overfitting', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Learning Rate vs Performance
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(df_complete['learning_rate'], df_complete['test_acc'],
                          c=df_complete['epochs_trained'], cmap='viridis',
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax4.set_xlabel('Learning Rate', fontsize=11)
    ax4.set_ylabel('Test Accuracy', fontsize=11)
    ax4.set_title('Learning Rate vs Test Accuracy', fontsize=13, fontweight='bold')
    ax4.set_xscale('log')
    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label('Epochs', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Training Efficiency (Epochs vs Accuracy)
    ax5 = axes[2, 0]
    ax5.scatter(df_complete['epochs_trained'], df_complete['test_acc'],
               s=200, alpha=0.7, edgecolors='black', linewidth=1.5, color='purple')
    ax5.set_xlabel('Epochs Trained', fontsize=11)
    ax5.set_ylabel('Test Accuracy', fontsize=11)
    ax5.set_title('Training Efficiency (Fewer epochs = better)', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Performance Distribution
    ax6 = axes[2, 1]
    metrics = ['Train', 'Val', 'Test']
    means = [df_complete['final_train_acc'].mean(),
             df_complete['final_val_acc'].mean(),
             df_complete['test_acc'].mean()]
    colors_bars = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax6.bar(metrics, means, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Mean Accuracy', fontsize=11)
    ax6.set_title('Average Performance Across All Experiments', fontsize=13, fontweight='bold')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('experiment_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved to 'experiment_analysis.png'")

    # Print summary statistics
    print("\n" + "="*70)
    print("📈 SUMMARY STATISTICS")
    print("="*70)
    print(f"Total Experiments:      {len(df_complete)}")
    print(f"Best Test Accuracy:     {df_complete['test_acc'].max():.2%}")
    print(f"Average Test Accuracy:  {df_complete['test_acc'].mean():.2%}")
    print(f"Lowest Overfitting:     {df_complete['train_val_gap'].min():.2%}")
    print(f"Average Overfitting:    {df_complete['train_val_gap'].mean():.2%}")

    # Find best experiment
    best_idx = df_complete['test_acc'].idxmax()
    best_exp = df_complete.loc[best_idx]
    print("\n🏆 BEST EXPERIMENT:")
    print(f"ID:           {best_exp['experiment_id']}")
    print(f"Test Acc:     {best_exp['test_acc']:.2%}")
    print(f"Overfitting:  {best_exp['train_val_gap']:.2%}")
    print(f"Dropout:      {best_exp['dropout_rate']}")
    print(f"Notes:        {best_exp['notes']}")
    print("="*70)

    plt.show()

if __name__ == "__main__":
    plot_experiments()
