import csv
import os
from datetime import datetime
import json

class ExperimentTracker:
    def __init__(self, log_file='experiments.csv'):
        self.log_file = log_file
        self._initialize_log()

    def _initialize_log(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'experiment_id',
                    'date',
                    'model_type',
                    'base_frozen',
                    'dropout_rate',
                    'dense_units',
                    'l2_regularization',
                    'learning_rate',
                    'batch_size',
                    'augmentation',
                    'final_train_acc',
                    'final_val_acc',
                    'test_acc',
                    'train_val_gap',
                    'overfitting_score',
                    'epochs_trained',
                    'notes'
                ])
            print(f"✅ Created experiment log: {self.log_file}")

    def log_experiment(self, model_config, history, test_accuracy=None, notes=""):
        """
        Log an experiment to CSV

        Args:
            model_config: dict with model configuration
            history: Keras training history object
            test_accuracy: float, test set accuracy (optional)
            notes: str, additional notes about the experiment
        """
        # Generate experiment ID
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract metrics from history
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        train_val_gap = final_train_acc - final_val_acc
        epochs_trained = len(history.history['accuracy'])

        # Calculate overfitting score (higher = worse overfitting)
        overfitting_score = train_val_gap * 100  # As percentage

        # Prepare row data
        row = [
            exp_id,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            model_config.get('model_type', 'MobileNetV2'),
            model_config.get('base_frozen', True),
            model_config.get('dropout_rate', 0.3),
            model_config.get('dense_units', 128),
            model_config.get('l2_reg', 0.001),
            model_config.get('learning_rate', 0.001),
            model_config.get('batch_size', 32),
            model_config.get('augmentation_level', 'moderate'),
            f"{final_train_acc:.4f}",
            f"{final_val_acc:.4f}",
            f"{test_accuracy:.4f}" if test_accuracy else "N/A",
            f"{train_val_gap:.4f}",
            f"{overfitting_score:.2f}",
            epochs_trained,
            notes
        ]

        # Append to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Print summary
        print("\n" + "="*70)
        print(f"📊 EXPERIMENT LOGGED: {exp_id}")
        print("="*70)
        print(f"Training Accuracy:   {final_train_acc:.2%}")
        print(f"Validation Accuracy: {final_val_acc:.2%}")
        if test_accuracy:
            print(f"Test Accuracy:       {test_accuracy:.2%}")
        print(f"Train-Val Gap:       {train_val_gap:.2%} {'⚠️ OVERFITTING!' if train_val_gap > 0.15 else '✅ Good'}")
        print(f"Epochs Trained:      {epochs_trained}")
        print(f"Notes:               {notes}")
        print("="*70)
        print(f"✅ Logged to {self.log_file}\n")

        return exp_id

    def view_experiments(self, top_n=10, sort_by='test_acc'):
        """
        View experiment results

        Args:
            top_n: int, number of experiments to show
            sort_by: str, column to sort by ('test_acc', 'val_acc', 'train_val_gap')
        """
        if not os.path.exists(self.log_file):
            print("No experiments logged yet!")
            return

        import pandas as pd
        df = pd.read_csv(self.log_file)

        # Convert to float for sorting
        df['test_acc'] = pd.to_numeric(df['test_acc'], errors='coerce')
        df['final_val_acc'] = pd.to_numeric(df['final_val_acc'], errors='coerce')
        df['train_val_gap'] = pd.to_numeric(df['train_val_gap'], errors='coerce')

        # Sort
        if sort_by == 'train_val_gap':
            df_sorted = df.sort_values(sort_by, ascending=True)  # Lower is better
        else:
            df_sorted = df.sort_values(sort_by, ascending=False)  # Higher is better

        print("\n" + "="*100)
        print(f"📈 TOP {top_n} EXPERIMENTS (sorted by {sort_by})")
        print("="*100)

        # Select key columns
        display_cols = [
            'experiment_id', 'date', 'dropout_rate', 'learning_rate',
            'final_train_acc', 'final_val_acc', 'test_acc',
            'train_val_gap', 'epochs_trained', 'notes'
        ]

        print(df_sorted[display_cols].head(top_n).to_string(index=False))
        print("="*100 + "\n")

        return df_sorted

    def compare_experiments(self, exp_ids):
        """Compare specific experiments side by side"""
        import pandas as pd
        df = pd.read_csv(self.log_file)

        comparison = df[df['experiment_id'].isin(exp_ids)]

        if len(comparison) == 0:
            print("No matching experiments found!")
            return

        print("\n" + "="*100)
        print("🔍 EXPERIMENT COMPARISON")
        print("="*100)
        print(comparison.to_string(index=False))
        print("="*100 + "\n")

        return comparison

    def get_best_experiment(self, metric='test_acc'):
        """Get the best performing experiment"""
        import pandas as pd
        df = pd.read_csv(self.log_file)

        # Convert to float
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

        if metric == 'train_val_gap':
            best = df.loc[df[metric].idxmin()]  # Lower is better
        else:
            best = df.loc[df[metric].idxmax()]  # Higher is better

        print("\n" + "="*70)
        print(f"🏆 BEST EXPERIMENT (by {metric})")
        print("="*70)
        for key, value in best.items():
            print(f"{key:20s}: {value}")
        print("="*70 + "\n")

        return best


# Convenience function
def log_experiment(model_config, history, test_accuracy=None, notes=""):
    """Quick function to log an experiment"""
    tracker = ExperimentTracker()
    return tracker.log_experiment(model_config, history, test_accuracy, notes)


def view_experiments(top_n=10, sort_by='test_acc'):
    """Quick function to view experiments"""
    tracker = ExperimentTracker()
    return tracker.view_experiments(top_n, sort_by)


def get_best_experiment(metric='test_acc'):
    """Quick function to get best experiment"""
    tracker = ExperimentTracker()
    return tracker.get_best_experiment(metric)
