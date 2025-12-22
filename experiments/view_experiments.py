"""
Quick script to view and analyze your experiments
Usage:
    python view_experiments.py              # View top 10 experiments
    python view_experiments.py --all        # View all experiments
    python view_experiments.py --best       # Show best experiment
"""

from experiment_tracker import ExperimentTracker
import sys

def main():
    tracker = ExperimentTracker()

    if '--best' in sys.argv:
        print("\n BEST BY TEST ACCURACY:")
        tracker.get_best_experiment(metric='test_acc')

        print("\nBEST BY LEAST OVERFITTING:")
        tracker.get_best_experiment(metric='train_val_gap')

    elif '--all' in sys.argv:
        tracker.view_experiments(top_n=100)

    else:
        # Default: show top 10 by test accuracy
        tracker.view_experiments(top_n=10, sort_by='test_acc')

        print("\n TIPS:")
        print("  - Use 'python view_experiments.py --best' to see best experiments")
        print("  - Use 'python view_experiments.py --all' to see all experiments")
        print("  - Check 'experiments.csv' for full details")

if __name__ == "__main__":
    main()
