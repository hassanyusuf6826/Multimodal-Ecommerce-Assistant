import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os
import csv
from datetime import datetime

def update_experiment_with_test_accuracy(test_accuracy):
    """Update the most recent experiment with test accuracy"""
    log_file = 'experiments.csv'

    if not os.path.exists(log_file):
        print("⚠️ No experiment log found. Train a model first!")
        return

    # Read all experiments
    rows = []
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if len(rows) == 0:
        print("⚠️ No experiments logged yet!")
        return

    # Update the most recent experiment (last row)
    last_row = rows[-1]
    test_acc_index = headers.index('test_acc')
    last_row[test_acc_index] = f"{test_accuracy:.4f}"
    rows[-1] = last_row

    # Write back
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    exp_id = last_row[0]  # First column is experiment_id
    print(f"\n✅ Updated experiment {exp_id} with test accuracy: {test_accuracy:.2%}")

def evaluate():
    # 1. Check if model exists
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model not found at {config.MODEL_PATH}. Train it first!")
        return

    print("Loading model...")
    model = load_model(config.MODEL_PATH)

    # 2. Prepare Data Generator (Use the same split as training)
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    print("Loading test data (from validation split)...")
    test_gen = test_datagen.flow_from_directory(
        config.IMAGE_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # 3. Run Evaluation
    print("\n--- Calculating Accuracy ---")
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # 4. Update experiment log with test accuracy
    update_experiment_with_test_accuracy(accuracy)

    # 5. Generate Predictions for Confusion Matrix
    print("\n--- Generating Confusion Matrix ---")
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # 6. Classification Report (Precision & Recall)
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print("\nClassification Report:")
    print(report)

    # 7. Plot Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Test Acc: {accuracy:.2%})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n✅ Confusion Matrix saved to 'confusion_matrix.png'")
    plt.show()

    # 8. Show performance summary
    print("\n" + "="*60)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Test Accuracy:  {accuracy:.2%}")
    print(f"Test Loss:      {loss:.4f}")
    print("="*60)

    return accuracy

if __name__ == "__main__":
    evaluate()
