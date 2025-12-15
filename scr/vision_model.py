import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import config
from experiment_tracker import ExperimentTracker  # ADD THIS

class ProductClassifier:
    def __init__(self):
        self.model = None
        self.classes = None
        self.tracker = ExperimentTracker()  # ADD THIS

    def train(self, experiment_notes=""):
        """Trains the model with BALANCED regularization."""

        # Store configuration for logging
        model_config = {
            'model_type': 'MobileNetV2',
            'base_frozen': True,  # Change if you unfreeze base
            'dropout_rate': 0.3,
            'dense_units': 128,
            'l2_reg': 0.001,
            'learning_rate': 0.001,
            'batch_size': config.BATCH_SIZE,
            'augmentation_level': 'moderate'
        }

        # 1. MODERATE Data Augmentation (was too aggressive)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,      # Reduced from 40
            width_shift_range=0.15, # Reduced from 0.3
            height_shift_range=0.15,
            zoom_range=0.15,        # Reduced from 0.3
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        print("Preparing data generators...")
        train_gen = train_datagen.flow_from_directory(
            config.IMAGE_DIR, target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE, subset='training',
            class_mode='categorical'
        )

        val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        val_gen = val_datagen.flow_from_directory(
            config.IMAGE_DIR, target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE, subset='validation',
            class_mode='categorical'
        )

        self.classes = list(train_gen.class_indices.keys())
        with open(config.CLASS_NAMES_FILE, 'wb') as f:
            pickle.dump(self.classes, f)

        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")

        # 2. Build Model with REDUCED Regularization
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=config.IMG_SIZE + (3,))

        # OPTION 1: Keep base frozen initially (safer for small datasets)
        base_model.trainable = False

        # OPTION 2: Unfreeze base (try this if Option 1 still fails)
        # base_model.trainable = True
        # model_config['base_frozen'] = False  # Update config if you unfreeze

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # REDUCED regularization
        x = Dropout(0.3)(x)  # Reduced from 0.5
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # Increased units, reduced L2
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)  # Reduced from 0.5

        outputs = Dense(len(self.classes), activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=outputs)

        # Standard learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Increased from 0.0005
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # 3. Callbacks with more patience
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]

        print("\n=== Starting Training ===")
        print("Model summary:")
        self.model.summary()

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,  # More epochs since we have early stopping
            callbacks=callbacks,
            verbose=1
        )

        self.model.save(config.MODEL_PATH)
        print(f"\nModel saved to {config.MODEL_PATH}")

        # Print final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")

        self.plot_performance(history)

        # NEW: Log experiment (without test accuracy yet)
        exp_id = self.tracker.log_experiment(
            model_config=model_config,
            history=history,
            test_accuracy=None,  # Will add later from evaluate_model.py
            notes=experiment_notes
        )

        print(f"\n✅ Experiment logged as: {exp_id}")
        print("💡 Run evaluate_model.py to get test accuracy and update the log!")

        return history

    def plot_performance(self, history):
        """Plots accuracy and loss graphs."""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
        plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--', linewidth=2)
        plt.legend(loc='lower right')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
        plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle='--', linewidth=2)
        plt.legend(loc='upper right')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_performance.png', dpi=300)
        print("Performance plots saved to 'training_performance.png'")

if __name__ == "__main__":
    classifier = ProductClassifier()
    if os.path.exists(config.IMAGE_DIR):
        # Add notes about this experiment
        notes = "Baseline with moderate augmentation, frozen base, dropout 0.3"
        classifier.train(experiment_notes=notes)
    else:
        print(f"Error: Image directory '{config.IMAGE_DIR}' not found.")
