"""
LULC Classification - Complete Pipeline
Automated Land Use and Land Cover Classification Using Satellite Images
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import SatelliteDataPreprocessor
from src.model_architecture import LULCClassifier
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.explainability import ExplainableAI

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'results/graphs',
        'results/confusion_matrix',
        'results/predictions',
        'results/explainability'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Project directories created")

def run_preprocessing(data_path, img_size=(64, 64)):
    """Run data preprocessing pipeline"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = SatelliteDataPreprocessor(
        data_path=data_path,
        img_size=img_size
    )
    
    # Load dataset
    X, y, classes = preprocessor.load_dataset()
    
    # Visualize samples
    print("\nVisualizing sample images...")
    preprocessor.visualize_samples(X, y, classes)
    
    # Normalize
    X_normalized = preprocessor.normalize_images(X)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X_normalized, y, test_size=0.2, val_size=0.1
    )
    
    # Create augmentation pipeline
    datagen = preprocessor.create_data_augmentation()
    
    # Visualize augmentation
    print("\nVisualizing augmented images...")
    preprocessor.visualize_augmentation(X_train, y_train, datagen)
    
    # Save preprocessed data
    import numpy as np
    np.savez_compressed(
        'data/processed/preprocessed_data.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        classes=classes
    )
    
    print("\n✓ Preprocessing complete!")
    return len(classes)

def run_training(num_classes, img_size=(64, 64), epochs=50, batch_size=32):
    """Run model training pipeline"""
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    # Build model
    classifier = LULCClassifier(
        input_shape=(*img_size, 3),
        num_classes=num_classes
    )
    
    model = classifier.build_custom_cnn()
    model = classifier.compile_model(model, learning_rate=0.001)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    trainer = ModelTrainer(model)
    history = trainer.train_with_augmentation(epochs=epochs, batch_size=batch_size)
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\n✓ Training complete!")

def run_evaluation():
    """Run model evaluation pipeline"""
    print("\n" + "="*60)
    print("STEP 3: MODEL EVALUATION")
    print("="*60)
    
    evaluator = ModelEvaluator(
        model_path='models/saved_models/best_model.h5'
    )
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    # Generate visualizations
    print("\nGenerating confusion matrix...")
    evaluator.plot_confusion_matrix()
    
    print("\nPlotting per-class metrics...")
    evaluator.plot_per_class_metrics()
    
    print("\nVisualizing predictions...")
    evaluator.visualize_predictions()
    
    print("\nAnalyzing errors...")
    evaluator.analyze_errors()
    
    print("\n✓ Evaluation complete!")
    return metrics

def run_explainability():
    """Run explainability analysis"""
    print("\n" + "="*60)
    print("STEP 4: EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    explainer = ExplainableAI(
        model_path='models/saved_models/best_model.h5'
    )
    
    # Grad-CAM
    print("\nGenerating Grad-CAM visualizations...")
    explainer.visualize_gradcam(n_samples=6)
    
    # Feature maps
    print("\nVisualizing feature maps...")
    explainer.visualize_feature_maps(img_idx=0, layer_indices=[1, 5, 10])
    
    # Attention analysis
    print("\nAnalyzing model attention...")
    explainer.analyze_model_attention()
    
    print("\n✓ Explainability analysis complete!")

def print_summary(metrics):
    """Print final summary"""
    print("\n" + "="*60)
    print("PROJECT SUMMARY")
    print("="*60)
    
    print("\n📊 Final Metrics:")
    print(f"   • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   • Precision: {metrics['precision']:.4f}")
    print(f"   • Recall:    {metrics['recall']:.4f}")
    print(f"   • F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\n📁 Generated Files:")
    print("   • Trained Model: models/saved_models/best_model.h5")
    print("   • Training Plots: results/graphs/")
    print("   • Confusion Matrix: results/confusion_matrix/")
    print("   • Predictions: results/predictions/")
    print("   • Explainability: results/explainability/")
    
    print("\n✓ All tasks completed successfully!")
    print("="*60 + "\n")

def main():
    """Main pipeline execution"""
    print("\n" + "="*60)
    print("LULC CLASSIFICATION PROJECT")
    print("Automated Land Use and Land Cover Classification")
    print("="*60 + "\n")
    
    # Configuration
    DATA_PATH = 'data/raw/EuroSAT_RGB'  # Path to your dataset
    IMG_SIZE = (64, 64)
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Create directories
    create_directories()
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\n⚠️  ERROR: Dataset not found at {DATA_PATH}")
        print("\nPlease make sure your dataset is extracted to:")
        print(f"   {DATA_PATH}")
        print("\nYour dataset should have these folders:")
        print("   - AnnualCrop, Forest, HerbaceousVegetation, Highway,")
        print("   - Industrial, Pasture, PermanentCrop, Residential, River, SeaLake")
        sys.exit(1)
    
    try:
        # Step 1: Preprocessing
        num_classes = run_preprocessing(DATA_PATH, IMG_SIZE)
        
        # Step 2: Training
        run_training(num_classes, IMG_SIZE, EPOCHS, BATCH_SIZE)
        
        # Step 3: Evaluation
        metrics = run_evaluation()
        
        # Step 4: Explainability
        run_explainability()
        
        # Print summary
        print_summary(metrics)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()