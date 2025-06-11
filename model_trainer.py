"""
Machine learning model trainer to improve signal generation over time
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import logging
from utils import setup_logging

logger = setup_logging()

class ModelTrainer:
    """Trains ML models to improve trading signal accuracy"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.model_path = "trading_model.pkl"
        self.performance_threshold = 0.6  # Minimum accuracy to use model
        
        # Load existing model if available
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
                
                logger.info(f"Loaded trained model with {len(self.feature_columns)} features")
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
    
    def save_model(self):
        """Save trained model to disk"""
        if self.model:
            try:
                model_data = {
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'trained_date': datetime.now().isoformat(),
                    'model_type': 'RandomForestClassifier'
                }
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                logger.info("Model saved successfully")
                
            except Exception as e:
                logger.error(f"Error saving model: {e}")
    
    def collect_training_data(self) -> pd.DataFrame:
        """Collect historical trade data for training"""
        training_data = []
        
        # Collect data from all monthly trade files
        for filename in os.listdir('.'):
            if filename.startswith('trade_results_') and filename.endswith('.json'):
                try:
                    with open(filename, 'r') as f:
                        trades = json.load(f)
                    
                    for trade in trades:
                        if trade.get('type') == 'close_position' and 'signal_data' in trade:
                            # Extract features from the original signal
                            signal = trade['signal_data']
                            
                            features = {
                                'sentiment_score': signal.get('sentiment_score', 0),
                                'negative_ratio': signal.get('negative_ratio', 0),
                                'signal_strength': signal.get('signal_strength', 0),
                                'confidence': signal.get('confidence', 0),
                                'price_change': signal.get('price_change', 0),
                                'volume_ratio': signal.get('volume_ratio', 1),
                                'news_count': signal.get('news_count', 0),
                                'position_size': signal.get('position_size', 0)
                            }
                            
                            # Target: was the trade profitable?
                            target = 1 if trade.get('pnl', 0) > 0 else 0
                            
                            # Additional features
                            features['trade_outcome'] = target
                            features['pnl'] = trade.get('pnl', 0)
                            features['pnl_percent'] = trade.get('pnl_percent', 0)
                            features['hold_time_hours'] = trade.get('hold_time_hours', 0)
                            
                            training_data.append(features)
                            
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        if not training_data:
            logger.warning("No training data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(training_data)
        logger.info(f"Collected {len(df)} training samples")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        # Define feature columns (exclude target and outcome columns)
        feature_cols = [
            'sentiment_score', 'negative_ratio', 'signal_strength', 'confidence',
            'price_change', 'volume_ratio', 'news_count', 'position_size'
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            logger.error("No valid features found in training data")
            return np.array([]), np.array([])
        
        self.feature_columns = available_features
        
        # Prepare features and targets
        X = df[self.feature_columns].fillna(0).values
        y = df['trade_outcome'].values
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, y
    
    def train_model(self) -> Dict[str, Any]:
        """Train the machine learning model"""
        logger.info("Starting model training...")
        
        # Collect training data
        df = self.collect_training_data()
        
        if df.empty:
            logger.warning("No training data available")
            return {"status": "no_data", "message": "No training data available"}
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if len(X) == 0:
            logger.warning("No valid features for training")
            return {"status": "no_features", "message": "No valid features"}
        
        if len(np.unique(y)) < 2:
            logger.warning("Not enough variety in outcomes for training")
            return {"status": "insufficient_variety", "message": "Need both profitable and unprofitable trades"}
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails (too few samples), split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        # Calculate trade statistics
        profitable_trades = (df['trade_outcome'] == 1).sum()
        total_trades = len(df)
        baseline_accuracy = max(profitable_trades, total_trades - profitable_trades) / total_trades
        
        results = {
            "status": "success",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "baseline_accuracy": baseline_accuracy,
            "improvement": accuracy - baseline_accuracy,
            "feature_importance": feature_importance,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "profitable_trades": profitable_trades,
            "total_trades": total_trades,
            "model_usable": accuracy > self.performance_threshold
        }
        
        # Save model if it's good enough
        if results["model_usable"]:
            self.save_model()
            logger.info(f"Model trained successfully - Accuracy: {accuracy:.2%}, Improvement: {results['improvement']:.2%}")
        else:
            logger.warning(f"Model accuracy ({accuracy:.2%}) below threshold ({self.performance_threshold:.2%}) - not saving")
        
        return results
    
    def predict_signal_quality(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """Predict if a signal is likely to be profitable"""
        if not self.model or not self.feature_columns:
            return {"probability": 0.5, "confidence": 0.0, "recommendation": "neutral"}
        
        try:
            # Extract features
            features = []
            for col in self.feature_columns:
                value = signal.get(col, 0)
                features.append(value)
            
            # Make prediction
            features_array = np.array([features])
            probability = self.model.predict_proba(features_array)[0][1]  # Probability of success
            
            # Get confidence (distance from 0.5)
            confidence = abs(probability - 0.5) * 2
            
            # Make recommendation
            if probability > 0.7:
                recommendation = "strong_buy"
            elif probability > 0.6:
                recommendation = "buy"
            elif probability < 0.3:
                recommendation = "avoid"
            elif probability < 0.4:
                recommendation = "weak_avoid"
            else:
                recommendation = "neutral"
            
            return {
                "probability": probability,
                "confidence": confidence,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error predicting signal quality: {e}")
            return {"probability": 0.5, "confidence": 0.0, "recommendation": "neutral"}
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about what the model has learned"""
        if not self.model or not self.feature_columns:
            return {"status": "no_model"}
        
        # Feature importance
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Collect recent performance data
        df = self.collect_training_data()
        
        insights = {
            "most_important_features": sorted_importance[:5],
            "feature_count": len(self.feature_columns),
            "model_exists": True
        }
        
        if not df.empty:
            # Calculate feature correlations with profitability
            correlations = {}
            for feature in self.feature_columns:
                if feature in df.columns:
                    corr = df[feature].corr(df['trade_outcome'])
                    if not np.isnan(corr):
                        correlations[feature] = corr
            
            insights["feature_correlations"] = correlations
            
            # Recent performance trends
            df['date'] = pd.to_datetime(df.get('timestamp', ''), errors='coerce')
            recent_df = df[df['date'] > datetime.now() - timedelta(days=30)]
            
            if not recent_df.empty:
                insights["recent_win_rate"] = recent_df['trade_outcome'].mean()
                insights["recent_avg_pnl"] = recent_df['pnl_percent'].mean()
                insights["recent_trades"] = len(recent_df)
        
        return insights

# Global instance
model_trainer = ModelTrainer()