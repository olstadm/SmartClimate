"""
Machine Learning Correction Model for Residual Error Prediction
Complements the physics-based RC model with data-driven corrections
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class MLCorrector:
    """
    Machine learning model to predict residual errors in the RC thermal model
    Learns patterns that the physics model might miss
    """
    
    def __init__(self, config, data_store):
        self.config = config
        self.data_store = data_store
        
        # Model configuration
        self.model_type = config.get('ml_model_type', 'random_forest')
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature configuration
        self.feature_columns = [
            'indoor_temp', 'outdoor_temp', 'indoor_humidity', 'outdoor_humidity',
            'solar_irradiance', 'hour_of_day', 'day_of_week', 'month',
            'temp_diff', 'humidity_diff', 'hvac_heat', 'hvac_cool',
            'base_prediction', 'outdoor_temp_lag1h', 'outdoor_temp_lag3h',
            'indoor_temp_lag1h', 'indoor_temp_change_1h'
        ]
        
        # Model performance tracking
        self.performance_history = []
        
    async def initialize(self):
        """Initialize the ML correction model"""
        logger.info("Initializing ML correction model...")
        
        # Load saved model if available
        await self.load_model()
        
        # If no saved model, try to train one
        if not self.is_trained:
            await self.retrain()
            
        logger.info("ML correction model initialized")
        
    def _create_model(self):
        """Create the ML model based on configuration"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    async def retrain(self):
        """Retrain the ML model on recent data"""
        try:
            logger.info("Retraining ML correction model...")
            
            # Get training data from the last N days
            days_back = self.config.get('ml_training_days', 30)
            training_data = await self.data_store.get_training_data(days_back)
            
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return
                
            # Prepare features and targets
            X, y = self._prepare_training_data(training_data)
            
            if X is None or len(X) < 100:
                logger.warning("Could not prepare sufficient training data")
                return
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            self._create_model()
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store performance metrics
            self.performance_history.append({
                'timestamp': datetime.now(),
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            })
            
            logger.info(f"ML model retrained - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            self.is_trained = True
            
            # Save model
            await self.save_model()
            
            # Analyze feature importance
            if hasattr(self.model, 'feature_importances_'):
                self._log_feature_importance()
                
        except Exception as e:
            logger.error(f"Error retraining ML model: {e}", exc_info=True)
            
    def _prepare_training_data(self, raw_data: List[Dict]) -> tuple:
        """Prepare features and targets from raw data"""
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(raw_data)
            
            # Sort by timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate time-based features
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Calculate differences
            df['temp_diff'] = df['outdoor_temp'] - df['indoor_temp']
            df['humidity_diff'] = df['outdoor_humidity'] - df['indoor_humidity']
            
            # Create HVAC binary indicators
            df['hvac_heat'] = (df['hvac_state'] == 'heat').astype(int)
            df['hvac_cool'] = (df['hvac_state'] == 'cool').astype(int)
            
            # Create lag features
            df['outdoor_temp_lag1h'] = df['outdoor_temp'].shift(12)  # 12 * 5min = 1 hour
            df['outdoor_temp_lag3h'] = df['outdoor_temp'].shift(36)  # 36 * 5min = 3 hours
            df['indoor_temp_lag1h'] = df['indoor_temp'].shift(12)
            df['indoor_temp_change_1h'] = df['indoor_temp'] - df['indoor_temp_lag1h']
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Extract features
            feature_cols = [col for col in self.feature_columns if col in df.columns]
            X = df[feature_cols].values
            
            # Target is the residual error (actual - predicted)
            y = df['residual_error'].values if 'residual_error' in df else np.zeros(len(df))
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}", exc_info=True)
            return None, None
            
    def predict_correction(self, features: Dict) -> float:
        """
        Predict correction to apply to RC model prediction
        
        Args:
            features: Current feature values
            
        Returns:
            Correction value in °C/hour
        """
        if not self.is_trained or self.model is None:
            return 0.0
            
        try:
            # Prepare feature vector
            feature_vector = self._prepare_features(features)
            
            if feature_vector is None:
                return 0.0
                
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            
            # Predict correction
            correction = self.model.predict(feature_vector_scaled)[0]
            
            # Clip to reasonable range
            correction = np.clip(correction, -1.0, 1.0)  # Max ±1°C/hour correction
            
            return float(correction)
            
        except Exception as e:
            logger.error(f"Error predicting correction: {e}", exc_info=True)
            return 0.0
            
    def _prepare_features(self, raw_features: Dict) -> Optional[np.ndarray]:
        """Prepare feature vector from raw input"""
        try:
            # Get current time features
            now = datetime.now()
            
            # Build feature dict
            features = {
                'indoor_temp': raw_features.get('indoor_temp', 20),
                'outdoor_temp': raw_features.get('outdoor_temp', 20),
                'indoor_humidity': raw_features.get('indoor_humidity', 50),
                'outdoor_humidity': raw_features.get('outdoor_humidity', 50),
                'solar_irradiance': raw_features.get('solar_irradiance', 0),
                'hour_of_day': now.hour,
                'day_of_week': now.weekday(),
                'month': now.month,
                'temp_diff': raw_features.get('outdoor_temp', 20) - raw_features.get('indoor_temp', 20),
                'humidity_diff': raw_features.get('outdoor_humidity', 50) - raw_features.get('indoor_humidity', 50),
                'hvac_heat': 1 if raw_features.get('hvac_state') == 'heat' else 0,
                'hvac_cool': 1 if raw_features.get('hvac_state') == 'cool' else 0,
                'base_prediction': raw_features.get('base_prediction', 0),
                # Lag features would come from historical data
                'outdoor_temp_lag1h': raw_features.get('outdoor_temp', 20),  # Simplified
                'outdoor_temp_lag3h': raw_features.get('outdoor_temp', 20),  # Simplified
                'indoor_temp_lag1h': raw_features.get('indoor_temp', 20),    # Simplified
                'indoor_temp_change_1h': 0  # Simplified
            }
            
            # Extract features in correct order
            feature_vector = np.array([features.get(col, 0) for col in self.feature_columns])
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}", exc_info=True)
            return None
            
    def _log_feature_importance(self):
        """Log feature importance for model interpretability"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            logger.info("Top 10 most important features:")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                logger.info(f"  {self.feature_columns[idx]}: {importances[idx]:.4f}")
                
    async def save_model(self):
        """Save the trained model and scaler"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'timestamp': datetime.now().isoformat(),
                'performance': self.performance_history[-1] if self.performance_history else None
            }
            
            await self.data_store.save_model_parameters('ml_corrector', model_data)
            logger.info("ML correction model saved")
            
        except Exception as e:
            logger.error(f"Error saving ML model: {e}", exc_info=True)
            
    async def load_model(self):
        """Load a saved model"""
        try:
            model_data = await self.data_store.load_model_parameters('ml_corrector')
            
            if model_data:
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_columns = model_data.get('feature_columns', self.feature_columns)
                self.model_type = model_data.get('model_type', self.model_type)
                self.is_trained = model_data.get('is_trained', False)
                
                if model_data.get('performance'):
                    self.performance_history.append(model_data['performance'])
                    
                logger.info(f"Loaded ML correction model from {model_data.get('timestamp')}")
            else:
                logger.info("No saved ML model found")
                
        except Exception as e:
            logger.error(f"Error loading ML model: {e}", exc_info=True)
            
    def reset_model(self):
        """Reset ML corrector to initial state"""
        logger.info("Resetting ML correction model...")
        
        # Clear model and state
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.performance_history = []
        
        # Remove saved model file
        model_path = self.data_store.models_dir / f"{self.model_type}_correction.pkl"
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Removed saved model file: {model_path}")
        
        logger.info("ML correction model reset complete")

    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        info = {
            'is_trained': self.is_trained,
            'model_type': self.model_type,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns
        }
        
        if self.performance_history:
            latest = self.performance_history[-1]
            info['latest_performance'] = {
                'mse': latest.get('mse'),
                'mae': latest.get('mae'),
                'r2': latest.get('r2'),
                'training_samples': latest.get('training_samples')
            }
            
        return info