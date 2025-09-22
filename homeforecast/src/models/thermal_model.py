"""
RC Thermal Model with Recursive Least Squares (RLS) Learning
Implements the physics-based thermal dynamics model with adaptive learning
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RecursiveLeastSquares:
    """RLS algorithm for online parameter estimation"""
    
    def __init__(self, n_params: int, forgetting_factor: float = 0.99):
        """
        Initialize RLS algorithm
        
        Args:
            n_params: Number of parameters to estimate
            forgetting_factor: Forgetting factor (0.95-0.99 typical)
        """
        self.n_params = n_params
        self.lambda_ff = forgetting_factor
        
        # Initialize parameters
        self.theta = np.zeros(n_params)  # Parameter vector
        self.P = np.eye(n_params) * 1000  # Covariance matrix
        
    def update(self, phi: np.ndarray, y: float) -> np.ndarray:
        """
        Update parameters with new measurement
        
        Args:
            phi: Feature vector
            y: Measured output
            
        Returns:
            Updated parameter vector
        """
        # Prediction error
        y_pred = np.dot(phi, self.theta)
        e = y - y_pred
        
        # Kalman gain
        den = self.lambda_ff + np.dot(phi, np.dot(self.P, phi))
        K = np.dot(self.P, phi) / den
        
        # Update parameters
        self.theta = self.theta + K * e
        
        # Update covariance matrix
        self.P = (self.P - np.outer(K, np.dot(phi, self.P))) / self.lambda_ff
        
        return self.theta
        

class ThermalModel:
    """
    RC thermal model of home dynamics
    
    Model equation:
    dT_in/dt = a*(T_out - T_in) + k_H*I_h + k_C*I_c + b + k_E*(h_out - h_in) + k_S*Solar
    
    Parameters:
    - a: 1/τ (inverse time constant)
    - k_H: Heating effectiveness
    - k_C: Cooling effectiveness  
    - b: Baseline drift/gains
    - k_E: Enthalpy coupling factor
    - k_S: Solar gain factor
    """
    
    def __init__(self, config, data_store):
        self.config = config
        self.data_store = data_store
        
        # Model parameters (6 parameters total)
        self.n_params = 6
        self.rls = RecursiveLeastSquares(self.n_params)
        
        # Parameter names for reference
        self.param_names = ['a', 'k_H', 'k_C', 'b', 'k_E', 'k_S']
        
        # State tracking
        self.last_update = None
        self.last_temp_in = None
        self.parameter_history = []
        
        # Current sensor readings for API access
        self.current_indoor_temp = None
        self.current_outdoor_temp = None
        self.current_hvac_state = 'unknown'
        self.current_indoor_humidity = None
        self.current_outdoor_humidity = None
        
        # ML correction model (optional)
        self.ml_corrector = None
        self.ml_enabled = config.get('enable_ml_correction', False)
        
    async def initialize(self):
        """Initialize the thermal model"""
        logger.info("Initializing thermal model...")
        
        # Load saved parameters if available
        await self.load_parameters()
        
        # Initialize ML corrector if enabled
        if self.ml_enabled:
            from .ml_corrector import MLCorrector
            self.ml_corrector = MLCorrector(self.config, self.data_store)
            await self.ml_corrector.initialize()
            
        logger.info("Thermal model initialized")
        
    async def update(self, sensor_data: Dict) -> Dict:
        """
        Update model with new sensor data
        
        Args:
            sensor_data: Current sensor readings
            
        Returns:
            Updated model parameters
        """
        try:
            logger.info("🏠 Updating thermal model with sensor data...")
            
            # Extract required values
            t_in = sensor_data['indoor_temp']
            t_out = sensor_data.get('outdoor_temp', t_in)
            h_in = sensor_data.get('indoor_humidity', 50)
            h_out = sensor_data.get('outdoor_humidity', h_in)
            hvac_state = sensor_data.get('hvac_state', 'off')
            solar = sensor_data.get('solar_irradiance', 0)
            
            logger.info(f"Sensor data - Indoor: {t_in}°F, Outdoor: {t_out}°F, Humidity: {h_in}%, HVAC: {hvac_state}")
            
            # Calculate time delta
            current_time = datetime.now()
            if self.last_update and self.last_temp_in is not None:
                dt = (current_time - self.last_update).total_seconds() / 3600  # hours
                
                # Calculate temperature derivative
                dT_dt = (t_in - self.last_temp_in) / dt
                
                # Build feature vector
                phi = self._build_feature_vector(
                    t_in, t_out, h_in, h_out, hvac_state, solar
                )
                
                # Update parameters with RLS
                self.rls.update(phi, dT_dt)
                
                # Store parameter history
                self.parameter_history.append({
                    'timestamp': current_time,
                    'parameters': self.rls.theta.copy(),
                    'prediction_error': dT_dt - np.dot(phi, self.rls.theta)
                })
                
                # Trim history to last 1000 points
                if len(self.parameter_history) > 1000:
                    self.parameter_history = self.parameter_history[-1000:]
                    
                # Save parameters periodically
                if len(self.parameter_history) % 100 == 0:
                    await self.save_parameters()
                    
            # Update state
            self.last_update = current_time
            self.last_temp_in = t_in
            
            # Store current sensor readings for API access
            self.current_indoor_temp = t_in
            self.current_outdoor_temp = t_out  
            self.current_hvac_state = hvac_state
            self.current_indoor_humidity = h_in
            self.current_outdoor_humidity = h_out
            
            logger.info(f"✅ Thermal model updated - stored current values: Indoor={t_in}°F, Outdoor={t_out}°F")
            
            return self.get_parameters()
            
        except Exception as e:
            logger.error(f"Error updating thermal model: {e}", exc_info=True)
            return self.get_parameters()
            
    def _build_feature_vector(self, t_in: float, t_out: float, 
                             h_in: float, h_out: float,
                             hvac_state: str, solar: float) -> np.ndarray:
        """Build feature vector for RLS update"""
        # Convert HVAC state to binary indicators
        i_heat = 1.0 if hvac_state == 'heat' else 0.0
        i_cool = 1.0 if hvac_state == 'cool' else 0.0
        
        # Calculate enthalpy difference (simplified)
        # h = c_p * T + L * w (specific enthalpy)
        # Using psychrometric approximation
        h_diff = self._calculate_enthalpy_diff(t_out, h_out, t_in, h_in)
        
        # Feature vector: [T_out - T_in, I_heat, I_cool, 1, h_diff, solar]
        phi = np.array([
            t_out - t_in,    # Temperature difference
            i_heat,          # Heating indicator
            i_cool,          # Cooling indicator  
            1.0,             # Constant (for baseline drift)
            h_diff,          # Enthalpy difference
            solar / 1000.0   # Solar irradiance (normalized)
        ])
        
        return phi
        
    def _calculate_enthalpy_diff(self, t_out: float, rh_out: float,
                                 t_in: float, rh_in: float) -> float:
        """Calculate enthalpy difference between outdoor and indoor air"""
        # Simplified psychrometric calculation
        # Specific humidity from RH and temperature
        def specific_humidity(T, RH):
            # Saturation vapor pressure (kPa)
            p_sat = 0.6108 * np.exp(17.27 * T / (T + 237.3))
            # Vapor pressure
            p_v = RH / 100.0 * p_sat
            # Specific humidity (kg/kg)
            return 0.622 * p_v / (101.325 - p_v)
            
        w_out = specific_humidity(t_out, rh_out)
        w_in = specific_humidity(t_in, rh_in)
        
        # Enthalpy calculation (kJ/kg)
        h_out = 1.006 * t_out + w_out * (2501 + 1.86 * t_out)
        h_in = 1.006 * t_in + w_in * (2501 + 1.86 * t_in)
        
        return (h_out - h_in) / 100.0  # Scaled
        
    def predict_temperature_change(self, t_in: float, t_out: float,
                                  h_in: float, h_out: float,
                                  hvac_state: str, solar: float) -> float:
        """
        Predict temperature rate of change given current conditions
        
        Returns:
            dT/dt in °C/hour
        """
        phi = self._build_feature_vector(t_in, t_out, h_in, h_out, hvac_state, solar)
        dT_dt = np.dot(phi, self.rls.theta)
        
        # Apply ML correction if available
        if self.ml_corrector and self.ml_enabled:
            correction = self.ml_corrector.predict_correction({
                'indoor_temp': t_in,
                'outdoor_temp': t_out,
                'indoor_humidity': h_in,
                'outdoor_humidity': h_out,
                'hvac_state': hvac_state,
                'solar_irradiance': solar,
                'base_prediction': dT_dt
            })
            dT_dt += correction
            
        return dT_dt
        
    def get_parameters(self) -> Dict:
        """Get current model parameters in interpretable form"""
        theta = self.rls.theta
        
        # Extract and interpret parameters
        a = theta[0]  # 1/τ
        tau = 1 / abs(a) if abs(a) > 0.001 else 1000  # Time constant in hours
        
        params = {
            'time_constant': tau,
            'heating_rate': theta[1],  # °F/hour when heating
            'cooling_rate': -theta[2],  # °F/hour when cooling (positive)
            'baseline_drift': theta[3],  # °F/hour baseline
            'enthalpy_factor': theta[4],
            'solar_factor': theta[5],
            'raw_parameters': theta.tolist()
        }
        
        return params
        
    async def save_parameters(self):
        """Save model parameters to persistent storage"""
        try:
            params = {
                'theta': self.rls.theta.tolist(),
                'P': self.rls.P.tolist(),
                'timestamp': datetime.now().isoformat(),
                'param_names': self.param_names
            }
            
            await self.data_store.save_model_parameters('thermal_model', params)
            logger.debug("Thermal model parameters saved")
            
        except Exception as e:
            logger.error(f"Error saving parameters: {e}", exc_info=True)
            
    async def load_parameters(self):
        """Load saved model parameters"""
        try:
            params = await self.data_store.load_model_parameters('thermal_model')
            
            if params:
                self.rls.theta = np.array(params['theta'])
                self.rls.P = np.array(params['P'])
                logger.info(f"Loaded thermal model parameters from {params['timestamp']}")
            else:
                # Initialize with reasonable defaults
                self._set_default_parameters()
                
        except Exception as e:
            logger.error(f"Error loading parameters: {e}", exc_info=True)
            self._set_default_parameters()
            
    def _set_default_parameters(self):
        """Set reasonable default parameters"""
        # Default values based on typical home characteristics (Fahrenheit)
        self.rls.theta = np.array([
            0.1,    # a: 1/τ (τ = 10 hours typical)
            3.6,    # k_H: 3.6°F/hour heating rate (2°C/hour * 1.8)
            -4.5,   # k_C: -4.5°F/hour cooling rate (-2.5°C/hour * 1.8)
            0.09,   # b: Small baseline drift (0.05°C/hour * 1.8)
            0.036,  # k_E: Small enthalpy effect (0.02°C/hour * 1.8)
            0.9     # k_S: Moderate solar gain (0.5°C/hour * 1.8)
        ])
        logger.info("Using default thermal model parameters")
        
    async def retrain_ml_correction(self):
        """Retrain the ML correction model"""
        if self.ml_corrector and self.ml_enabled:
            logger.info("Retraining ML correction model...")
            await self.ml_corrector.retrain()
            logger.info("ML correction model retrained")
            
    def get_model_quality_metrics(self) -> Dict:
        """Calculate model quality metrics from recent predictions"""
        if len(self.parameter_history) < 10:
            return {
                'mse': None,
                'mae': None,
                'r2': None,
                'sample_size': len(self.parameter_history)
            }
            
        # Calculate prediction errors from recent history
        errors = [p['prediction_error'] for p in self.parameter_history[-100:]]
        errors = np.array(errors)
        
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        
        # R-squared would require actual vs predicted over time
        # For now, use error variance as quality indicator
        error_var = np.var(errors)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'error_variance': float(error_var),
            'sample_size': len(errors),
            'parameter_convergence': self._check_parameter_convergence()
        }
        
    def _check_parameter_convergence(self) -> bool:
        """Check if parameters have converged"""
        if len(self.parameter_history) < 50:
            return False
            
        # Check variance of parameters over last 50 updates
        recent_params = np.array([p['parameters'] for p in self.parameter_history[-50:]])
        param_std = np.std(recent_params, axis=0)
        
        # Parameters considered converged if std < 0.01
        return np.all(param_std < 0.01)