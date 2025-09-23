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
        logger.info("🏠 Initializing thermal model...")
        logger.info(f"📊 ML correction enabled: {self.ml_enabled}")
        
        # Load saved parameters if available
        await self.load_parameters()
        
        # Initialize ML corrector if enabled
        if self.ml_enabled:
            logger.info("🤖 Initializing ML corrector...")
            from .ml_corrector import MLCorrector
            self.ml_corrector = MLCorrector(self.config, self.data_store)
            await self.ml_corrector.initialize()
            logger.info("✅ ML corrector initialized")
        else:
            logger.info("⚠️ ML correction is disabled in configuration")
            
        logger.info("✅ Thermal model initialized")
        
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

    def analyze_temperature_trends(self, historical_weather: List[Dict], current_conditions: Dict) -> Dict:
        """
        Analyze temperature trends from historical data to inform predictions
        
        Args:
            historical_weather: List of historical weather data points
            current_conditions: Current sensor/weather conditions
            
        Returns:
            Dict with trend analysis results
        """
        try:
            logger.info("📈 Analyzing temperature trends from available data...")
            
            # Try to use recent historical data for trend analysis
            timestamps = []
            outdoor_temps = []
            
            if historical_weather and len(historical_weather) >= 2:
                logger.info("Using historical weather data for trend analysis")
                for point in historical_weather[-12:]:  # Use last 12 points only
                    if isinstance(point, dict) and 'timestamp' in point and 'temperature' in point:
                        try:
                            temp = float(point['temperature'])
                            if -50 <= temp <= 150:  # Reasonable bounds
                                timestamps.append(point['timestamp'])
                                outdoor_temps.append(temp)
                        except (ValueError, TypeError):
                            continue
            
            # If insufficient historical data, check if we have current conditions for basic trend
            if len(outdoor_temps) < 2:
                logger.info("Insufficient historical data - using current conditions for basic trend")
                current_temp = current_conditions.get('outdoor_temp')
                if current_temp and isinstance(current_temp, (int, float)) and -50 <= current_temp <= 150:
                    now = datetime.now()
                    timestamps = [now - timedelta(hours=1), now]  # Create 1-hour baseline
                    outdoor_temps = [current_temp, current_temp]  # Flat trend assumption
            
            if len(outdoor_temps) < 2:
                logger.warning("⚠️ Insufficient temperature data points")
                return self._get_fallback_trend_analysis()
            
            # Calculate outdoor temperature trend (°F/hour)
            # Normalize timestamps to handle timezone-aware vs naive datetime mixing
            normalized_timestamps = []
            reference_ts = timestamps[0]
            
            for ts in timestamps:
                try:
                    # Convert both timestamps to same timezone awareness
                    if hasattr(reference_ts, 'tzinfo') and reference_ts.tzinfo is not None:
                        # Reference is timezone-aware
                        if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                            # Convert naive to UTC
                            import pytz
                            ts = pytz.UTC.localize(ts)
                    else:
                        # Reference is naive
                        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                            # Convert timezone-aware to naive (use local time)
                            ts = ts.replace(tzinfo=None)
                    
                    normalized_timestamps.append(ts)
                except Exception as e:
                    logger.debug(f"Timestamp normalization error: {e}")
                    # Fallback - use timestamp as-is
                    normalized_timestamps.append(ts)
            
            time_hours = [(ts - normalized_timestamps[0]).total_seconds() / 3600 for ts in normalized_timestamps]
            outdoor_trend = np.polyfit(time_hours, outdoor_temps, 1)[0] if len(time_hours) > 1 else 0
            
            # Calculate temperature variation and stability
            outdoor_std = np.std(outdoor_temps)
            outdoor_range = max(outdoor_temps) - min(outdoor_temps)
            
            # Predict expected indoor response based on outdoor trends
            current_outdoor = current_conditions.get('outdoor_temp', 70.0)
            current_indoor = current_conditions.get('indoor_temp', 70.0)
            
            # Use thermal model parameters to predict indoor response
            thermal_coupling = abs(self.rls.theta[0]) if len(self.rls.theta) > 0 else 0.1  # 'a' parameter
            expected_indoor_trend = outdoor_trend * thermal_coupling * 0.5  # Damped response
            
            # Calculate trend alignment score (how well indoor should follow outdoor)
            if abs(outdoor_trend) > 0.5:  # Significant outdoor trend
                alignment_strength = "strong" if abs(outdoor_trend) > 2.0 else "moderate"
            else:
                alignment_strength = "weak"
            
            trend_analysis = {
                'outdoor_trend': {
                    'rate_per_hour': round(outdoor_trend, 3),
                    'direction': 'rising' if outdoor_trend > 0.2 else 'falling' if outdoor_trend < -0.2 else 'stable',
                    'stability': 'stable' if outdoor_std < 2.0 else 'variable' if outdoor_std < 5.0 else 'unstable',
                    'temperature_range': round(outdoor_range, 1)
                },
                'expected_indoor_response': {
                    'predicted_trend': round(expected_indoor_trend, 3),
                    'coupling_strength': alignment_strength,
                    'thermal_lag_hours': round(1 / thermal_coupling, 1) if thermal_coupling > 0 else 8.0
                },
                'trend_validation': {
                    'outdoor_rising_indoor_should_follow': outdoor_trend > 0.5,
                    'outdoor_falling_indoor_should_follow': outdoor_trend < -0.5,
                    'significant_trend_detected': abs(outdoor_trend) > 0.5,
                    'coupling_factor': thermal_coupling
                },
                'data_quality': {
                    'historical_points': len(historical_weather),
                    'time_span_hours': round(max(time_hours) - min(time_hours), 1),
                    'temperature_reliability': 'high' if outdoor_std < 3.0 else 'medium' if outdoor_std < 6.0 else 'low'
                }
            }
            
            logger.info(f"✅ Trend analysis complete - Outdoor: {outdoor_trend:+.2f}°F/hr, Expected indoor response: {expected_indoor_trend:+.2f}°F/hr")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}", exc_info=True)
            return self._get_fallback_trend_analysis()

    def _get_fallback_trend_analysis(self) -> Dict:
        """Return fallback trend analysis when data is insufficient"""
        return {
            'outdoor_trend': {
                'rate_per_hour': 0.0,
                'direction': 'stable',
                'stability': 'unknown',
                'temperature_range': 0.0
            },
            'expected_indoor_response': {
                'predicted_trend': 0.0,
                'coupling_strength': 'unknown',
                'thermal_lag_hours': 6.0
            },
            'trend_validation': {
                'outdoor_rising_indoor_should_follow': False,
                'outdoor_falling_indoor_should_follow': False,
                'significant_trend_detected': False,
                'coupling_factor': 0.1
            },
            'data_quality': {
                'historical_points': 0,
                'time_span_hours': 0.0,
                'temperature_reliability': 'low'
            }
        }

    def validate_prediction_against_trends(self, prediction: float, trend_analysis: Dict, 
                                         current_conditions: Dict, time_hours: float) -> Dict:
        """
        Validate a temperature prediction against expected trends and apply corrections
        
        Args:
            prediction: Predicted temperature
            trend_analysis: Results from analyze_temperature_trends
            current_conditions: Current sensor conditions
            time_hours: Prediction time horizon in hours
            
        Returns:
            Dict with validation results and corrected prediction
        """
        try:
            current_indoor = current_conditions.get('indoor_temp', 70.0)
            outdoor_trend = trend_analysis['outdoor_trend']['rate_per_hour']
            expected_response = trend_analysis['expected_indoor_response']['predicted_trend']
            
            # Calculate expected change based on trends
            expected_change = expected_response * time_hours
            expected_temp = current_indoor + expected_change
            
            # Calculate prediction error vs trend expectation
            trend_error = abs(prediction - expected_temp)
            
            # Determine if correction is needed
            needs_correction = False
            correction_reason = ""
            
            # Check for illogical predictions
            if trend_analysis['trend_validation']['significant_trend_detected']:
                if outdoor_trend > 0.5 and prediction < current_indoor - 1.0:
                    needs_correction = True
                    correction_reason = f"Outdoor rising ({outdoor_trend:+.2f}°F/hr) but prediction shows indoor falling"
                elif outdoor_trend < -0.5 and prediction > current_indoor + 1.0:
                    needs_correction = True
                    correction_reason = f"Outdoor falling ({outdoor_trend:+.2f}°F/hr) but prediction shows indoor rising"
            
            # Apply correction if needed
            corrected_prediction = prediction
            if needs_correction:
                # Blend original prediction with trend-based expectation
                blend_factor = 0.3  # 30% trend correction, 70% original model
                corrected_prediction = prediction * (1 - blend_factor) + expected_temp * blend_factor
                logger.warning(f"🔧 Trend correction applied: {prediction:.1f}°F → {corrected_prediction:.1f}°F")
                logger.warning(f"   Reason: {correction_reason}")
            
            validation_result = {
                'original_prediction': prediction,
                'corrected_prediction': corrected_prediction,
                'trend_expected_temp': expected_temp,
                'trend_error': trend_error,
                'correction_applied': needs_correction,
                'correction_reason': correction_reason,
                'confidence_score': self._calculate_prediction_confidence(trend_analysis, trend_error)
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in prediction validation: {e}", exc_info=True)
            return {
                'original_prediction': prediction,
                'corrected_prediction': prediction,
                'trend_expected_temp': prediction,
                'trend_error': 0.0,
                'correction_applied': False,
                'correction_reason': "Validation error",
                'confidence_score': 0.5
            }

    def _calculate_prediction_confidence(self, trend_analysis: Dict, trend_error: float) -> float:
        """Calculate confidence score for prediction based on trend alignment"""
        base_confidence = 0.7
        
        # Reduce confidence for high trend errors
        error_penalty = min(0.3, trend_error * 0.1)
        
        # Increase confidence for good data quality
        data_quality = trend_analysis['data_quality']['temperature_reliability']
        quality_bonus = {'high': 0.2, 'medium': 0.1, 'low': 0.0}.get(data_quality, 0.0)
        
        # Adjust for trend strength
        if trend_analysis['trend_validation']['significant_trend_detected']:
            trend_bonus = 0.1
        else:
            trend_bonus = 0.0
            
        confidence = base_confidence - error_penalty + quality_bonus + trend_bonus
        return max(0.1, min(1.0, confidence))
        
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
            base_prediction = dT_dt
            correction = self.ml_corrector.predict_correction({
                'indoor_temp': t_in,
                'outdoor_temp': t_out,
                'indoor_humidity': h_in,
                'outdoor_humidity': h_out,
                'hvac_state': hvac_state,
                'solar_irradiance': solar,
                'base_prediction': dT_dt
            })
            
            if correction != 0.0:
                logger.info(f"🤖 ML Correction Applied:")
                logger.info(f"   📊 Base RC prediction: {base_prediction:.4f}°F/h")
                logger.info(f"   🔮 ML correction: {correction:.4f}°F/h")
                logger.info(f"   📈 Final prediction: {base_prediction + correction:.4f}°F/h")
                logger.info(f"   🏠 Context: T_in={t_in:.1f}°F, T_out={t_out:.1f}°F, HVAC={hvac_state}")
            
            dT_dt += correction
        
        # Apply reasonable bounds to temperature change rate
        # Maximum realistic HVAC heating/cooling: ±15°F/hour
        # Maximum realistic natural drift: ±5°F/hour
        max_rate = 20.0  # °F/hour absolute maximum
        dT_dt = max(-max_rate, min(max_rate, dT_dt))
        
        # Log extreme values for debugging
        if abs(dT_dt) > 10.0:
            logger.debug(f"High temperature change rate: {dT_dt:.2f}°F/hr (T_in: {t_in:.1f}°F, T_out: {t_out:.1f}°F, HVAC: {hvac_state})")
            
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
    
    def reset_model(self):
        """Reset thermal model to initial state"""
        logger.info("Resetting thermal model...")
        
        # Clear history
        self.parameter_history = []
        
        # Reset state variables
        self.last_temp_in = None
        self.last_update = None
        self.current_indoor_temp = None
        self.current_outdoor_temp = None
        self.current_hvac_state = 'unknown'
        
        # Reinitialize RLS with default parameters
        self.rls = RecursiveLeastSquares(n_params=6, forgetting_factor=0.99)
        self._set_default_parameters()
        
        # Reset ML corrector if available
        if self.ml_corrector:
            self.ml_corrector.reset_model()
            
        logger.info("Thermal model reset complete")