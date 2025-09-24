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
        self.base_lambda = forgetting_factor  # Store original value
        self.update_count = 0  # Track number of updates for adaptive learning
        
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
        self.update_count += 1
        
        # Adaptive forgetting factor for aggressive early learning
        if self.update_count < 50:
            # Early learning stage: aggressive adaptation
            self.lambda_ff = 0.92  # More aggressive learning
        elif self.update_count < 100:
            # Intermediate stage: moderate adaptation
            self.lambda_ff = 0.96
        else:
            # Mature stage: conservative adaptation
            self.lambda_ff = self.base_lambda
        
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
        
        # Apply retroactive physics correction to clean any existing HVAC contamination
        await self.retroactive_physics_correction()
        
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
        
    def _build_clean_natural_vector(self, t_in: float, t_out: float, 
                                   h_in: float, h_out: float, solar: float) -> np.ndarray:
        """Build feature vector for natural (no-HVAC) thermal predictions"""
        # Calculate enthalpy difference (simplified)
        h_diff = self._calculate_enthalpy_diff(t_out, h_out, t_in, h_in)
        
        # Clean natural physics vector: [T_out - T_in, 0, 0, 1, h_diff, solar]
        # Zero out HVAC coefficients for pure natural thermal response
        phi = np.array([
            t_out - t_in,    # Temperature difference (primary driver)
            0.0,             # NO heating influence
            0.0,             # NO cooling influence  
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
            
            # Predict expected indoor response based on TEMPERATURE DIFFERENTIAL, not outdoor trend
            current_outdoor = current_conditions.get('outdoor_temp', 70.0)
            current_indoor = current_conditions.get('indoor_temp', 70.0)
            
            # Use thermal model parameters to predict indoor response
            thermal_coupling = abs(self.rls.theta[0]) if len(self.rls.theta) > 0 else 0.1  # 'a' parameter
            
            # CRITICAL FIX: Indoor trend should be based on temperature differential
            temp_differential = current_outdoor - current_indoor
            
            # Indoor temperature moves towards outdoor temperature (basic thermal physics)
            if temp_differential > 0:
                # Outdoor warmer: indoor should warm up (positive trend)
                expected_indoor_trend = thermal_coupling * min(abs(temp_differential), 5.0)  # Max 5°F/hr natural warming
            elif temp_differential < 0:
                # Outdoor cooler: indoor should cool down (negative trend)  
                expected_indoor_trend = -thermal_coupling * min(abs(temp_differential), 5.0)  # Max 5°F/hr natural cooling
            else:
                # Temperatures equal: minimal change
                expected_indoor_trend = 0.0
            
            logger.info(f"🌡️ Physics-based trend: T_out={current_outdoor:.1f}°F, T_in={current_indoor:.1f}°F, " +
                       f"diff={temp_differential:.1f}°F → expected_indoor_trend={expected_indoor_trend:+.3f}°F/hr")
            
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
                    'outdoor_warmer_indoor_should_warm': temp_differential > 1.0,
                    'outdoor_cooler_indoor_should_cool': temp_differential < -1.0,
                    'significant_temperature_differential': abs(temp_differential) > 2.0,
                    'significant_trend_detected': abs(temp_differential) > 2.0 and abs(outdoor_trend) > 0.5,
                    'coupling_factor': thermal_coupling,
                    'temperature_differential': round(temp_differential, 1)
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
            outdoor_trend = trend_analysis.get('outdoor_trend', {}).get('rate_per_hour', 0.0)
            expected_response = trend_analysis.get('expected_indoor_response', {}).get('predicted_trend', 0.0)
            
            # Calculate expected change based on trends
            expected_change = expected_response * time_hours
            expected_temp = current_indoor + expected_change
            
            # Calculate prediction error vs trend expectation
            trend_error = abs(prediction - expected_temp)
            
            # Determine if correction is needed
            needs_correction = False
            correction_reason = ""
            
            # Check for physics violations based on temperature differential (not outdoor trend)
            current_outdoor = current_conditions.get('outdoor_temp', 70.0)
            temp_differential = current_outdoor - current_indoor
            
            # Physics check: Indoor temp should move towards outdoor temp
            if time_hours > 0.5:  # Only check for predictions > 30 minutes out
                if temp_differential > 2.0 and prediction < current_indoor:
                    needs_correction = True
                    correction_reason = f"Outdoor warmer by {temp_differential:.1f}°F but prediction shows cooling"
                elif temp_differential < -2.0 and prediction > current_indoor:
                    needs_correction = True  
                    correction_reason = f"Outdoor cooler by {temp_differential:.1f}°F but prediction shows warming"
            
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
        data_quality = trend_analysis.get('data_quality', {}).get('temperature_reliability', 'low')
        quality_bonus = {'high': 0.2, 'medium': 0.1, 'low': 0.0}.get(data_quality, 0.0)
        
        # Adjust for trend strength
        try:
            if trend_analysis.get('trend_validation', {}).get('significant_trend_detected', False):
                trend_bonus = 0.1
            else:
                trend_bonus = 0.0
        except (KeyError, TypeError):
            logger.debug("Missing trend validation data, using default trend bonus")
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
        # Build clean feature vector for no-HVAC predictions
        if hvac_state in ['off', 'idle']:
            # For idle predictions, use ONLY natural thermal physics - zero out HVAC parameters
            phi = self._build_clean_natural_vector(t_in, t_out, h_in, h_out, solar)
            # Use only natural parameters: [temp_diff, 0, 0, baseline, enthalpy, solar]
            clean_theta = self.rls.theta.copy()
            clean_theta[1] = 0.0  # Zero heating coefficient
            clean_theta[2] = 0.0  # Zero cooling coefficient  
            dT_dt = np.dot(phi, clean_theta)
        else:
            # Normal prediction with all parameters for HVAC scenarios
            phi = self._build_feature_vector(t_in, t_out, h_in, h_out, hvac_state, solar)
            dT_dt = np.dot(phi, self.rls.theta)
        
        # Apply ML correction if available (DISABLED in v2.0 for physics accuracy)
        if self.ml_corrector and self.ml_enabled and False:  # Temporarily disabled
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
        
        # Enhanced physics-based constraints for temperature change rate
        if hvac_state in ['off', 'idle']:
            # STRICT natural physics constraints for idle/off mode
            temp_diff = t_out - t_in
            
            # Calculate maximum natural drift rate based on building physics
            # Use thermal time constant if available from enhanced training
            thermal_time_constant = getattr(self, 'thermal_time_constant_hours', 8.0)  # Default 8 hours
            max_natural_approach_rate = abs(temp_diff) / thermal_time_constant  # Approach at 1/τ rate
            
            # Physics Rule 1: Direction must follow temperature differential
            if temp_diff > 0.1:
                # Outdoor warmer: indoor MUST warm up (positive dT_dt only)
                if dT_dt < 0:
                    logger.warning(f"🔬 Physics violation corrected: Predicted cooling ({dT_dt:.3f}°F/hr) when outdoor warmer ({temp_diff:.1f}°F)")
                    dT_dt = 0.1  # Minimum warming rate
                else:
                    # Limit warming to natural approach rate
                    max_rate = min(max_natural_approach_rate, 2.0)  # Max 2°F/hr natural warming
                    dT_dt = min(dT_dt, max_rate)
                    
            elif temp_diff < -0.1:
                # Outdoor cooler: indoor MUST cool down (negative dT_dt only)
                if dT_dt > 0:
                    logger.warning(f"🔬 Physics violation corrected: Predicted warming ({dT_dt:.3f}°F/hr) when outdoor cooler ({temp_diff:.1f}°F)")
                    dT_dt = -0.1  # Minimum cooling rate
                else:
                    # Limit cooling to natural approach rate
                    max_rate = min(max_natural_approach_rate, 2.0)  # Max 2°F/hr natural cooling
                    dT_dt = max(dT_dt, -max_rate)
                    
            else:
                # Temperatures very close: minimal drift allowed
                max_equilibrium_drift = 0.2  # Very slow drift near equilibrium
                dT_dt = max(-max_equilibrium_drift, min(max_equilibrium_drift, dT_dt))
                
            # Absolute bounds for natural drift
            absolute_natural_max = 3.0  # °F/hour absolute maximum for natural drift
            dT_dt = max(-absolute_natural_max, min(absolute_natural_max, dT_dt))
                
            # Log physics corrections for uncontrolled scenarios
            if abs(dT_dt) > 0.5 or len(self.parameter_history) < 50:  # Log significant changes or early learning
                logger.info(f"🔬 Natural physics (no HVAC): T_in={t_in:.1f}°F, T_out={t_out:.1f}°F, " +
                          f"dT_dt={dT_dt:.3f}°F/hr (diff={temp_diff:.1f}°F, τ={thermal_time_constant:.1f}h)")
        else:
            # HVAC active: Allow stronger heating/cooling but still bounded
            temp_diff = t_out - t_in
            
            # Enhanced HVAC bounds based on system capabilities
            if hvac_state == 'heat':
                # Heating should always warm, but limited by capacity
                max_heating_rate = getattr(self, 'max_heating_rate', 8.0)  # °F/hour
                if dT_dt < 0:
                    logger.warning(f"🔬 HVAC Physics violation: Heating predicted cooling ({dT_dt:.3f}°F/hr)")
                    dT_dt = 0.5  # Minimum heating effectiveness
                dT_dt = min(dT_dt, max_heating_rate)
                
            elif hvac_state == 'cool':
                # Cooling should always cool, but limited by capacity
                max_cooling_rate = getattr(self, 'max_cooling_rate', -8.0)  # °F/hour (negative)
                if dT_dt > 0:
                    logger.warning(f"🔬 HVAC Physics violation: Cooling predicted warming ({dT_dt:.3f}°F/hr)")
                    dT_dt = -0.5  # Minimum cooling effectiveness
                dT_dt = max(dT_dt, max_cooling_rate)
            
            # Overall HVAC bounds
            max_hvac_rate = 12.0  # °F/hour for active HVAC
            dT_dt = max(-max_hvac_rate, min(max_hvac_rate, dT_dt))
        
        # Absolute maximum bounds for all scenarios
        absolute_max = 15.0  # °F/hour absolute maximum
        dT_dt = max(-absolute_max, min(absolute_max, dT_dt))
        
        # Log extreme values for debugging
        if abs(dT_dt) > 5.0:
            logger.debug(f"High temperature change rate: {dT_dt:.2f}°F/hr (T_in: {t_in:.1f}°F, T_out: {t_out:.1f}°F, HVAC: {hvac_state})")
            
        return dT_dt
        
    def get_parameters(self) -> Dict:
        """Get current model parameters in interpretable form"""
        theta = self.rls.theta
        
        # Extract and interpret parameters
        a = theta[0]  # 1/τ
        
        # Constrain thermal time constant to realistic range (12-16 hours)
        if abs(a) > 0.001:
            raw_tau = 1 / abs(a)
            # Clamp to reasonable range while allowing some adjustment
            tau = max(12.0, min(16.0, raw_tau))
            # Update the parameter if it was constrained
            if tau != raw_tau:
                theta[0] = 1 / tau if a > 0 else -1 / tau
                self.rls.theta[0] = theta[0]
                logger.debug(f"Constrained thermal time constant from {raw_tau:.1f}h to {tau:.1f}h")
        else:
            tau = 14.0  # Default to middle of range
        
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
    
    async def apply_enhanced_training_results(self, training_results: Dict):
        """
        Apply physics-validated parameters from enhanced training to the thermal model
        
        Args:
            training_results: Results from enhanced training system containing validated parameters
        """
        try:
            if not training_results or training_results.get('accuracy_score', 0) < 0.85:
                logger.warning(f"Enhanced training results quality too low (accuracy: {training_results.get('accuracy_score', 0):.3f}), not applying")
                return
                
            logger.info("🧠 Applying enhanced training results to thermal model...")
            logger.info(f"   Training accuracy: {training_results.get('accuracy_score', 0):.1%}")
            logger.info(f"   Physics compliance: {training_results.get('physics_compliance', 0):.1%}")
            logger.info(f"   Valid samples: {training_results.get('valid_samples', 0)}/{training_results.get('total_samples', 0)}")
            
            # Get current parameters for comparison
            old_params = self.get_parameters()
            
            # Apply enhanced training parameters from model_parameters structure
            if 'model_parameters' in training_results:
                model_params = training_results['model_parameters']
                
                # The enhanced training system provides theta array with validated physics parameters
                if 'theta' in model_params:
                    validated_theta = np.array(model_params['theta'])
                    if len(validated_theta) == len(self.rls.theta):
                        logger.info(f"📊 Applying validated theta parameters: {validated_theta}")
                        self.rls.theta = validated_theta
                        
                        # Also update thermal_model.theta if it exists (from building model integration)
                        if hasattr(self, 'theta'):
                            self.theta = validated_theta[0]  # The 'a' parameter (1/τ)
                    
            # Apply building-specific parameters if available
            if 'building_type' in training_results:
                logger.info(f"📋 Training was done for building type: {training_results['building_type']}")
                
            if 'training_location' in training_results:
                logger.info(f"🌤️ Training was done for location: {training_results['training_location']}")
                
            # Reset covariance matrix to allow for some continued adaptation but with more confidence
            # in the physics-validated parameters
            self.rls.P = np.eye(self.rls.n_params) * 100  # Lower uncertainty than initial (1000 -> 100)
            
            # Log the parameter changes
            new_params = self.get_parameters()
            logger.info("📊 Enhanced training parameter updates:")
            logger.info(f"   Time constant: {old_params['time_constant']:.1f}h → {new_params['time_constant']:.1f}h")
            logger.info(f"   Heating rate: {old_params['heating_rate']:.1f}°F/h → {new_params['heating_rate']:.1f}°F/h")  
            logger.info(f"   Cooling rate: {old_params['cooling_rate']:.1f}°F/h → {new_params['cooling_rate']:.1f}°F/h")
            logger.info(f"   Baseline drift: {old_params['baseline_drift']:.3f}°F/h → {new_params['baseline_drift']:.3f}°F/h")
            
            # Save the enhanced parameters
            await self.save_parameters()
            
            # Store enhanced training characteristics for better physics
            if 'building_characteristics' in training_results:
                chars = training_results['building_characteristics']
                self.thermal_time_constant_hours = chars.get('thermal_time_constant_hours', 8.0)
                self.max_heating_rate = chars.get('suggested_heating_rate_F_per_hr', 8.0)
                self.max_cooling_rate = -chars.get('suggested_cooling_rate_F_per_hr', 8.0)
                
                logger.info(f"   Applied thermal characteristics:")
                logger.info(f"     Thermal time constant: {self.thermal_time_constant_hours:.1f} hours")
                logger.info(f"     Max heating rate: {self.max_heating_rate:.1f} °F/hr") 
                logger.info(f"     Max cooling rate: {self.max_cooling_rate:.1f} °F/hr")
            
            logger.info("✅ Enhanced training results successfully applied to thermal model")
            logger.info("🔬 Natural physics predictions will now use strict thermal constraints")
            
        except Exception as e:
            logger.error(f"❌ Error applying enhanced training results: {e}", exc_info=True)
            
    async def retroactive_physics_correction(self):
        """
        Apply retroactive physics corrections to clean existing parameter contamination
        
        This method retrains the thermal model using only natural (no-HVAC) data points
        from historical data to eliminate HVAC influence on idle predictions.
        """
        try:
            logger.info("🔧 Starting retroactive physics correction...")
            logger.info("   Purpose: Remove HVAC contamination from natural thermal parameters")
            
            # Get historical data for retraining
            historical_data = await self.data_store.get_recent_data(hours=168)  # Last week
            if not historical_data or len(historical_data) < 20:
                logger.warning("⚠️ Insufficient historical data for retroactive correction")
                return
                
            logger.info(f"   Available historical data: {len(historical_data)} points")
            
            # Filter for natural thermal data only (HVAC off/idle)
            natural_data = []
            hvac_data = []
            
            for point in historical_data:
                hvac_state = point.get('hvac_state', 'unknown')
                if hvac_state in ['off', 'idle', 'unknown']:
                    natural_data.append(point)
                else:
                    hvac_data.append(point)
                    
            logger.info(f"   Natural thermal points: {len(natural_data)}")
            logger.info(f"   HVAC-active points: {len(hvac_data)} (excluded)")
            
            if len(natural_data) < 10:
                logger.warning("⚠️ Insufficient natural thermal data for correction")
                return
                
            # Create a separate RLS instance for natural parameters only
            natural_rls = RecursiveLeastSquares(n_params=6, forgetting_factor=0.98)
            
            # Train only on natural thermal data with strict physics validation
            valid_natural_updates = 0
            physics_violations = 0
            
            for point in natural_data:
                try:
                    # Extract data
                    t_in = point.get('indoor_temp')
                    t_out = point.get('outdoor_temp') 
                    h_in = point.get('indoor_humidity', 50.0)
                    h_out = point.get('outdoor_humidity', 50.0)
                    solar = point.get('solar_irradiance', 0.0)
                    
                    if None in [t_in, t_out]:
                        continue
                        
                    # Calculate actual temperature change if available
                    next_point = None
                    for p in historical_data:
                        if p.get('timestamp', 0) > point.get('timestamp', 0):
                            next_point = p
                            break
                            
                    if not next_point:
                        continue
                        
                    dt_hours = (next_point['timestamp'] - point['timestamp']) / 3600  # seconds to hours
                    if dt_hours <= 0 or dt_hours > 2:  # Reasonable time window
                        continue
                        
                    actual_temp_change = (next_point['indoor_temp'] - t_in) / dt_hours
                    
                    # Strict physics validation for natural thermal response
                    temp_diff = t_out - t_in
                    
                    # Check if this follows natural thermal physics
                    if abs(temp_diff) > 0.5:  # Significant temperature difference
                        if temp_diff > 0.5 and actual_temp_change < -0.2:
                            # Outdoor warmer but significant cooling - possible HVAC influence
                            physics_violations += 1
                            continue
                        elif temp_diff < -0.5 and actual_temp_change > 0.2:
                            # Outdoor cooler but significant warming - possible HVAC influence
                            physics_violations += 1
                            continue
                            
                    # Validate rate is realistic for natural response
                    max_natural_rate = min(3.0, abs(temp_diff) * 0.6)
                    if abs(actual_temp_change) > max_natural_rate:
                        physics_violations += 1
                        continue
                        
                    # This point passes physics validation - use for natural parameter training
                    phi = self._build_clean_natural_vector(t_in, t_out, h_in, h_out, solar)
                    natural_rls.update(phi, actual_temp_change)
                    valid_natural_updates += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing natural data point: {e}")
                    continue
                    
            logger.info(f"   Physics-valid natural updates: {valid_natural_updates}")
            logger.info(f"   Physics violations filtered: {physics_violations}")
            
            if valid_natural_updates >= 5:
                # Apply the corrected natural parameters to the main model
                # Only update the natural thermal coefficients, preserve HVAC coefficients
                old_theta = self.rls.theta.copy()
                
                # Update natural thermal parameters: [temp_diff, 0, 0, baseline, enthalpy, solar]
                self.rls.theta[0] = natural_rls.theta[0]  # Temperature difference coefficient
                # Keep theta[1] and theta[2] (HVAC coefficients) unchanged
                self.rls.theta[3] = natural_rls.theta[3]  # Baseline drift
                self.rls.theta[4] = natural_rls.theta[4]  # Enthalpy coefficient
                self.rls.theta[5] = natural_rls.theta[5]  # Solar coefficient
                
                logger.info("📊 Retroactive correction applied:")
                logger.info(f"   Natural temp coeff: {old_theta[0]:.4f} → {self.rls.theta[0]:.4f}")
                logger.info(f"   Baseline drift: {old_theta[3]:.4f} → {self.rls.theta[3]:.4f}") 
                logger.info(f"   Enthalpy coeff: {old_theta[4]:.4f} → {self.rls.theta[4]:.4f}")
                logger.info(f"   Solar coeff: {old_theta[5]:.4f} → {self.rls.theta[5]:.4f}")
                logger.info(f"   HVAC coefficients preserved: Heat={self.rls.theta[1]:.4f}, Cool={self.rls.theta[2]:.4f}")
                
                # Save corrected parameters
                await self.save_parameters()
                
                logger.info("✅ Retroactive physics correction completed successfully")
                logger.info("🔬 Natural thermal predictions should now be more physically accurate")
            else:
                logger.warning("⚠️ Insufficient valid natural data for retroactive correction")
                
        except Exception as e:
            logger.error(f"❌ Error in retroactive physics correction: {e}")
            logger.exception("Retroactive correction error:")
            
    def _set_default_parameters(self):
        """Set reasonable default parameters with conservative uncontrolled behavior"""
        # Default values based on typical home characteristics (Fahrenheit)
        # More conservative parameters to prevent unrealistic temperature drops
        self.rls.theta = np.array([
            0.15,   # a: 1/τ (τ = 6.7 hours - faster thermal response)
            3.6,    # k_H: 3.6°F/hour heating rate (2°C/hour * 1.8)
            -4.5,   # k_C: -4.5°F/hour cooling rate (-2.5°C/hour * 1.8)
            0.0,    # b: Zero baseline drift (conservative - no spontaneous cooling)
            0.02,   # k_E: Minimal enthalpy effect (more conservative)
            0.5     # k_S: Moderate solar gain (0.28°C/hour * 1.8)
        ])
        logger.info("✅ Using conservative default thermal model parameters (no spontaneous cooling)")
        
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
        """Check if parameters have converged using adaptive criteria"""
        sample_count = len(self.parameter_history)
        
        # Adaptive convergence criteria based on learning stage
        if sample_count < 20:
            return False  # Need minimum samples
        elif sample_count < 50:
            # Early stage: looser convergence criteria
            min_samples = 20
            convergence_threshold = 0.05  # More lenient
        else:
            # Mature stage: strict convergence criteria
            min_samples = 50
            convergence_threshold = 0.01  # Strict
            
        if sample_count < min_samples:
            return False
            
        # Check variance of parameters over recent updates
        recent_params = np.array([p['parameters'] for p in self.parameter_history[-min_samples:]])
        param_std = np.std(recent_params, axis=0)
        
        # Check if all parameters are stable
        converged = np.all(param_std < convergence_threshold)
        
        # Log convergence status during early learning
        if sample_count < 100 and sample_count % 10 == 0:
            logger.info(f"🎯 Learning Progress: {sample_count} samples, " +
                       f"param_std={param_std.max():.4f}, threshold={convergence_threshold:.4f}, " +
                       f"converged={converged}")
        
        return converged
    
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