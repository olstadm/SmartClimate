"""
Forecast Engine for simulating temperature trajectories
Projects indoor temperature 12 hours into the future
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import copy

logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Simulates future indoor temperature trajectories
    Produces both idle (no HVAC) and controlled (smart HVAC) scenarios
    """
    
    def __init__(self, config, thermal_model, data_store):
        self.config = config
        self.thermal_model = thermal_model
        self.data_store = data_store
        
        # Simulation parameters
        self.time_step_minutes = 5  # 5-minute simulation steps
        self.horizon_hours = config.get('forecast_horizon_hours', 12)
        self.horizon_steps = int(self.horizon_hours * 60 / self.time_step_minutes)
        
        # Control parameters
        self.comfort_min = config.get('comfort_min_temp', 20.0)
        self.comfort_max = config.get('comfort_max_temp', 24.0)
        self.control_deadband = 0.5  # °C hysteresis
        
    async def generate_forecast(self, current_data: Dict, weather_forecast: Dict) -> Dict:
        """
        Generate temperature forecast for next 12 hours
        
        Args:
            current_data: Current sensor readings
            weather_forecast: Weather forecast data from AccuWeather
            
        Returns:
            Forecast results with multiple trajectories
        """
        try:
            logger.info("=== Starting Forecast Generation ===")
            logger.info(f"Current data keys: {list(current_data.keys())}")
            logger.info(f"Current indoor temp: {current_data.get('indoor_temp')}°F")
            logger.info(f"Current outdoor temp: {current_data.get('outdoor_temp')}°F")
            logger.info(f"Weather forecast keys: {list(weather_forecast.keys())}")
            
            if 'current_outdoor' in weather_forecast:
                current_outdoor = weather_forecast['current_outdoor']
                logger.info(f"AccuWeather current: {current_outdoor.get('temperature')}°F, {current_outdoor.get('humidity')}%")
            else:
                logger.warning("No current_outdoor data from AccuWeather")
                
            if 'hourly_forecast' in weather_forecast:
                hourly_count = len(weather_forecast['hourly_forecast'])
                logger.info(f"AccuWeather hourly forecast points: {hourly_count}")
                if hourly_count > 0:
                    first_hour = weather_forecast['hourly_forecast'][0]
                    logger.info(f"First forecast hour: {first_hour.get('temperature')}°F at {first_hour.get('timestamp')}")
            else:
                logger.warning("No hourly_forecast data from AccuWeather")
            
            # Prepare outdoor conditions series
            logger.info("Preparing outdoor conditions series...")
            outdoor_series = self._prepare_outdoor_series(current_data, weather_forecast)
            logger.info(f"Generated outdoor series with {len(outdoor_series)} points")
            if outdoor_series:
                logger.info(f"First outdoor point: {outdoor_series[0]['outdoor_temp']}°F at {outdoor_series[0]['timestamp']}")
                logger.info(f"Last outdoor point: {outdoor_series[-1]['outdoor_temp']}°F at {outdoor_series[-1]['timestamp']}")
            
            # Initialize state
            initial_state = {
                'indoor_temp': current_data['indoor_temp'],
                'indoor_humidity': current_data['indoor_humidity'],
                'hvac_state': current_data.get('hvac_state', 'off'),
                'timestamp': current_data['timestamp']
            }
            
            # Generate idle trajectory (no HVAC control)
            idle_trajectory = await self._simulate_trajectory(
                initial_state, outdoor_series, control_mode='idle'
            )
            
            # Generate controlled trajectory (smart HVAC control)
            controlled_trajectory = await self._simulate_trajectory(
                initial_state, outdoor_series, control_mode='smart'
            )
            
            # Generate current HVAC trajectory (maintain current state)
            current_trajectory = await self._simulate_trajectory(
                initial_state, outdoor_series, control_mode='current',
                fixed_hvac_state=current_data.get('hvac_state', 'off')
            )
            
            # Extract key results
            timestamps = [step['timestamp'] for step in idle_trajectory]
            
            result = {
                'timestamp': datetime.now(),
                'initial_conditions': initial_state,
                'outdoor_forecast': [step['outdoor_temp'] for step in outdoor_series],
                'indoor_forecast': [step['indoor_temp'] for step in controlled_trajectory],
                'idle_trajectory': idle_trajectory,
                'controlled_trajectory': controlled_trajectory,
                'current_trajectory': current_trajectory,
                'timestamps': timestamps,
                'hvac_schedule': self._extract_hvac_schedule(controlled_trajectory),
                'forecast_confidence': self._calculate_confidence(outdoor_series)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}", exc_info=True)
            raise
            
    def _prepare_outdoor_series(self, current_data: Dict, weather_forecast: Dict) -> List[Dict]:
        """Prepare outdoor conditions time series"""
        logger.info("=== Preparing Outdoor Conditions Series ===")
        series = []
        
        # Use current outdoor temp if available, otherwise use AccuWeather current
        current_outdoor_temp = current_data.get('outdoor_temp')
        logger.info(f"Local outdoor temp from sensors: {current_outdoor_temp}°F")
        
        if current_outdoor_temp is None:
            accuweather_temp = weather_forecast.get('current_outdoor', {}).get('temperature')
            current_outdoor_temp = accuweather_temp if accuweather_temp is not None else 70.0  # Fahrenheit fallback
            logger.info(f"Using AccuWeather current temp: {current_outdoor_temp}°F")
        else:
            logger.info(f"Using local sensor outdoor temp: {current_outdoor_temp}°F")
            
        current_outdoor_humidity = current_data.get('outdoor_humidity')
        logger.info(f"Local outdoor humidity from sensors: {current_outdoor_humidity}%")
        
        if current_outdoor_humidity is None:
            accuweather_humidity = weather_forecast.get('current_outdoor', {}).get('humidity')
            current_outdoor_humidity = accuweather_humidity if accuweather_humidity is not None else 50.0
            logger.info(f"Using AccuWeather current humidity: {current_outdoor_humidity}%")
        else:
            logger.info(f"Using local sensor outdoor humidity: {current_outdoor_humidity}%")
            
        # Start with current conditions
        current_time = datetime.now()
        
        # Build series from forecast
        forecast_data = weather_forecast.get('hourly_forecast', [])
        logger.info(f"Processing {len(forecast_data)} hourly forecast points for {self.horizon_steps} simulation steps")
        
        forecast_hours_used = []
        
        for i in range(self.horizon_steps):
            timestamp = current_time + timedelta(minutes=i * self.time_step_minutes)
            
            # For the first few steps (first 30 minutes), use current conditions
            # This ensures we start with accurate current outdoor temp
            if i < 6:  # First 30 minutes (6 steps * 5 minutes each)
                forecast_temp = current_outdoor_temp
                forecast_humidity = current_outdoor_humidity
                solar_irradiance = 0
                if i == 0:
                    logger.info(f"Step {i}: Using current outdoor conditions: {forecast_temp}°F, {forecast_humidity}%")
            else:
                # After 30 minutes, use AccuWeather forecast data
                forecast_temp = current_outdoor_temp  # Default fallback
                forecast_humidity = current_outdoor_humidity
                solar_irradiance = 0
                matched_hour = None
                
                for forecast_hour in forecast_data:
                    forecast_timestamp = forecast_hour.get('timestamp')
                    if hasattr(forecast_timestamp, 'hour'):
                        if forecast_timestamp.hour == timestamp.hour:
                            forecast_temp = forecast_hour.get('temperature', forecast_temp)
                            forecast_humidity = forecast_hour.get('humidity', forecast_humidity)
                            solar_irradiance = forecast_hour.get('solar_irradiance', 0)
                            matched_hour = forecast_timestamp.hour
                            if matched_hour not in forecast_hours_used:
                                forecast_hours_used.append(matched_hour)
                                logger.info(f"Hour {matched_hour}: Using AccuWeather forecast temp {forecast_temp}°F, humidity {forecast_humidity}%")
                            break
                        
            # Interpolate between hours if needed (but only after initial current conditions period)
            if i > 6 and len(series) > 0:
                # Simple linear interpolation for forecast data
                minutes_into_hour = timestamp.minute
                if minutes_into_hour > 0:
                    weight = minutes_into_hour / 60.0
                    forecast_temp = (1 - weight) * series[-1]['outdoor_temp'] + weight * forecast_temp
                    forecast_humidity = (1 - weight) * series[-1]['outdoor_humidity'] + weight * forecast_humidity
                    
            series.append({
                'timestamp': timestamp,
                'outdoor_temp': forecast_temp,
                'outdoor_humidity': forecast_humidity,
                'solar_irradiance': solar_irradiance
            })
            
        return series
        
    async def _simulate_trajectory(self, initial_state: Dict, outdoor_series: List[Dict],
                                 control_mode: str = 'idle',
                                 fixed_hvac_state: Optional[str] = None) -> List[Dict]:
        """
        Simulate temperature trajectory
        
        Args:
            initial_state: Starting conditions
            outdoor_series: Outdoor conditions over time
            control_mode: 'idle', 'smart', or 'current'
            fixed_hvac_state: Fixed HVAC state for 'current' mode
            
        Returns:
            List of states over time
        """
        trajectory = []
        
        # Current state
        state = {
            'indoor_temp': initial_state['indoor_temp'],
            'indoor_humidity': initial_state['indoor_humidity'],
            'hvac_state': initial_state['hvac_state']
        }
        
        # Simulate each time step
        for i, outdoor in enumerate(outdoor_series):
            # Determine HVAC state based on control mode
            if control_mode == 'idle':
                state['hvac_state'] = 'off'
            elif control_mode == 'current':
                state['hvac_state'] = fixed_hvac_state or 'off'
            elif control_mode == 'smart':
                state['hvac_state'] = self._smart_hvac_control(
                    state['indoor_temp'],
                    state['hvac_state'],
                    outdoor['outdoor_temp']
                )
                
            # Calculate temperature change rate
            dT_dt = self.thermal_model.predict_temperature_change(
                state['indoor_temp'],
                outdoor['outdoor_temp'],
                state['indoor_humidity'],
                outdoor['outdoor_humidity'],
                state['hvac_state'],
                outdoor['solar_irradiance']
            )
            
            # Update temperature
            dt_hours = self.time_step_minutes / 60.0
            new_temp = state['indoor_temp'] + dT_dt * dt_hours
            
            # Update humidity (simplified - tends toward outdoor)
            humidity_rate = 0.1  # 10% approach per hour
            new_humidity = state['indoor_humidity'] + (
                outdoor['outdoor_humidity'] - state['indoor_humidity']
            ) * humidity_rate * dt_hours
            
            # Update state
            state['indoor_temp'] = new_temp
            state['indoor_humidity'] = new_humidity
            
            # Record trajectory point
            trajectory.append({
                'timestamp': outdoor['timestamp'],
                'indoor_temp': new_temp,
                'indoor_humidity': new_humidity,
                'outdoor_temp': outdoor['outdoor_temp'],
                'outdoor_humidity': outdoor['outdoor_humidity'],
                'hvac_state': state['hvac_state'],
                'solar_irradiance': outdoor['solar_irradiance'],
                'temp_change_rate': dT_dt
            })
            
        return trajectory
        
    def _smart_hvac_control(self, current_temp: float, current_hvac: str,
                          outdoor_temp: float) -> str:
        """
        Determine HVAC state using smart control logic
        
        Args:
            current_temp: Current indoor temperature
            current_hvac: Current HVAC state
            outdoor_temp: Current outdoor temperature
            
        Returns:
            New HVAC state: 'heat', 'cool', or 'off'
        """
        # Hysteresis thresholds
        heat_on = self.comfort_min - self.control_deadband
        heat_off = self.comfort_min + self.control_deadband
        cool_on = self.comfort_max + self.control_deadband
        cool_off = self.comfort_max - self.control_deadband
        
        # Current state logic with hysteresis
        if current_hvac == 'heat':
            # Heating is on - turn off when warm enough
            if current_temp >= heat_off:
                return 'off'
            else:
                return 'heat'
                
        elif current_hvac == 'cool':
            # Cooling is on - turn off when cool enough
            if current_temp <= cool_off:
                return 'off'
            else:
                return 'cool'
                
        else:  # HVAC is off
            # Check if we need heating
            if current_temp < heat_on:
                return 'heat'
            # Check if we need cooling
            elif current_temp > cool_on:
                return 'cool'
            else:
                # Stay off - but consider predictive control
                # If outdoor temp suggests we'll need action soon, prepare
                if outdoor_temp < current_temp - 5 and current_temp < self.comfort_min + 1:
                    # Cold outside, might need heat soon
                    return 'heat' if current_temp < self.comfort_min + 0.5 else 'off'
                elif outdoor_temp > current_temp + 5 and current_temp > self.comfort_max - 1:
                    # Hot outside, might need cooling soon
                    return 'cool' if current_temp > self.comfort_max - 0.5 else 'off'
                    
                return 'off'
                
    def _extract_hvac_schedule(self, trajectory: List[Dict]) -> List[Dict]:
        """Extract HVAC on/off schedule from trajectory"""
        schedule = []
        
        current_state = None
        start_time = None
        
        for point in trajectory:
            hvac_state = point['hvac_state']
            
            if hvac_state != current_state:
                # State change
                if current_state and current_state != 'off':
                    # End of HVAC operation
                    schedule.append({
                        'mode': current_state,
                        'start': start_time,
                        'end': point['timestamp'],
                        'duration_minutes': (point['timestamp'] - start_time).total_seconds() / 60
                    })
                    
                if hvac_state != 'off':
                    # Start of HVAC operation
                    start_time = point['timestamp']
                    
                current_state = hvac_state
                
        # Handle last segment
        if current_state and current_state != 'off' and start_time:
            schedule.append({
                'mode': current_state,
                'start': start_time,
                'end': trajectory[-1]['timestamp'],
                'duration_minutes': (trajectory[-1]['timestamp'] - start_time).total_seconds() / 60
            })
            
        return schedule
        
    def _calculate_confidence(self, outdoor_series: List[Dict]) -> float:
        """
        Calculate forecast confidence based on various factors
        
        Returns:
            Confidence score 0-1
        """
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence for longer horizons
        horizon_factor = 1.0 - (self.horizon_hours / 24.0) * 0.2
        confidence *= horizon_factor
        
        # Reduce confidence if weather data is sparse
        if len(outdoor_series) < self.horizon_steps:
            data_factor = len(outdoor_series) / self.horizon_steps
            confidence *= data_factor
            
        # Model convergence factor
        model_metrics = self.thermal_model.get_model_quality_metrics()
        if model_metrics.get('parameter_convergence'):
            confidence *= 1.1  # Boost if converged
            
        # Clip to valid range
        return min(max(confidence, 0.1), 1.0)
        
    def analyze_trajectories(self, forecast_result: Dict) -> Dict:
        """
        Analyze forecast trajectories for insights
        
        Args:
            forecast_result: Result from generate_forecast
            
        Returns:
            Analysis with energy usage, comfort violations, etc.
        """
        idle = forecast_result['idle_trajectory']
        controlled = forecast_result['controlled_trajectory']
        current = forecast_result['current_trajectory']
        
        analysis = {
            'energy_savings': self._calculate_energy_savings(controlled, current),
            'comfort_violations': self._find_comfort_violations(idle, controlled, current),
            'hvac_runtime': self._calculate_hvac_runtime(controlled),
            'peak_temperatures': self._find_peak_temperatures(idle, controlled, current),
            'recommended_actions': self._generate_recommendations(forecast_result)
        }
        
        return analysis
        
    def _calculate_energy_savings(self, controlled: List[Dict], current: List[Dict]) -> Dict:
        """Calculate potential energy savings"""
        controlled_runtime = sum(1 for p in controlled if p['hvac_state'] != 'off')
        current_runtime = sum(1 for p in current if p['hvac_state'] != 'off')
        
        runtime_reduction = (current_runtime - controlled_runtime) / max(current_runtime, 1)
        
        return {
            'runtime_reduction_percent': runtime_reduction * 100,
            'controlled_runtime_minutes': controlled_runtime * self.time_step_minutes,
            'current_runtime_minutes': current_runtime * self.time_step_minutes
        }
        
    def _find_comfort_violations(self, idle: List[Dict], controlled: List[Dict], 
                               current: List[Dict]) -> Dict:
        """Find when temperature goes outside comfort range"""
        violations = {
            'idle': [],
            'controlled': [],
            'current': []
        }
        
        for name, trajectory in [('idle', idle), ('controlled', controlled), ('current', current)]:
            for point in trajectory:
                temp = point['indoor_temp']
                if temp < self.comfort_min or temp > self.comfort_max:
                    violations[name].append({
                        'timestamp': point['timestamp'],
                        'temperature': temp,
                        'violation': 'too_cold' if temp < self.comfort_min else 'too_hot'
                    })
                    
        return violations
        
    def _calculate_hvac_runtime(self, trajectory: List[Dict]) -> Dict:
        """Calculate HVAC runtime statistics"""
        heating_steps = sum(1 for p in trajectory if p['hvac_state'] == 'heat')
        cooling_steps = sum(1 for p in trajectory if p['hvac_state'] == 'cool')
        
        return {
            'heating_minutes': heating_steps * self.time_step_minutes,
            'cooling_minutes': cooling_steps * self.time_step_minutes,
            'total_runtime_minutes': (heating_steps + cooling_steps) * self.time_step_minutes,
            'runtime_percent': (heating_steps + cooling_steps) / len(trajectory) * 100
        }
        
    def _find_peak_temperatures(self, idle: List[Dict], controlled: List[Dict],
                              current: List[Dict]) -> Dict:
        """Find minimum and maximum temperatures"""
        return {
            'idle': {
                'min': min(p['indoor_temp'] for p in idle),
                'max': max(p['indoor_temp'] for p in idle)
            },
            'controlled': {
                'min': min(p['indoor_temp'] for p in controlled),
                'max': max(p['indoor_temp'] for p in controlled)
            },
            'current': {
                'min': min(p['indoor_temp'] for p in current),
                'max': max(p['indoor_temp'] for p in current)
            }
        }
        
    def _generate_recommendations(self, forecast_result: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        analysis = self.analyze_trajectories(forecast_result)
        
        # Energy savings recommendation
        if analysis['energy_savings']['runtime_reduction_percent'] > 10:
            recommendations.append(
                f"Smart control can reduce HVAC runtime by "
                f"{analysis['energy_savings']['runtime_reduction_percent']:.0f}%"
            )
            
        # Comfort violation warnings
        if analysis['comfort_violations']['idle']:
            first_violation = analysis['comfort_violations']['idle'][0]
            time_until = (first_violation['timestamp'] - datetime.now()).total_seconds() / 60
            recommendations.append(
                f"Without HVAC, temperature will be {first_violation['violation'].replace('_', ' ')} "
                f"in {time_until:.0f} minutes"
            )
            
        return recommendations