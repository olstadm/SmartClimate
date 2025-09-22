"""
Forecast Engine for simulating temperature trajectories
Projects indoor temperature 12 hours into the future
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import copy

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    try:
        import zoneinfo
        HAS_ZONEINFO = True
    except ImportError:
        HAS_ZONEINFO = False

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
        
    async def generate_forecast(self, current_data: Dict, weather_forecast: Dict, timezone_name: str = 'UTC') -> Dict:
        """
        Generate enhanced temperature forecast with historical context and trend analysis
        
        Args:
            current_data: Current sensor readings
            weather_forecast: Weather forecast data from AccuWeather (includes historical_weather if available)
            
        Returns:
            Enhanced forecast results with 18-hour timeline and trend validation
        """
        try:
            logger.info("=== Starting Enhanced Forecast Generation ===")
            logger.info(f"Current data keys: {list(current_data.keys())}")
            logger.info(f"Current indoor temp: {current_data.get('indoor_temp')}°F")
            logger.info(f"Current outdoor temp: {current_data.get('outdoor_temp')}°F")
            logger.info(f"Weather forecast keys: {list(weather_forecast.keys())}")
            
            # Extract and analyze historical data for trend analysis
            historical_weather = weather_forecast.get('historical_weather', [])
            logger.info(f"Historical weather data points: {len(historical_weather)}")
            
            # Perform trend analysis
            trend_analysis = None
            if historical_weather:
                trend_analysis = self.thermal_model.analyze_temperature_trends(
                    historical_weather, current_data
                )
                logger.info(f"✅ Trend analysis complete - Outdoor trend: {trend_analysis['outdoor_trend']['rate_per_hour']:+.2f}°F/hr")
            else:
                logger.warning("⚠️ No historical data available - trend analysis limited")
            
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
            
            # Prepare extended outdoor series (historical + current + forecast)
            logger.info("Preparing extended outdoor conditions series...")
            outdoor_series = self._prepare_extended_outdoor_series(
                current_data, weather_forecast, timezone_name, historical_weather
            )
            logger.info(f"Generated extended outdoor series with {len(outdoor_series)} points")
            if outdoor_series:
                logger.info(f"First outdoor point: {outdoor_series[0]['outdoor_temp']}°F at {outdoor_series[0]['timestamp']}")
                logger.info(f"Last outdoor point: {outdoor_series[-1]['outdoor_temp']}°F at {outdoor_series[-1]['timestamp']}")
            
            # Find current time index in the extended series
            current_time_index = self._find_current_time_index(outdoor_series)
            logger.info(f"Current time index in extended series: {current_time_index}")
            
            # Initialize state
            initial_state = {
                'indoor_temp': current_data['indoor_temp'],
                'indoor_humidity': current_data['indoor_humidity'],
                'hvac_state': current_data.get('hvac_state', 'off'),
                'timestamp': current_data['timestamp']
            }
            
            # Generate idle trajectory (no HVAC control)
            idle_trajectory = await self._simulate_trajectory(
                initial_state, outdoor_series, control_mode='idle', 
                current_time_index=current_time_index
            )
            
            # Generate controlled trajectory (smart HVAC control)  
            controlled_trajectory = await self._simulate_trajectory(
                initial_state, outdoor_series, control_mode='smart',
                current_time_index=current_time_index
            )
            
            # Generate current HVAC trajectory (maintain current state)
            current_trajectory = await self._simulate_trajectory(
                initial_state, outdoor_series, control_mode='current',
                fixed_hvac_state=current_data.get('hvac_state', 'off'),
                current_time_index=current_time_index
            )
            
            # Apply trend validation to predictions
            if trend_analysis:
                logger.info("🔧 Applying trend validation to predictions...")
                controlled_trajectory = self._apply_trend_validation(
                    controlled_trajectory, trend_analysis, current_data, current_time_index
                )
                idle_trajectory = self._apply_trend_validation(
                    idle_trajectory, trend_analysis, current_data, current_time_index
                )
            
            # Extract key results with extended timeline
            timestamps = [step['timestamp'] for step in idle_trajectory]
            
            # Log trajectory data for debugging
            logger.info(f"Trajectory generation complete:")
            logger.info(f"  - Idle trajectory: {len(idle_trajectory)} points")
            logger.info(f"  - Controlled trajectory: {len(controlled_trajectory)} points")  
            logger.info(f"  - Current trajectory: {len(current_trajectory)} points")
            logger.info(f"  - Outdoor series: {len(outdoor_series)} points")
            logger.info(f"  - Timestamps: {len(timestamps)} points")
            
            if idle_trajectory:
                logger.info(f"  - Idle temp range: {min(s['indoor_temp'] for s in idle_trajectory):.1f}°F - {max(s['indoor_temp'] for s in idle_trajectory):.1f}°F")
            if controlled_trajectory:
                logger.info(f"  - Controlled temp range: {min(s['indoor_temp'] for s in controlled_trajectory):.1f}°F - {max(s['indoor_temp'] for s in controlled_trajectory):.1f}°F")
            
            result = {
                'timestamp': datetime.now(),
                'initial_conditions': initial_state,
                'outdoor_forecast': [step['outdoor_temp'] for step in outdoor_series],
                'indoor_forecast': [step['indoor_temp'] for step in controlled_trajectory],
                'idle_trajectory': idle_trajectory,
                'controlled_trajectory': controlled_trajectory,
                'current_trajectory': current_trajectory,
                'timestamps': timestamps,
                'current_time_index': current_time_index,
                'historical_weather': historical_weather,
                'trend_analysis': trend_analysis,
                'hvac_schedule': self._extract_hvac_schedule(controlled_trajectory),
                'forecast_confidence': self._calculate_confidence(outdoor_series),
                'extended_timeline': {
                    'total_hours': len(outdoor_series) * self.time_step_minutes / 60,
                    'historical_hours': current_time_index * self.time_step_minutes / 60 if current_time_index else 0,
                    'future_hours': (len(outdoor_series) - (current_time_index or 0)) * self.time_step_minutes / 60
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}", exc_info=True)
            raise
            
    def _prepare_outdoor_series(self, current_data: Dict, weather_forecast: Dict, timezone_name: str = 'UTC') -> List[Dict]:
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
            
        # Start with current conditions in local timezone
        try:
            if timezone_name != 'UTC' and HAS_PYTZ:
                tz = pytz.timezone(timezone_name)
                current_time = datetime.now(tz)
            elif timezone_name != 'UTC' and HAS_ZONEINFO:
                import zoneinfo
                tz = zoneinfo.ZoneInfo(timezone_name)
                current_time = datetime.now(tz)
            else:
                current_time = datetime.now()
            logger.info(f"Using timezone {timezone_name} for forecast timestamps, current time: {current_time}")
        except Exception as e:
            logger.warning(f"Could not set timezone {timezone_name}, using UTC: {e}")
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
                                 fixed_hvac_state: Optional[str] = None,
                                 current_time_index: Optional[int] = None) -> List[Dict]:
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
            try:
                # Handle timezone-aware vs timezone-naive datetime comparison
                violation_time = first_violation['timestamp']
                if hasattr(violation_time, 'tzinfo') and violation_time.tzinfo is not None:
                    # Use timezone-aware current time
                    current_time = datetime.now(violation_time.tzinfo)
                else:
                    # Use naive current time
                    current_time = datetime.now()
                
                time_until = (violation_time - current_time).total_seconds() / 60
                recommendations.append(
                    f"Without HVAC, temperature will be {first_violation['violation'].replace('_', ' ')} "
                    f"in {time_until:.0f} minutes"
                )
            except TypeError as e:
                logger.warning(f"Could not calculate time until comfort violation: {e}")
                recommendations.append(
                    f"Without HVAC, temperature will be {first_violation['violation'].replace('_', ' ')}"
                )
            
        return recommendations

    def _prepare_extended_outdoor_series(self, current_data: Dict, weather_forecast: Dict, 
                                       timezone_name: str = 'UTC', historical_weather: List[Dict] = None) -> List[Dict]:
        """Prepare extended outdoor conditions time series including historical data"""
        logger.info("=== Preparing Extended Outdoor Conditions Series ===")
        series = []
        
        # Add historical data points (past 6 hours)
        if historical_weather:
            # Handle case where historical_weather might be wrapped in another list
            if len(historical_weather) == 1 and isinstance(historical_weather[0], list):
                historical_weather = historical_weather[0]
            
            logger.info(f"Adding {len(historical_weather)} historical weather points")
            valid_count = 0
            for hist_point in historical_weather:
                try:
                    # Handle string data (skip invalid entries)
                    if isinstance(hist_point, str):
                        logger.debug(f"Skipping string historical weather point: {hist_point[:50]}...")
                        continue
                    
                    # Ensure hist_point is a dictionary
                    if not isinstance(hist_point, dict):
                        logger.warning(f"Skipping invalid historical weather point (not a dict): {type(hist_point)}")
                        continue
                    
                    # Ensure required fields exist and have valid values
                    if 'timestamp' not in hist_point or 'temperature' not in hist_point:
                        logger.debug(f"Skipping historical weather point missing required fields: {list(hist_point.keys()) if isinstance(hist_point, dict) else 'not dict'}")
                        continue
                    
                    # Validate temperature is numeric
                    try:
                        temp = float(hist_point['temperature'])
                        if temp < -50 or temp > 150:  # Reasonable temperature bounds
                            logger.debug(f"Skipping historical weather point with invalid temperature: {temp}°F")
                            continue
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping historical weather point with non-numeric temperature: {hist_point['temperature']}")
                        continue
                    
                    series.append({
                        'timestamp': hist_point['timestamp'],
                        'outdoor_temp': temp,
                        'outdoor_humidity': max(0, min(100, float(hist_point.get('humidity', 50.0)))),  # Clamp 0-100%
                        'solar_irradiance': max(0, float(hist_point.get('solar_irradiance', 0.0))),  # Non-negative
                        'data_type': 'historical'
                    })
                    valid_count += 1
                except Exception as e:
                    logger.debug(f"Error processing historical weather point: {e}")
                    continue
            
            logger.info(f"Successfully added {valid_count} valid historical weather points")
        
        # Add current point
        current_outdoor_temp = current_data.get('outdoor_temp')
        if current_outdoor_temp is None:
            accuweather_temp = weather_forecast.get('current_outdoor', {}).get('temperature')
            current_outdoor_temp = accuweather_temp if accuweather_temp is not None else 70.0
        
        current_outdoor_humidity = current_data.get('outdoor_humidity')
        if current_outdoor_humidity is None:
            accuweather_humidity = weather_forecast.get('current_outdoor', {}).get('humidity')
            current_outdoor_humidity = accuweather_humidity if accuweather_humidity is not None else 50.0
            
        # Use timezone-aware current time if available, otherwise fallback to naive
        if 'timestamp' in current_data:
            current_time = current_data['timestamp']
        else:
            # Create timezone-aware current time matching the timezone context
            try:
                if timezone_name != 'UTC' and HAS_PYTZ:
                    tz = pytz.timezone(timezone_name)
                    current_time = datetime.now(tz)
                elif timezone_name != 'UTC' and HAS_ZONEINFO:
                    import zoneinfo
                    tz = zoneinfo.ZoneInfo(timezone_name)
                    current_time = datetime.now(tz)
                else:
                    current_time = datetime.now()
            except Exception as e:
                logger.warning(f"Could not set timezone {timezone_name}, using naive time: {e}")
                current_time = datetime.now()
        series.append({
            'timestamp': current_time,
            'outdoor_temp': current_outdoor_temp,
            'outdoor_humidity': current_outdoor_humidity,
            'solar_irradiance': weather_forecast.get('current_outdoor', {}).get('solar_irradiance', 0.0),
            'data_type': 'current'
        })
        
        # Add AccuWeather forecast data (future 12+ hours)
        forecast_added = 0
        if 'hourly_forecast' in weather_forecast and weather_forecast['hourly_forecast']:
            logger.info(f"Processing {len(weather_forecast['hourly_forecast'])} AccuWeather forecast points")
            for forecast_point in weather_forecast['hourly_forecast']:
                try:
                    # Validate forecast point structure
                    if not isinstance(forecast_point, dict):
                        logger.debug(f"Skipping invalid forecast point: {type(forecast_point)}")
                        continue
                    
                    # Ensure required fields exist
                    if 'timestamp' not in forecast_point or 'temperature' not in forecast_point:
                        logger.debug(f"Skipping forecast point missing required fields: {list(forecast_point.keys()) if isinstance(forecast_point, dict) else 'not dict'}")
                        continue
                    
                    # Validate temperature
                    try:
                        temp = float(forecast_point['temperature'])
                        if temp < -50 or temp > 150:  # Reasonable bounds
                            logger.debug(f"Skipping forecast point with invalid temperature: {temp}°F")
                            continue
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping forecast point with non-numeric temperature: {forecast_point['temperature']}")
                        continue
                    
                    series.append({
                        'timestamp': forecast_point['timestamp'],
                        'outdoor_temp': temp,
                        'outdoor_humidity': max(0, min(100, float(forecast_point.get('humidity', 50.0)))),
                        'solar_irradiance': max(0, float(forecast_point.get('solar_irradiance', 0.0))),
                        'data_type': 'forecast'
                    })
                    forecast_added += 1
                except Exception as e:
                    logger.debug(f"Error processing forecast point: {e}")
                    continue
            
            logger.info(f"Successfully added {forecast_added} AccuWeather forecast points")
        else:
            logger.warning("No AccuWeather hourly forecast data available - using simplified prediction")
            # Generate basic forecast points if no AccuWeather data
            current_temp = current_outdoor_temp or 70.0
            for i in range(1, 13):  # Next 12 hours
                try:
                    future_time = current_time + timedelta(hours=i)
                    # Simple temperature variation based on time of day
                    hour_of_day = future_time.hour
                    temp_variation = -10 if 2 <= hour_of_day <= 6 else (5 if 14 <= hour_of_day <= 16 else 0)
                    
                    series.append({
                        'timestamp': future_time,
                        'outdoor_temp': current_temp + temp_variation,
                        'outdoor_humidity': 50.0,  # Default
                        'solar_irradiance': 800.0 if 8 <= hour_of_day <= 18 else 0.0,  # Daylight hours
                        'data_type': 'forecast_simple'
                    })
                    forecast_added += 1
                except Exception as e:
                    logger.warning(f"Error generating simple forecast point: {e}")
                    continue
        
        # Normalize all timestamps to be timezone-aware before sorting
        for point in series:
            timestamp = point['timestamp']
            if timestamp and hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is None:
                # Convert naive datetime to timezone-aware using system timezone
                try:
                    import pytz
                    if hasattr(self, 'ha_client') and self.ha_client:
                        tz_name = self.ha_client.get_timezone() or 'UTC'
                        tz = pytz.timezone(tz_name)
                        point['timestamp'] = tz.localize(timestamp)
                    else:
                        # Fallback to UTC
                        point['timestamp'] = pytz.UTC.localize(timestamp)
                except Exception as e:
                    logger.warning(f"Could not convert timestamp to timezone-aware: {e}")
                    # Keep as-is if conversion fails
        
        # Fill gaps and ensure proper time sequence
        series = sorted(series, key=lambda x: x['timestamp'])
        series = self._fill_time_gaps(series)
        
        logger.info(f"Extended series created: {len(series)} total points")
        if series:
            logger.info(f"Time range: {series[0]['timestamp']} to {series[-1]['timestamp']}")
        
        return series

    def _find_current_time_index(self, outdoor_series: List[Dict]) -> Optional[int]:
        """Find the index in outdoor_series that corresponds to current time"""
        if not outdoor_series:
            return None
            
        # Use timezone-aware current time to match the timestamps in the series
        if outdoor_series and 'timestamp' in outdoor_series[0]:
            sample_timestamp = outdoor_series[0]['timestamp']
            if hasattr(sample_timestamp, 'tzinfo') and sample_timestamp.tzinfo is not None:
                # Use timezone-aware current time matching the series timezone
                current_time = datetime.now(sample_timestamp.tzinfo)
            else:
                # Use naive current time
                current_time = datetime.now()
        else:
            current_time = datetime.now()
        
        # Find the point closest to current time
        min_diff = float('inf')
        current_index = None
        
        for i, point in enumerate(outdoor_series):
            if point.get('data_type') == 'current':
                return i
            
            # Fallback: find closest time to now
            try:
                time_diff = abs((point['timestamp'] - current_time).total_seconds())
                if time_diff < min_diff:
                    min_diff = time_diff
                    current_index = i
            except TypeError as e:
                # Handle timezone mismatch by converting both to UTC for comparison
                logger.warning(f"Timezone comparison error at index {i}: {e}")
                try:
                    point_utc = point['timestamp'].astimezone(pytz.UTC) if hasattr(point['timestamp'], 'astimezone') else point['timestamp']
                    current_utc = current_time.astimezone(pytz.UTC) if hasattr(current_time, 'astimezone') else current_time
                    time_diff = abs((point_utc - current_utc).total_seconds())
                    if time_diff < min_diff:
                        min_diff = time_diff
                        current_index = i
                except Exception as nested_e:
                    logger.warning(f"Could not compare timestamp at index {i}: {nested_e}")
                    continue
        
        return current_index

    def _apply_trend_validation(self, trajectory: List[Dict], trend_analysis: Dict, 
                              current_conditions: Dict, current_time_index: Optional[int]) -> List[Dict]:
        """Apply trend validation to a temperature trajectory"""
        if not trend_analysis or current_time_index is None:
            return trajectory
            
        logger.info("🔧 Applying trend validation to trajectory...")
        
        validated_trajectory = []
        
        for i, step in enumerate(trajectory):
            if i < current_time_index:
                # Historical/current data - don't modify
                validated_trajectory.append(step)
                continue
            
            # Future predictions - validate against trends
            time_from_current = (i - current_time_index) * self.time_step_minutes / 60  # hours
            
            validation_result = self.thermal_model.validate_prediction_against_trends(
                step['indoor_temp'], trend_analysis, current_conditions, time_from_current
            )
            
            # Create corrected step
            corrected_step = step.copy()
            corrected_step['indoor_temp'] = validation_result['corrected_prediction']
            corrected_step['trend_validation'] = {
                'original_prediction': validation_result['original_prediction'],
                'correction_applied': validation_result['correction_applied'],
                'correction_reason': validation_result['correction_reason'],
                'confidence_score': validation_result['confidence_score']
            }
            
            validated_trajectory.append(corrected_step)
        
        logger.info(f"✅ Trend validation complete - processed {len(trajectory)} steps")
        return validated_trajectory

    def _fill_time_gaps(self, series: List[Dict]) -> List[Dict]:
        """Fill gaps in time series with interpolated values"""
        if len(series) < 2:
            return series
        
        filled_series = []
        time_step_seconds = self.time_step_minutes * 60
        
        for i in range(len(series)):
            filled_series.append(series[i])
            
            # Check if we need to fill gaps to the next point
            if i < len(series) - 1:
                current_time = series[i]['timestamp']
                next_time = series[i + 1]['timestamp']
                time_diff = (next_time - current_time).total_seconds()
                
                # If gap is larger than time step, fill with interpolated values
                if time_diff > time_step_seconds * 1.5:
                    steps_needed = int(time_diff // time_step_seconds) - 1
                    
                    for step in range(1, steps_needed + 1):
                        interp_time = current_time + timedelta(seconds=step * time_step_seconds)
                        
                        # Linear interpolation
                        ratio = step / (steps_needed + 1)
                        interp_temp = series[i]['outdoor_temp'] + ratio * (series[i + 1]['outdoor_temp'] - series[i]['outdoor_temp'])
                        interp_humidity = series[i]['outdoor_humidity'] + ratio * (series[i + 1]['outdoor_humidity'] - series[i]['outdoor_humidity'])
                        
                        filled_series.append({
                            'timestamp': interp_time,
                            'outdoor_temp': interp_temp,
                            'outdoor_humidity': interp_humidity,
                            'solar_irradiance': 0.0,
                            'data_type': 'interpolated'
                        })
        
        return filled_series