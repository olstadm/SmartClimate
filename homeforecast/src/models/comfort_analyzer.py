"""
Comfort Analyzer for HomeForecast
Analyzes forecast trajectories to determine HVAC operation recommendations
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import math

logger = logging.getLogger(__name__)


class ComfortAnalyzer:
    """
    Analyzes temperature forecasts to determine:
    - Time until comfort limits are reached
    - Optimal HVAC start/stop times
    - Energy-efficient control strategies
    """
    
    def __init__(self, config, forecast_engine):
        self.config = config
        self.forecast_engine = forecast_engine
        
        # Comfort settings (use comfort range which handles F to C conversion)
        self.comfort_min, self.comfort_max = config.get_comfort_range()
        
        # Control parameters
        self.anticipation_minutes = 30  # Look ahead for HVAC decisions
        self.min_runtime_minutes = 10   # Minimum HVAC runtime
        self.min_offtime_minutes = 10   # Minimum time between cycles
        
    def _parse_timestamp(self, timestamp):
        """Parse timestamp string to datetime object"""
        if timestamp is None:
            logger.warning("Received None timestamp")
            return None
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try standard datetime format
                    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Try with microseconds
                        return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        logger.warning(f"Could not parse timestamp: {timestamp}")
                        return None
        logger.warning(f"Invalid timestamp type: {type(timestamp)} - {timestamp}")
        return None
        
    async def analyze(self, forecast_result: Dict) -> Dict:
        """
        Analyze forecast to generate comfort recommendations
        
        Args:
            forecast_result: Output from forecast engine
            
        Returns:
            Comfort analysis with recommendations
        """
        try:
            logger.info("=== Starting Comfort Analysis ===")
            
            # Log what we received from forecast engine
            logger.info(f"Forecast result keys available: {list(forecast_result.keys())}")
            
            # Log current sensor data if available in trajectory
            if forecast_result.get('idle_trajectory'):
                first_point = forecast_result['idle_trajectory'][0]
                logger.info("Current sensor readings from trajectory:")
                logger.info(f"  Indoor temp: {first_point.get('indoor_temp', 'N/A')}°F")
                logger.info(f"  Outdoor temp: {first_point.get('outdoor_temp', 'N/A')}°F") 
                logger.info(f"  HVAC mode: {first_point.get('hvac_mode', 'N/A')}")
                logger.info(f"  Timestamp: {first_point.get('timestamp', 'N/A')}")
            
            # Get trajectories with fallback for missing keys
            idle_traj = forecast_result.get('idle_trajectory', [])
            controlled_traj = forecast_result.get('controlled_trajectory', [])
            current_traj = forecast_result.get('current_trajectory', idle_traj)  # Fallback to idle_traj
            
            logger.info(f"Trajectory data points - Idle: {len(idle_traj)}, Controlled: {len(controlled_traj)}, Current: {len(current_traj)}")
            
            # If no trajectories are available, return empty analysis
            if not idle_traj:
                logger.warning("No trajectory data available for comfort analysis")
                return {
                    'time_to_upper_limit': None,
                    'time_to_lower_limit': None,
                    'recommended_mode': 'off',
                    'comfort_score': 50,
                    'analysis': 'Insufficient data for comfort analysis'
                }
            
            # Validate trajectory data has required fields
            if not self._validate_trajectory_data(idle_traj):
                logger.error("Trajectory data missing required fields")
                return self._generate_error_response("Invalid trajectory data")
            
            logger.info("Finding critical comfort zone timing...")
            
            # Find critical times
            time_to_upper, upper_timestamp = self._find_time_to_limit(
                idle_traj, self.comfort_max, direction='upper'
            )
            time_to_lower, lower_timestamp = self._find_time_to_limit(
                idle_traj, self.comfort_min, direction='lower'
            )
            
            logger.info(f"Time to upper limit ({self.comfort_max}°F): {time_to_upper} minutes")
            logger.info(f"Time to lower limit ({self.comfort_min}°F): {time_to_lower} minutes")
            
            # Determine recommended mode (only if smart control is enabled)
            if self.config.is_smart_hvac_enabled():
                recommended_mode = self._determine_recommended_mode(
                    idle_traj[0]['indoor_temp'],
                    time_to_upper,
                    time_to_lower,
                    idle_traj[0]['outdoor_temp']
                )
            else:
                recommended_mode = 'monitoring'  # Learning mode
            
            # Calculate optimal HVAC timing
            hvac_timing = self._calculate_optimal_hvac_timing(
                idle_traj,
                controlled_traj,
                recommended_mode
            )
            
            # Analyze energy efficiency
            efficiency_metrics = self._analyze_efficiency(
                controlled_traj,
                current_traj,
                forecast_result.get('hvac_schedule', [])
            )
            
            # Generate detailed recommendations
            recommendations = self._generate_detailed_recommendations(
                forecast_result,
                time_to_upper,
                time_to_lower,
                hvac_timing,
                efficiency_metrics
            )
            
            result = {
                'time_to_upper': time_to_upper,
                'time_to_lower': time_to_lower,
                'upper_limit_timestamp': upper_timestamp,
                'lower_limit_timestamp': lower_timestamp,
                'recommended_mode': recommended_mode,
                'smart_hvac_enabled': self.config.is_smart_hvac_enabled(),
                'hvac_start_time': hvac_timing.get('start_time'),
                'hvac_stop_time': hvac_timing.get('stop_time'),
                'optimal_runtime_minutes': hvac_timing.get('runtime_minutes'),
                'efficiency_metrics': efficiency_metrics,
                'recommendations': recommendations,
                'comfort_score': self._calculate_comfort_score(controlled_traj),
                'analysis_timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comfort analysis: {e}", exc_info=True)
            return self._generate_error_response(f"Analysis error: {e}")
            
    def _validate_trajectory_data(self, trajectory: List[Dict]) -> bool:
        """Validate that trajectory data has all required fields"""
        if not trajectory:
            return False
            
        required_fields = ['indoor_temp', 'timestamp']
        for point in trajectory[:3]:  # Check first few points
            for field in required_fields:
                if field not in point:
                    logger.error(f"Missing required field '{field}' in trajectory point")
                    return False
        return True
        
    def _generate_error_response(self, error_message: str) -> Dict:
        """Generate a safe error response for comfort analysis"""
        logger.warning(f"Returning error response: {error_message}")
        return {
            'time_to_upper_limit': None,
            'time_to_lower_limit': None,
            'upper_limit_timestamp': None,
            'lower_limit_timestamp': None,
            'recommended_mode': 'off',
            'smart_hvac_enabled': False,
            'hvac_start_time': None,
            'hvac_stop_time': None,
            'hvac_duration_minutes': 0,
            'comfort_score': 0,
            'efficiency_score': 0,
            'recommendations': [{
                'type': 'system_error',
                'priority': 'high',
                'message': f'Comfort analysis unavailable: {error_message}',
                'action': 'Check system logs and sensor configuration'
            }],
            'analysis': error_message
        }
            
    def _find_time_to_limit(self, trajectory: List[Dict], limit: float, 
                           direction: str) -> Tuple[Optional[float], Optional[datetime]]:
        """
        Find when temperature will reach comfort limit
        
        Args:
            trajectory: Temperature trajectory
            limit: Temperature limit
            direction: 'upper' or 'lower'
            
        Returns:
            (minutes until limit, timestamp when limit reached)
        """
        try:
            # Use timezone-aware current time if trajectory has timezone-aware timestamps
            current_time = datetime.now()
            if trajectory and trajectory[0].get('timestamp'):
                sample_timestamp = self._parse_timestamp(trajectory[0]['timestamp'])
                if sample_timestamp and sample_timestamp.tzinfo is not None:
                    # Trajectory has timezone-aware timestamps, make current_time timezone-aware
                    try:
                        current_time = current_time.replace(tzinfo=sample_timestamp.tzinfo)
                    except Exception as tz_e:
                        logger.warning(f"Could not set timezone on current_time: {tz_e}")
                        # Convert sample to naive instead
                        current_time = datetime.now()
            
            for i, point in enumerate(trajectory):
                # Safely get temperature with fallback
                temp = point.get('indoor_temp')
                if temp is None:
                    logger.warning(f"Missing indoor_temp in trajectory point {i}")
                    continue
                    
                # Parse timestamp using helper method
                point_time = self._parse_timestamp(point.get('timestamp'))
                if point_time is None:
                    logger.warning(f"Invalid timestamp in trajectory point {i}")
                    continue
                
                if direction == 'upper' and temp >= limit:
                    # Found upper limit breach
                    minutes = (point_time - current_time).total_seconds() / 60
                    return minutes, point_time
                    
                elif direction == 'lower' and temp <= limit:
                    # Found lower limit breach
                    minutes = (point_time - current_time).total_seconds() / 60
                    return minutes, point_time
                    
            # Limit not reached in forecast period
            return None, None
            
        except Exception as e:
            logger.error(f"Error in _find_time_to_limit: {e}")
            return None, None
        
    def _determine_recommended_mode(self, current_temp: float,
                                   time_to_upper: Optional[float],
                                   time_to_lower: Optional[float],
                                   outdoor_temp: float) -> str:
        """
        Determine recommended HVAC mode based on analysis
        
        Returns:
            'heat', 'cool', 'off', or 'auto'
        """
        # Temperature distance from comfort bounds
        dist_to_min = current_temp - self.comfort_min
        dist_to_max = self.comfort_max - current_temp
        
        # If we're outside comfort range, act immediately
        if current_temp < self.comfort_min:
            return 'heat'
        elif current_temp > self.comfort_max:
            return 'cool'
            
        # Check if action needed soon
        if time_to_lower is not None and time_to_lower < self.anticipation_minutes:
            return 'heat'
        elif time_to_upper is not None and time_to_upper < self.anticipation_minutes:
            return 'cool'
            
        # Consider outdoor conditions
        outdoor_delta = outdoor_temp - current_temp
        
        # If outdoor will push us toward limits, prepare
        if outdoor_delta > 5 and dist_to_max < 2:
            # Hot outside, getting close to upper limit
            return 'cool' if time_to_upper and time_to_upper < 60 else 'off'
        elif outdoor_delta < -5 and dist_to_min < 2:
            # Cold outside, getting close to lower limit
            return 'heat' if time_to_lower and time_to_lower < 60 else 'off'
            
        return 'off'
        
    def _calculate_optimal_hvac_timing(self, idle_traj: List[Dict],
                                      controlled_traj: List[Dict],
                                      recommended_mode: str) -> Dict:
        """
        Calculate optimal HVAC start/stop times
        
        Returns:
            Dict with start_time, stop_time, runtime_minutes
        """
        if recommended_mode == 'off':
            return {
                'start_time': None,
                'stop_time': None,
                'runtime_minutes': 0
            }
            
        current_time = datetime.now()
        
        # Find latest safe start time
        start_time = None
        stop_time = None
        
        if recommended_mode == 'heat':
            # Find when we must start heating to avoid going below comfort_min
            for i, point in enumerate(idle_traj):
                if point['indoor_temp'] <= self.comfort_min:
                    # Back up to find safe start time
                    # Estimate heating rate from model
                    heat_rate = abs(self.forecast_engine.thermal_model.get_parameters()['heating_rate'])
                    needed_rise = self.comfort_min + 0.5 - point['indoor_temp']
                    lead_time_hours = needed_rise / heat_rate
                    
                    point_time = self._parse_timestamp(point['timestamp'])
                    start_time = point_time - timedelta(hours=lead_time_hours)
                    break
                    
        elif recommended_mode == 'cool':
            # Find when we must start cooling to avoid going above comfort_max
            for i, point in enumerate(idle_traj):
                if point['indoor_temp'] >= self.comfort_max:
                    # Back up to find safe start time
                    cool_rate = abs(self.forecast_engine.thermal_model.get_parameters()['cooling_rate'])
                    needed_drop = point['indoor_temp'] - (self.comfort_max - 0.5)
                    lead_time_hours = needed_drop / cool_rate
                    
                    point_time = self._parse_timestamp(point['timestamp'])
                    start_time = point_time - timedelta(hours=lead_time_hours)
                    break
                    
        # Find optimal stop time by looking at controlled trajectory
        if start_time:
            # Find when we can safely turn off
            target_temp = (self.comfort_min + self.comfort_max) / 2  # Aim for middle
            
            for point in controlled_traj:
                point_time = self._parse_timestamp(point['timestamp'])
                if point_time > start_time:
                    if recommended_mode == 'heat' and point['indoor_temp'] >= target_temp:
                        stop_time = point_time
                        break
                    elif recommended_mode == 'cool' and point['indoor_temp'] <= target_temp:
                        stop_time = point_time
                        break
                        
        # Calculate runtime
        runtime_minutes = 0
        if start_time and stop_time:
            runtime_minutes = (stop_time - start_time).total_seconds() / 60
            
            # Enforce minimum runtime
            if runtime_minutes < self.min_runtime_minutes:
                stop_time = start_time + timedelta(minutes=self.min_runtime_minutes)
                runtime_minutes = self.min_runtime_minutes
                
        # Make sure start time is not in the past
        if start_time and start_time < current_time:
            start_time = current_time
            
        return {
            'start_time': start_time,
            'stop_time': stop_time,
            'runtime_minutes': runtime_minutes
        }
        
    def _analyze_efficiency(self, controlled_traj: List[Dict],
                          current_traj: List[Dict],
                          hvac_schedule: List[Dict]) -> Dict:
        """
        Analyze energy efficiency metrics
        """
        # Calculate total runtime
        controlled_runtime = sum(1 for p in controlled_traj if p['hvac_state'] != 'off')
        current_runtime = sum(1 for p in current_traj if p['hvac_state'] != 'off')
        
        # Calculate average temperature deviation from setpoint
        setpoint = (self.comfort_min + self.comfort_max) / 2
        controlled_deviation = sum(abs(p['indoor_temp'] - setpoint) for p in controlled_traj) / len(controlled_traj)
        current_deviation = sum(abs(p['indoor_temp'] - setpoint) for p in current_traj) / len(current_traj)
        
        # Calculate cycling frequency
        controlled_cycles = len(hvac_schedule)
        
        # Energy efficiency score (0-100)
        runtime_score = max(0, 100 * (1 - controlled_runtime / max(current_runtime, 1)))
        deviation_score = max(0, 100 * (1 - controlled_deviation / max(current_deviation, 1)))
        cycle_score = max(0, 100 * (1 - controlled_cycles / 10))  # Penalize excessive cycling
        
        efficiency_score = (runtime_score + deviation_score + cycle_score) / 3
        
        return {
            'runtime_reduction_percent': 100 * (current_runtime - controlled_runtime) / max(current_runtime, 1),
            'deviation_improvement_percent': 100 * (current_deviation - controlled_deviation) / max(current_deviation, 1),
            'controlled_cycles': controlled_cycles,
            'efficiency_score': efficiency_score,
            'controlled_runtime_minutes': controlled_runtime * self.forecast_engine.time_step_minutes,
            'current_runtime_minutes': current_runtime * self.forecast_engine.time_step_minutes
        }
        
    def _calculate_comfort_score(self, trajectory: List[Dict]) -> float:
        """
        Calculate comfort score (0-100) based on time spent in comfort zone
        """
        in_comfort = 0
        total = len(trajectory)
        
        for point in trajectory:
            if self.comfort_min <= point['indoor_temp'] <= self.comfort_max:
                in_comfort += 1
                
        return 100 * in_comfort / total if total > 0 else 0
        
    def _generate_detailed_recommendations(self, forecast_result: Dict,
                                         time_to_upper: Optional[float],
                                         time_to_lower: Optional[float],
                                         hvac_timing: Dict,
                                         efficiency_metrics: Dict) -> List[Dict]:
        """
        Generate detailed, actionable recommendations
        """
        logger.info("Generating detailed recommendations...")
        
        recommendations = []
        
        # Get current temperature from trajectory (fallback for missing initial_conditions)
        current_temp = None
        if 'initial_conditions' in forecast_result and 'indoor_temp' in forecast_result['initial_conditions']:
            current_temp = forecast_result['initial_conditions']['indoor_temp']
            logger.info(f"Using current temp from initial_conditions: {current_temp}°F")
        elif forecast_result.get('idle_trajectory'):
            current_temp = forecast_result['idle_trajectory'][0]['indoor_temp']
            logger.info(f"Using current temp from first trajectory point: {current_temp}°F")
        else:
            logger.warning("No current temperature available - using comfort zone midpoint")
            current_temp = (self.comfort_min + self.comfort_max) / 2
        
        # Smart HVAC control status
        if not self.config.is_smart_hvac_enabled():
            recommendations.append({
                'type': 'system_status',
                'priority': 'medium',
                'message': 'Smart HVAC Control is disabled - System is in learning mode',
                'action': 'Enable Smart HVAC Control in settings when ready to receive automated recommendations'
            })
        
        # Temperature trend recommendation
        if time_to_lower is not None and time_to_lower < 60:
            recommendations.append({
                'type': 'temperature_warning',
                'priority': 'high',
                'message': f"Temperature will drop below comfort zone in {time_to_lower:.0f} minutes",
                'action': 'Consider starting heating soon'
            })
        elif time_to_upper is not None and time_to_upper < 60:
            recommendations.append({
                'type': 'temperature_warning',
                'priority': 'high',
                'message': f"Temperature will exceed comfort zone in {time_to_upper:.0f} minutes",
                'action': 'Consider starting cooling soon'
            })
            
        # HVAC timing recommendation
        if hvac_timing['start_time']:
            start_in_minutes = (hvac_timing['start_time'] - datetime.now()).total_seconds() / 60
            if start_in_minutes > 0:
                recommendations.append({
                    'type': 'hvac_schedule',
                    'priority': 'medium',
                    'message': f"Optimal HVAC start time in {start_in_minutes:.0f} minutes",
                    'action': f"Schedule {forecast_result['initial_conditions']['hvac_state']} to start at {hvac_timing['start_time'].strftime('%H:%M')}"
                })
                
        # Efficiency recommendation
        if efficiency_metrics['runtime_reduction_percent'] > 20:
            recommendations.append({
                'type': 'efficiency',
                'priority': 'low',
                'message': f"Smart control can reduce HVAC runtime by {efficiency_metrics['runtime_reduction_percent']:.0f}%",
                'action': 'Enable predictive HVAC control for energy savings'
            })
            
        # Comfort optimization
        if current_temp < self.comfort_min or current_temp > self.comfort_max:
            recommendations.append({
                'type': 'comfort',
                'priority': 'high',
                'message': 'Currently outside comfort zone',
                'action': 'Immediate HVAC action recommended'
            })
            
        # Pre-conditioning recommendation
        outdoor_temp = forecast_result['outdoor_forecast'][0]
        temp_diff = abs(outdoor_temp - current_temp)
        
        if temp_diff > 10:
            if outdoor_temp > current_temp and current_temp < (self.comfort_min + self.comfort_max) / 2:
                recommendations.append({
                    'type': 'pre_conditioning',
                    'priority': 'medium',
                    'message': f"Large temperature difference ({temp_diff:.1f}°C) between indoor and outdoor",
                    'action': 'Consider pre-cooling to reduce peak load'
                })
            elif outdoor_temp < current_temp and current_temp > (self.comfort_min + self.comfort_max) / 2:
                recommendations.append({
                    'type': 'pre_conditioning',
                    'priority': 'medium',
                    'message': f"Large temperature difference ({temp_diff:.1f}°C) between indoor and outdoor",
                    'action': 'Consider pre-heating to maintain comfort efficiently'
                })
                
        return recommendations
        
    def generate_automation_triggers(self, analysis_result: Dict) -> List[Dict]:
        """
        Generate Home Assistant automation triggers based on analysis
        """
        triggers = []
        
        # Start HVAC trigger
        if analysis_result['hvac_start_time']:
            triggers.append({
                'platform': 'time',
                'at': analysis_result['hvac_start_time'].strftime('%H:%M:%S'),
                'action': {
                    'service': 'climate.set_hvac_mode',
                    'data': {
                        'hvac_mode': analysis_result['recommended_mode']
                    }
                }
            })
            
        # Stop HVAC trigger
        if analysis_result['hvac_stop_time']:
            triggers.append({
                'platform': 'time',
                'at': analysis_result['hvac_stop_time'].strftime('%H:%M:%S'),
                'action': {
                    'service': 'climate.set_hvac_mode',
                    'data': {
                        'hvac_mode': 'off'
                    }
                }
            })
            
        # Temperature threshold triggers
        if analysis_result['time_to_lower'] and analysis_result['time_to_lower'] < 30:
            triggers.append({
                'platform': 'numeric_state',
                'entity_id': 'sensor.homeforecast_forecast_temperature',
                'below': self.comfort_min + 0.5,
                'action': {
                    'service': 'climate.set_hvac_mode',
                    'data': {
                        'hvac_mode': 'heat'
                    }
                }
            })
            
        return triggers