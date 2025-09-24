"""
Web server and API for HomeForecast addon
Provides REST API and web interface for Home Assistant integration
"""
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import asyncio
import threading
import os
import tempfile

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

# Import V2.0 components
try:
    # Try relative imports first
    try:
        from ..models.enhanced_training_system import EnhancedTrainingSystem
    except ImportError:
        # Fallback to absolute imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
        from enhanced_training_system import EnhancedTrainingSystem
    
    HAS_ENHANCED_TRAINING = True
    ENHANCED_TRAINING_ERROR = None
    TrainingSystemClass = EnhancedTrainingSystem
except ImportError as e:
    HAS_ENHANCED_TRAINING = False
    ENHANCED_TRAINING_ERROR = str(e)
    # Try to import simple training system as fallback
    try:
        try:
            from ..models.simple_training_system import SimpleTrainingSystem
        except ImportError:
            # Fallback to absolute imports
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
            from simple_training_system import SimpleTrainingSystem
        
        TrainingSystemClass = SimpleTrainingSystem
        HAS_SIMPLE_TRAINING = True
    except ImportError as e2:
        HAS_SIMPLE_TRAINING = False
        TrainingSystemClass = None
        ENHANCED_TRAINING_ERROR += f"; SimpleTrainingSystem also failed: {e2}"


def format_time_consistent(dt, include_seconds=False) -> str:
    """Format time consistently across the system (h:mm AM/PM or h:mm:ss AM/PM)"""
    import platform
    
    # Choose format string based on platform and seconds preference
    if include_seconds:
        if platform.system() == 'Windows':
            return dt.strftime("%#I:%M:%S %p")
        else:
            return dt.strftime("%-I:%M:%S %p")
    else:
        if platform.system() == 'Windows':
            return dt.strftime("%#I:%M %p")
        else:
            return dt.strftime("%-I:%M %p")


def convert_timestamps_to_local_timezone(data, target_timezone='America/Los_Angeles'):
    """Convert UTC timestamps in forecast data to local timezone for frontend display"""
    if not HAS_PYTZ:
        logger.warning("pytz not available for timezone conversion, returning data as-is")
        return data
        
    try:
        # Get target timezone
        if target_timezone and target_timezone != 'UTC':
            target_tz = pytz.timezone(target_timezone)
        else:
            target_tz = pytz.UTC
            
        def convert_timestamp_list(timestamps):
            """Convert a list of timestamp strings to local timezone"""
            converted = []
            for ts in timestamps:
                if isinstance(ts, str):
                    try:
                        # Parse the timestamp
                        if ts.endswith('Z'):
                            dt = datetime.fromisoformat(ts[:-1]).replace(tzinfo=pytz.UTC)
                        elif '+' in ts or ts.count('-') > 2:  # Has timezone info
                            dt = datetime.fromisoformat(ts)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=pytz.UTC)
                        else:
                            # Assume UTC if no timezone info
                            dt = datetime.fromisoformat(ts).replace(tzinfo=pytz.UTC)
                        
                        # Convert to target timezone and format for frontend
                        local_dt = dt.astimezone(target_tz)
                        # Return in ISO format but without timezone info for JavaScript
                        converted.append(local_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])
                        
                    except Exception as e:
                        logger.warning(f"Could not convert timestamp {ts}: {e}")
                        converted.append(ts)  # Keep original if conversion fails
                else:
                    converted.append(ts)
            return converted
        
        # Convert timestamps in the data structure
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == 'timestamps' and isinstance(value, list):
                    result[key] = convert_timestamp_list(value)
                elif isinstance(value, dict):
                    result[key] = convert_timestamps_to_local_timezone(value, target_timezone)
                elif isinstance(value, list) and key in ['timestamps']:
                    result[key] = convert_timestamp_list(value)
                else:
                    result[key] = value
            return result
        else:
            return data
            
    except Exception as e:
        logger.warning(f"Timezone conversion failed: {e}")
        return data


def convert_numpy_types(obj):
    """Convert numpy types and other problematic types to native Python types for JSON serialization"""
    import datetime
    
    # Handle None values
    if obj is None:
        return None
    
    # Handle numpy types
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif HAS_NUMPY and isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle numpy boolean types (version-safe)
    if HAS_NUMPY:
        try:
            if isinstance(obj, np.bool_):
                return bool(obj)
        except (AttributeError, TypeError):
            pass
        
        # Handle numpy integer types
        try:
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
        except (AttributeError, TypeError):
            pass
        
        # Handle numpy float types
        try:
            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
        except (AttributeError, TypeError):
            pass
    
    # Handle Python boolean explicitly
    elif isinstance(obj, (bool, type(True), type(False))):
        return bool(obj)
    
    # Handle datetime objects
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, datetime.time):
        return obj.isoformat()
    
    # Handle complex data structures
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, set):
        return [convert_numpy_types(v) for v in obj]
    
    # Handle other numeric types that might cause issues
    elif hasattr(obj, '__int__') and not isinstance(obj, str):
        try:
            return int(obj)
        except (ValueError, TypeError):
            pass
    elif hasattr(obj, '__float__') and not isinstance(obj, str):
        try:
            return float(obj)
        except (ValueError, TypeError):
            pass
    
    # For anything else, convert to string if it's not JSON serializable
    try:
        import json
        json.dumps(obj)  # Test if it's serializable
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _analyze_hvac_schedule_from_trajectory(trajectory, current_temp, hvac_mode):
    """Extract HVAC timing insights from forecast trajectory"""
    insights = {
        'recommended_action': 'MONITOR',
        'next_action_time': 'N/A',
        'action_off_time': 'N/A',
        'estimated_runtime': 'N/A'
    }
    
    if not trajectory or len(trajectory) < 2:
        return insights


def _analyze_optimal_hvac_timing(idle_trajectory, controlled_trajectory, current_temp, hvac_mode, comfort_min=68.0, comfort_max=80.0):
    """
    Analyze uncontrolled temperature trajectory to determine optimal HVAC timing
    Based on 'set it and forget it' logic using comfort thresholds
    
    Logic:
    - COOLING: Turn on when uncontrolled temp would exceed comfort_max, turn off when it drops to comfort_max
    - HEATING: Turn on when uncontrolled temp would fall below comfort_min, turn off when it rises to comfort_min
    """
    insights = {
        'recommended_action': 'MONITOR',
        'next_action_time': 'N/A',
        'action_off_time': 'N/A',
        'estimated_runtime': 'N/A'
    }
    
    if not idle_trajectory or len(idle_trajectory) < 2:
        return insights
    
    try:
        # Use timezone-aware datetime to match trajectory timestamps
        if HAS_PYTZ:
            pacific_tz = pytz.timezone('America/Los_Angeles')
            now = datetime.now(pacific_tz)
        else:
            now = datetime.now()
            
        logger.info(f"🔍 Analyzing optimal HVAC timing - Mode: {hvac_mode}, Current: {current_temp}°F, " +
                   f"Comfort range: {comfort_min}-{comfort_max}°F")
        
        # Debug: Log uncontrolled temperature trajectory for physics validation
        if idle_trajectory and len(idle_trajectory) > 0:
            idle_temps = [point.get('indoor_temp', current_temp) for point in idle_trajectory[:10]]  # First 10 points
            logger.info(f"📊 Uncontrolled temp trend (next 2.5hrs): {idle_temps[:6]} ...")
            
            # Physics check: ensure uncontrolled temps are realistic
            if len(idle_temps) >= 2:
                temp_change = idle_temps[1] - idle_temps[0]  # Change in first step
                if abs(temp_change) > 1.0:  # > 1°F change in 5 minutes is suspicious
                    logger.warning(f"⚠️ Physics Alert: Uncontrolled temp shows {temp_change:.2f}°F change in 5min " +
                                 f"({idle_temps[0]:.1f}°F → {idle_temps[1]:.1f}°F)")
        
        turn_on_time = None
        turn_off_time = None
        
        # Check if we need immediate action based on current temperature
        if hvac_mode == 'cool' and current_temp >= comfort_max:
            insights['recommended_action'] = 'COOL NOW'
            insights['next_action_time'] = "Now"
            turn_on_time = now
            
        elif hvac_mode == 'heat' and current_temp <= comfort_min:
            insights['recommended_action'] = 'HEAT NOW'
            insights['next_action_time'] = "Now"  
            turn_on_time = now
            
        elif hvac_mode == 'off':
            insights['recommended_action'] = 'SYSTEM OFF'
            insights['next_action_time'] = 'Switch mode to HEAT/COOL'
            insights['action_off_time'] = 'N/A (Mode set to OFF)'
            insights['estimated_runtime'] = 'N/A (OFF Mode)'
            return insights
            
        else:
            # Analyze uncontrolled trajectory to find optimal timing
            for point in idle_trajectory:
                timestamp = point.get('timestamp', now)
                idle_temp = point.get('indoor_temp', current_temp)
                
                # Skip historical points - only look at future
                if timestamp <= now:
                    continue
                    
                # COOLING MODE: Turn on when uncontrolled temp would exceed max
                if hvac_mode == 'cool' and not turn_on_time:
                    if idle_temp >= comfort_max:
                        turn_on_time = timestamp
                        insights['recommended_action'] = 'COOL SOON'
                        insights['next_action_time'] = format_time_consistent(turn_on_time)
                        logger.info(f"📅 Cool start needed at {format_time_consistent(turn_on_time)} " +
                                  f"when uncontrolled temp reaches {idle_temp:.1f}°F")
                        
                # HEATING MODE: Turn on when uncontrolled temp would fall below min  
                elif hvac_mode == 'heat' and not turn_on_time:
                    if idle_temp <= comfort_min:
                        turn_on_time = timestamp
                        insights['recommended_action'] = 'HEAT SOON'
                        insights['next_action_time'] = format_time_consistent(turn_on_time)
                        logger.info(f"📅 Heat start needed at {format_time_consistent(turn_on_time)} " +
                                  f"when uncontrolled temp drops to {idle_temp:.1f}°F")
        
        # Find turn off time by analyzing when controlled temperature returns to comfort zone
        if turn_on_time and controlled_trajectory:
            for point in controlled_trajectory:
                timestamp = point.get('timestamp', now)
                controlled_temp = point.get('indoor_temp', current_temp)
                
                # Skip to points after turn_on_time
                if timestamp <= turn_on_time:
                    continue
                    
                # COOLING: Turn off when controlled temp reaches comfort_max (our setpoint)
                if hvac_mode == 'cool':
                    if controlled_temp <= comfort_max:
                        turn_off_time = timestamp
                        insights['action_off_time'] = format_time_consistent(turn_off_time)
                        break
                        
                # HEATING: Turn off when controlled temp reaches comfort_min (our setpoint)
                elif hvac_mode == 'heat':
                    if controlled_temp >= comfort_min:
                        turn_off_time = timestamp
                        insights['action_off_time'] = format_time_consistent(turn_off_time)
                        break
        
        # Calculate estimated runtime
        if turn_on_time and turn_off_time:
            runtime_hours = (turn_off_time - turn_on_time).total_seconds() / 3600
            insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
            logger.info(f"⏱️ Estimated runtime: {insights['estimated_runtime']} " +
                       f"({format_time_consistent(turn_on_time)} to {format_time_consistent(turn_off_time)})")
        elif turn_on_time:
            insights['action_off_time'] = "Beyond forecast"
            insights['estimated_runtime'] = "Beyond forecast"
            
        # If no action needed in forecast period
        if not turn_on_time and hvac_mode in ['cool', 'heat']:
            insights['recommended_action'] = 'NO ACTION NEEDED'
            insights['next_action_time'] = 'Beyond forecast period'
            insights['action_off_time'] = 'N/A'
            insights['estimated_runtime'] = '0 min'
            
    except Exception as e:
        logger.warning(f"Error analyzing optimal HVAC timing: {e}")
        
    return insights


def _analyze_hvac_schedule_from_trajectory(trajectory, current_temp, hvac_mode):
    """Legacy function - kept for backwards compatibility"""
    insights = {
        'recommended_action': 'MONITOR',
        'next_action_time': 'N/A',
        'action_off_time': 'N/A',
        'estimated_runtime': 'N/A'
    }
    
    if not trajectory or len(trajectory) < 2:
        return insights


def calculate_climate_insights(current_data, thermostat_data, config, comfort_analyzer=None):
    """Calculate intelligent climate action insights with HVAC timing predictions"""
    try:
        insights = {
            'recommended_action': 'MONITOR',
            'next_action_time': 'N/A',  # When to turn on
            'action_off_time': 'N/A',   # When to turn off
            'estimated_runtime': 'N/A'
        }
        
        # Get current conditions
        current_temp = current_data.get('indoor_temp', 70.0)
        target_temp = thermostat_data.get('target_temperature', 72.0)
        hvac_mode = thermostat_data.get('hvac_mode', 'off')
        hvac_action = thermostat_data.get('hvac_action', 'idle')
        hvac_state = thermostat_data.get('hvac_state', 'off')
        
        # Also check HVAC state flags for more accurate status
        hvac_heating = thermostat_data.get('hvac_heat', False) or thermostat_data.get('heating', False)
        hvac_cooling = thermostat_data.get('hvac_cool', False) or thermostat_data.get('cooling', False)
        
        # Enhanced HVAC state detection using multiple sources
        if hvac_heating and not hvac_cooling:
            hvac_action = 'heating'
        elif hvac_cooling and not hvac_heating:
            hvac_action = 'cooling'
        elif hvac_state in ['heat', 'heating']:
            hvac_action = 'heating'
        elif hvac_state in ['cool', 'cooling']:
            hvac_action = 'cooling'
        elif not hvac_heating and not hvac_cooling and hvac_state == 'off':
            hvac_action = 'idle'
            
        logger.info(f"🌡️ Climate Context: Temp={current_temp}°F, Target={target_temp}°F, Mode={hvac_mode}, " +
                   f"Action={hvac_action}, State={hvac_state}, Heating={hvac_heating}, Cooling={hvac_cooling}")
        
        # Use actual comfort range from config (62-80°F full band)
        comfort_min = config.get('comfort_min_temp', 62.0)
        comfort_max = config.get('comfort_max_temp', 80.0)
        
        logger.info(f"📊 Using comfort band: {comfort_min}°F - {comfort_max}°F (target setpoint: {target_temp}°F)")
        
        # Try to get optimal timing insights from comfort analyzer trajectories
        forecast_insights = None
        if comfort_analyzer and hasattr(comfort_analyzer, 'homeforecast'):
            try:
                # Check if we have recent forecast data stored in comfort analyzer
                if hasattr(comfort_analyzer, 'latest_forecast_data') and comfort_analyzer.latest_forecast_data:
                    forecast_data = comfort_analyzer.latest_forecast_data
                    
                    # Use both idle and controlled trajectories for optimal timing analysis
                    if 'idle_trajectory' in forecast_data and 'controlled_trajectory' in forecast_data:
                        idle_trajectory = forecast_data['idle_trajectory']
                        controlled_trajectory = forecast_data['controlled_trajectory']
                        
                        # Use the new optimal timing analysis
                        forecast_insights = _analyze_optimal_hvac_timing(
                            idle_trajectory, controlled_trajectory, current_temp, hvac_mode, 
                            comfort_min, comfort_max
                        )
                        logger.debug(f"Optimal timing analysis: idle={len(idle_trajectory)} points, " +
                                   f"controlled={len(controlled_trajectory)} points")
                    
                    # Fallback to legacy method if only controlled trajectory available
                    elif 'controlled_trajectory' in forecast_data:
                        trajectory = forecast_data['controlled_trajectory']
                        forecast_insights = _analyze_hvac_schedule_from_trajectory(trajectory, current_temp, hvac_mode)
                        logger.debug(f"Legacy trajectory analysis: {len(trajectory)} points")
                        
            except Exception as e:
                logger.warning(f"Could not get forecast insights: {e}")
        
        # If we have forecast insights, use them
        if forecast_insights:
            insights.update(forecast_insights)
            logger.info(f"Using forecast-based optimal timing: {insights['recommended_action']} " +
                       f"(On: {insights['next_action_time']}, Off: {insights['action_off_time']})")
        else:
            # Fallback to current state analysis
            logger.info("Using current state for climate insights")
        
        # HVAC performance estimates (°F/hour)
        heating_rate = 3.5
        cooling_rate = 4.0
        drift_rate = 0.5  # Natural temperature drift when HVAC is off
        
        logger.info(f"Climate insights calculation - Current: {current_temp}°F, Target: {target_temp}°F, " +
                   f"Mode: {hvac_mode}, Action: {hvac_action}, Comfort: {comfort_min}-{comfort_max}°F")
        
        # Use timezone-aware datetime to match trajectory timestamps (PDT/PST)
        if HAS_PYTZ:
            pacific_tz = pytz.timezone('America/Los_Angeles')
            now = datetime.now(pacific_tz)
            logger.debug(f"Using Pacific timezone: {now} (PDT/PST)")
        else:
            now = datetime.now()
            logger.warning("pytz not available, using naive datetime - runtime calculations may be inaccurate")
        
        # Context-aware insights: Focus on hvac_mode (what we control) rather than hvac_action (what system does)
        # Priority: Mode settings > Physical heating/cooling flags > Action status
        is_currently_heating = hvac_mode == 'heat' or (hvac_heating and hvac_mode != 'cool') or (hvac_action in ['heating', 'heat'] and hvac_mode != 'cool')
        is_currently_cooling = hvac_mode == 'cool' or (hvac_cooling and hvac_mode != 'heat') or (hvac_action in ['cooling', 'cool'] and hvac_mode != 'heat')
        
        logger.info(f"HVAC Context - Mode: {hvac_mode} (what we control), Action: {hvac_action} (what system does)")
        logger.info(f"Physical State - Heating flag: {hvac_heating}, Cooling flag: {hvac_cooling}")
        logger.info(f"Decision Logic - Currently heating: {is_currently_heating}, Currently cooling: {is_currently_cooling}")
        
        # Mode-focused insights: Provide recommendations based on what user can control (hvac_mode)
        mode_recommendation = ""
        if hvac_mode == 'heat':
            mode_recommendation = "System set to HEAT mode - will heat when needed"
        elif hvac_mode == 'cool': 
            mode_recommendation = "System set to COOL mode - will cool when needed"
        elif hvac_mode == 'auto':
            mode_recommendation = "System set to AUTO mode - will heat/cool as needed"
        elif hvac_mode == 'off':
            mode_recommendation = "System is OFF - no heating or cooling"
        
        logger.info(f"Mode Analysis: {mode_recommendation}")
        
        # HVAC is actively heating
        if is_currently_heating:
            insights['recommended_action'] = 'HEATING'
            
            # Context: Already heating, focus on OFF time and runtime
            if current_temp < target_temp:
                temp_diff = target_temp - current_temp
                runtime_hours = temp_diff / heating_rate
                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                off_time = now + timedelta(hours=runtime_hours)
                insights['action_off_time'] = format_time_consistent(off_time)
                
                # Don't show next ON time since we're already heating
                insights['next_action_time'] = 'Currently Active'
            else:
                insights['action_off_time'] = "Now"
                insights['estimated_runtime'] = "0 min"
                insights['next_action_time'] = 'Currently Active'
                
        # HVAC is actively cooling  
        elif is_currently_cooling:
            insights['recommended_action'] = 'COOLING'
            
            # Context: Already cooling, focus on OFF time and runtime
            if current_temp > target_temp:
                temp_diff = current_temp - target_temp
                runtime_hours = temp_diff / cooling_rate
                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                off_time = now + timedelta(hours=runtime_hours)
                insights['action_off_time'] = format_time_consistent(off_time)
                
                # Don't show next ON time since we're already cooling
                insights['next_action_time'] = 'Currently Active'
            else:
                insights['action_off_time'] = "Now"
                insights['estimated_runtime'] = "0 min"
                insights['next_action_time'] = 'Currently Active'
                
        # HVAC is off or idle - predict when to start
        else:
            # Context: HVAC is idle, focus on NEXT ON time and estimated duration
            
            # Enhanced logic: Use thermostat setpoint for more accurate predictions
            if target_temp:
                # Calculate temperature difference from setpoint
                temp_diff_from_setpoint = current_temp - target_temp
                
                # Determine if immediate action is needed based on setpoint and mode
                if hvac_mode == 'cool' and temp_diff_from_setpoint > 1.0:
                    # Cooling mode and significantly above setpoint
                    insights['recommended_action'] = 'COOL NOW'
                    insights['next_action_time'] = "Now"
                    runtime_hours = abs(temp_diff_from_setpoint) / cooling_rate
                    insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                    insights['action_off_time'] = f"After reaching {target_temp}°F"
                    
                elif hvac_mode == 'heat' and temp_diff_from_setpoint < -1.0:
                    # Heating mode and significantly below setpoint
                    insights['recommended_action'] = 'HEAT NOW'
                    insights['next_action_time'] = "Now"
                    runtime_hours = abs(temp_diff_from_setpoint) / heating_rate
                    insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                    insights['action_off_time'] = f"After reaching {target_temp}°F"
                    
                # Fallback to comfort range logic for other modes
                elif current_temp < comfort_min and hvac_mode in ['heat', 'heat_cool', 'auto']:
                    insights['recommended_action'] = 'HEAT NOW'
                    insights['next_action_time'] = "Now"
                    temp_diff = comfort_min - current_temp
                    runtime_hours = temp_diff / heating_rate
                    insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                    insights['action_off_time'] = 'After heating completes'
                    
                elif current_temp > comfort_max and hvac_mode in ['cool', 'heat_cool', 'auto']:
                    insights['recommended_action'] = 'COOL NOW'
                    insights['next_action_time'] = "Now"
                    temp_diff = current_temp - comfort_max
                    runtime_hours = temp_diff / cooling_rate
                    insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                    insights['action_off_time'] = 'After cooling completes'
                
                else:
                    # Mode-focused recommendations
                    if hvac_mode == 'off':
                        insights['recommended_action'] = 'SYSTEM OFF'
                        insights['action_off_time'] = 'N/A (Mode set to OFF)'
                        insights['next_action_time'] = 'Switch mode to HEAT/COOL'
                        insights['estimated_runtime'] = 'N/A (OFF Mode)'
                    else:
                        insights['recommended_action'] = 'MONITOR'
                        insights['action_off_time'] = 'N/A (Currently idle)'
                    
                    # Predict when action will be needed based on setpoint and mode (only if not OFF)
                    if hvac_mode == 'cool':
                        # Cooling mode - predict when temp will rise above setpoint + deadband
                        temp_prediction = current_temp
                        for hour in range(1, 13):  # Check next 12 hours
                            temp_prediction += drift_rate
                            if temp_prediction >= target_temp + 1.0:  # 1°F above setpoint
                                cool_start_time = now + timedelta(hours=max(0, hour-0.5))
                                insights['next_action_time'] = format_time_consistent(cool_start_time)
                                # Estimate cooling duration to return to setpoint
                                temp_to_cool = 2.0  # Cool from +1°F to -0.5°F around setpoint
                                runtime_hours = temp_to_cool / cooling_rate
                                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min (cooling)"
                                break
                                
                    elif hvac_mode == 'heat':
                        # Heating mode - predict when temp will fall below setpoint - deadband
                        temp_prediction = current_temp
                        for hour in range(1, 13):  # Check next 12 hours
                            temp_prediction -= drift_rate
                            if temp_prediction <= target_temp - 1.0:  # 1°F below setpoint
                                heat_start_time = now + timedelta(hours=max(0, hour-0.5))
                                insights['next_action_time'] = format_time_consistent(heat_start_time)
                                # Estimate heating duration to return to setpoint
                                temp_to_heat = 2.0  # Heat from -1°F to +0.5°F around setpoint
                                runtime_hours = temp_to_heat / heating_rate
                                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min (heating)"
                                break
                    
                    # Fallback to comfort range predictions for auto/heat_cool modes
                    elif hvac_mode in ['heat_cool', 'auto']:
                        temp_prediction = current_temp
                        for hour in range(1, 13):  # Check next 12 hours
                            temp_prediction_cold = temp_prediction - (drift_rate * hour)
                            temp_prediction_hot = temp_prediction + (drift_rate * hour)
                            
                            if temp_prediction_cold <= comfort_min:
                                heat_start_time = now + timedelta(hours=max(0, hour-0.5))
                                insights['next_action_time'] = format_time_consistent(heat_start_time)
                                runtime_hours = (comfort_min + 1 - temp_prediction_cold) / heating_rate
                                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min (heating)"
                                break
                            elif temp_prediction_hot >= comfort_max:
                                cool_start_time = now + timedelta(hours=max(0, hour-0.5))
                                insights['next_action_time'] = format_time_consistent(cool_start_time)
                                runtime_hours = (temp_prediction_hot - comfort_max + 1) / cooling_rate
                                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min (cooling)"
                                break
            else:
                # No target temperature available, use comfort range fallback
                insights['recommended_action'] = 'MONITOR'
                insights['next_action_time'] = 'N/A (No setpoint)'
        
        # Format times based on timezone if available
        if hasattr(comfort_analyzer, 'homeforecast') and hasattr(comfort_analyzer.homeforecast, 'ha_client'):
            try:
                if insights['action_off_time'] not in ['N/A', 'Now']:
                    off_dt = datetime.strptime(insights['action_off_time'], "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day)
                    insights['action_off_time'] = comfort_analyzer.homeforecast.ha_client.format_time_for_display(off_dt)
                    
                if insights['next_action_time'] not in ['N/A', 'Now']:
                    next_dt = datetime.strptime(insights['next_action_time'], "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day)
                    insights['next_action_time'] = comfort_analyzer.homeforecast.ha_client.format_time_for_display(next_dt)
            except Exception as e:
                logger.warning(f"Could not format times with timezone: {e}")
        
        logger.info(f"Climate insights result: {insights}")
        return insights
        
    except Exception as e:
        logger.error(f"Error calculating climate insights: {e}")
        return {
            'recommended_action': 'UNKNOWN',
            'action_off_time': 'N/A',
            'next_action_time': 'N/A',
            'estimated_runtime': 'N/A'
        }


def create_app(homeforecast_instance):
    """Create Flask application with HomeForecast integration"""
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Set up Flask with template and static directories
    app = Flask(__name__,
                template_folder=os.path.join(parent_dir, 'templates'),
                static_folder=os.path.join(parent_dir, 'static'))
    
    # Configure file upload limits for building models and weather datasets
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit for EPW files
    app.config['UPLOAD_FOLDER'] = '/tmp/homeforecast_uploads'
    
    CORS(app)
    
    # Store reference to HomeForecast instance
    app.homeforecast = homeforecast_instance
    
    # API Routes
    
    @app.route('/api/status')
    def get_status():
        """Get current addon status"""
        try:
            logger.info("🌐 API: Processing /api/status request")
            
            # Get current sensor data for display
            current_data = {}
            try:
                # This will be synchronous, so we need to be careful about blocking
                # For now, we'll just get the model state which should have the latest data
                model_params = app.homeforecast.thermal_model.get_parameters()
                logger.info(f"Model parameters retrieved: {model_params}")
                
                # Try to get current data from thermal model
                indoor_temp = getattr(app.homeforecast.thermal_model, 'current_indoor_temp', None)
                outdoor_temp = getattr(app.homeforecast.thermal_model, 'current_outdoor_temp', None)
                hvac_state = getattr(app.homeforecast.thermal_model, 'current_hvac_state', 'unknown')
                
                logger.info(f"Current data from model - Indoor: {indoor_temp}°F, Outdoor: {outdoor_temp}°F, HVAC: {hvac_state}")
                
                # If model doesn't have current data, try to get from data store
                if indoor_temp is None:
                    logger.info("No current data in model, trying data store...")
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        recent_data = loop.run_until_complete(
                            app.homeforecast.data_store.get_recent_measurements(hours=1)
                        )
                        if recent_data:
                            latest = recent_data[-1]  # Most recent measurement
                            indoor_temp = latest.get('indoor_temp')
                            outdoor_temp = latest.get('outdoor_temp')
                            hvac_state = latest.get('hvac_state', 'unknown')
                            logger.info(f"Data from store - Indoor: {indoor_temp}°F, Outdoor: {outdoor_temp}°F, HVAC: {hvac_state}")
                    except Exception as store_e:
                        logger.warning(f"Could not get data from store: {store_e}")
                
                current_data = {
                    'indoor_temp': indoor_temp,
                    'outdoor_temp': outdoor_temp,
                    'hvac_state': hvac_state
                }
                logger.info(f"Final current_data: {current_data}")
                
            except Exception as e:
                logger.warning(f"Could not get current sensor data: {e}")
                current_data = {
                    'indoor_temp': None,
                    'outdoor_temp': None,
                    'hvac_state': 'unknown'
                }
            
            # Get thermostat data if available
            thermostat_data = {}
            try:
                if hasattr(app.homeforecast, 'last_sensor_data') and app.homeforecast.last_sensor_data:
                    thermostat_data = app.homeforecast.last_sensor_data.get('thermostat_data', {})
                    logger.info(f"Retrieved thermostat data: {thermostat_data}")
            except Exception as e:
                logger.warning(f"Could not get thermostat data: {e}")

            # Calculate climate action insights
            climate_insights = calculate_climate_insights(
                current_data, 
                thermostat_data,
                app.homeforecast.config,
                app.homeforecast.comfort_analyzer if hasattr(app.homeforecast, 'comfort_analyzer') else None
            )

            # Format last update time with timezone
            last_update_str = None
            if app.homeforecast.thermal_model.last_update:
                if hasattr(app.homeforecast, 'ha_client'):
                    last_update_str = app.homeforecast.ha_client.format_datetime_for_display(
                        app.homeforecast.thermal_model.last_update
                    )
                else:
                    # Use consistent date formatting: m/d h:mm AM/PM
                    import platform
                    if platform.system() == 'Windows':
                        last_update_str = app.homeforecast.thermal_model.last_update.strftime("%#m/%#d %#I:%M %p")
                    else:
                        last_update_str = app.homeforecast.thermal_model.last_update.strftime("%-m/%-d %-I:%M %p")

            # Get ML model performance data
            ml_performance = {}
            try:
                if (hasattr(app.homeforecast.thermal_model, 'ml_corrector') and 
                    app.homeforecast.thermal_model.ml_corrector and
                    hasattr(app.homeforecast.thermal_model.ml_corrector, 'performance_history') and
                    app.homeforecast.thermal_model.ml_corrector.performance_history):
                    
                    latest_perf = app.homeforecast.thermal_model.ml_corrector.performance_history[-1]
                    ml_performance = {
                        'status': 'trained' if app.homeforecast.thermal_model.ml_corrector.is_trained else 'not_trained',
                        'mae': round(latest_perf.get('mae', 0), 3),
                        'r2': round(latest_perf.get('r2', 0), 3),
                        'training_samples': latest_perf.get('training_samples', 0),
                        'last_update': latest_perf.get('timestamp').isoformat() if latest_perf.get('timestamp') else None
                    }
                else:
                    ml_performance = {
                        'status': 'disabled' if not app.homeforecast.config.get('enable_ml_correction') else 'not_trained',
                        'mae': None,
                        'r2': None,
                        'training_samples': 0,
                        'last_update': None
                    }
            except Exception as e:
                logger.warning(f"Could not get ML performance data: {e}")
                ml_performance = {
                    'status': 'error',
                    'mae': None,
                    'r2': None,
                    'training_samples': 0,
                    'last_update': None
                }

            # Get thermal model quality metrics
            thermal_metrics = {}
            try:
                thermal_metrics = app.homeforecast.thermal_model.get_model_quality_metrics()
            except Exception as e:
                logger.warning(f"Could not get thermal model metrics: {e}")
                thermal_metrics = {
                    'mae': None,
                    'sample_size': 0
                }

            # Get system information
            import sys
            import platform
            system_info = {
                'addon_version': '2.0.3',
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': platform.system(),
                'log_level': logging.getLogger().getEffectiveLevel()
            }

            # Format current local time
            current_local_time = "N/A"
            timezone_info = {}
            try:
                if hasattr(app.homeforecast, 'ha_client'):
                    current_local_time = app.homeforecast.ha_client.format_time_for_display(datetime.now())
                else:
                    current_local_time = format_time_consistent(datetime.now())
                    
                # Add detailed timezone debugging information
                import os, time
                timezone_info = {
                    'system_timezone': getattr(app.homeforecast, 'timezone', 'UTC'),
                    'container_tz_env': os.environ.get('TZ', 'Not set'),
                    'system_utc_offset': time.timezone / 3600,
                    'system_dst': time.daylight,
                    'current_utc_time': datetime.utcnow().isoformat(),
                    'current_local_time_raw': datetime.now().isoformat(),
                    'ha_timezone': getattr(app.homeforecast, 'ha_timezone', 'Not set') if hasattr(app.homeforecast, 'ha_timezone') else 'Not set'
                }
                
                # Add timezone from HA client if available
                if hasattr(app.homeforecast, 'ha_client') and hasattr(app.homeforecast.ha_client, 'timezone'):
                    timezone_info['ha_client_timezone'] = str(app.homeforecast.ha_client.timezone)
                    
            except Exception as e:
                logger.warning(f"Could not format current time: {e}")
                timezone_info['error'] = str(e)

            response_data = {
                'status': 'running',
                'version': '2.0.3',
                'last_update': app.homeforecast.thermal_model.last_update.isoformat() if app.homeforecast.thermal_model.last_update else None,
                'last_update_display': last_update_str,
                'timezone': getattr(app.homeforecast, 'timezone', 'UTC'),
                'timezone_debug': timezone_info,
                'current_time': current_local_time,
                'current_data': current_data,
                'thermostat_data': thermostat_data,
                'climate_insights': climate_insights,
                'model_parameters': app.homeforecast.thermal_model.get_parameters(),
                'thermal_metrics': thermal_metrics,
                'ml_performance': ml_performance,
                'system_info': system_info,
                'config': {
                    'comfort_min': app.homeforecast.config.get('comfort_min_temp'),
                    'comfort_max': app.homeforecast.config.get('comfort_max_temp'),
                    'update_interval': app.homeforecast.config.get('update_interval_minutes'),
                    'ml_enabled': app.homeforecast.config.get('enable_ml_correction'),
                    'smart_hvac_enabled': app.homeforecast.config.is_smart_hvac_enabled()
                }
            }
            
            # Convert all data to JSON-serializable types
            try:
                response_data = convert_numpy_types(response_data)
                logger.info(f"✅ API: Data converted for JSON serialization")
            except Exception as conv_error:
                logger.warning(f"⚠️ Data conversion warning: {conv_error}")
            
            logger.info(f"✅ API: Returning status response with {len(response_data)} keys")
            return jsonify(response_data)
        except Exception as e:
            logger.error(f"Error in get_status: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/forecast/latest')
    def get_latest_forecast():
        """Get the most recent forecast"""
        try:
            logger.info("🌐 API: Processing /api/forecast/latest request")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            forecast = loop.run_until_complete(
                app.homeforecast.data_store.get_latest_forecast()
            )
            
            if forecast:
                logger.info(f"Retrieved forecast with keys: {list(forecast.keys())}")
                if 'data' in forecast:
                    forecast_data = forecast['data']
                    logger.info(f"Forecast data keys: {list(forecast_data.keys())}")
                    
                    # 🔍 DEBUG: Check for enhanced data structure
                    has_enhanced = 'historical_data' in forecast_data and 'forecast_data' in forecast_data
                    logger.info(f"🔍 Enhanced structure check: historical_data={bool(forecast_data.get('historical_data'))}, forecast_data={bool(forecast_data.get('forecast_data'))}, has_enhanced={has_enhanced}")
                    
                    # Add chart-ready data for frontend
                    if 'timestamps' in forecast_data and 'indoor_forecast' in forecast_data and 'outdoor_forecast' in forecast_data:
                        chart_data = {
                            'timestamps': forecast_data['timestamps'],
                            'indoor_temps': forecast_data['indoor_forecast'],
                            'outdoor_temps': forecast_data['outdoor_forecast'],
                            'idle_trajectory': forecast_data.get('idle_trajectory', []),
                            'controlled_trajectory': forecast_data.get('controlled_trajectory', []),
                            'historical_weather': forecast_data.get('historical_weather', [])
                        }
                        forecast['chart_data'] = chart_data
                        logger.info(f"Added chart data with {len(chart_data['timestamps'])} points and {len(chart_data['historical_weather'])} historical points")
                
                # Add new separated data structure for enhanced chart
                if 'historical_data' in forecast_data and 'forecast_data' in forecast_data:
                    # Use the new enhanced structure  
                    logger.info(f"✅ Enhanced chart data structure FOUND in forecast_data")
                    logger.info(f"Historical data keys: {list(forecast_data['historical_data'].keys()) if forecast_data['historical_data'] else 'None'}")
                    logger.info(f"Forecast data keys: {list(forecast_data['forecast_data'].keys()) if forecast_data['forecast_data'] else 'None'}")
                    logger.info(f"Historical timestamps: {len(forecast_data['historical_data'].get('timestamps', []))}")
                    logger.info(f"Forecast timestamps: {len(forecast_data['forecast_data'].get('timestamps', []))}")
                elif 'controlled_trajectory' in forecast_data and 'idle_trajectory' in forecast_data:
                    # Create enhanced structure from legacy data
                    logger.info("🔄 Creating enhanced chart structure from legacy data")
                    
                    # Determine current time index for separation
                    current_time_index = forecast_data.get('current_time_index', 0)
                    if current_time_index is None:
                        current_time_index = 0
                    
                    logger.info(f"Using current_time_index: {current_time_index}")
                    
                    # Create historical data (if any)
                    historical_data = {}
                    if current_time_index > 0:
                        timestamps = forecast_data.get('timestamps', [])
                        controlled_traj = forecast_data.get('controlled_trajectory', [])
                        
                        historical_data = {
                            'timestamps': timestamps[:current_time_index],
                            'actual_outdoor_temp': [step.get('outdoor_temp', 70) for step in forecast_data.get('outdoor_series', [])[:current_time_index]],
                            'actual_indoor_temp': [step.get('indoor_temp', 70) for step in controlled_traj[:current_time_index]],
                            'actual_hvac_mode': [step.get('hvac_state', 'off') for step in controlled_traj[:current_time_index]]
                        }
                        logger.info(f"Created historical data with {len(historical_data['timestamps'])} points")
                    
                    # Create forecast data
                    timestamps = forecast_data.get('timestamps', [])
                    controlled_traj = forecast_data.get('controlled_trajectory', [])
                    idle_traj = forecast_data.get('idle_trajectory', [])
                    
                    forecast_data_section = {
                        'timestamps': timestamps[current_time_index:],
                        'forecasted_outdoor_temp': forecast_data.get('outdoor_forecast', [])[current_time_index:],
                        'projected_indoor_with_hvac': [step.get('indoor_temp', 70) for step in controlled_traj[current_time_index:]],
                        'projected_indoor_no_hvac': [step.get('indoor_temp', 70) for step in idle_traj[current_time_index:]],
                        'projected_hvac_mode': [step.get('hvac_state', 'off') for step in controlled_traj[current_time_index:]]
                    }
                    logger.info(f"Created forecast data with {len(forecast_data_section['timestamps'])} points")
                    
                    # Add to forecast response
                    forecast_data['historical_data'] = historical_data
                    forecast_data['forecast_data'] = forecast_data_section
                    
                    # Add timeline separator
                    forecast_data['timeline_separator'] = {
                        'historical_end_index': current_time_index - 1 if current_time_index > 0 else 0,
                        'forecast_start_index': current_time_index,
                        'separator_timestamp': timestamps[current_time_index] if current_time_index < len(timestamps) else None,
                        'separator_label': 'Current Time - Forecast Begins'
                    }
                    logger.info(f"Added timeline separator at index {current_time_index}")
                    
                    logger.info("✅ Enhanced chart structure created from legacy data")
                
                # Add timezone debugging information to the forecast response
                try:
                    import time, os
                    from datetime import datetime
                    
                    timezone_debug = {
                        'container_tz': os.environ.get('TZ', 'Not set'),
                        'system_timezone': time.tzname,
                        'utc_offset_hours': -time.timezone / 3600,
                        'current_utc_time': datetime.utcnow().isoformat() + 'Z',
                        'current_local_time': datetime.now().isoformat(),
                        'forecast_generated_at': forecast.get('timestamp', 'Unknown')
                    }
                    
                    # Check if we have timestamps in the forecast data
                    if 'timestamps' in forecast_data and forecast_data['timestamps']:
                        first_ts = forecast_data['timestamps'][0]
                        last_ts = forecast_data['timestamps'][-1]
                        current_index = forecast_data.get('current_time_index', 0)
                        current_ts = forecast_data['timestamps'][current_index] if current_index < len(forecast_data['timestamps']) else None
                        
                        timezone_debug['forecast_timestamps'] = {
                            'first_timestamp': str(first_ts),
                            'last_timestamp': str(last_ts), 
                            'current_timestamp': str(current_ts) if current_ts else None,
                            'current_time_index': current_index,
                            'total_timestamps': len(forecast_data['timestamps']),
                            'first_timestamp_type': str(type(first_ts))
                        }
                        
                        # Check timezone awareness
                        if hasattr(first_ts, 'tzinfo'):
                            timezone_debug['forecast_timestamps']['timezone_aware'] = first_ts.tzinfo is not None
                            timezone_debug['forecast_timestamps']['tzinfo'] = str(first_ts.tzinfo) if first_ts.tzinfo else None
                    
                    forecast['timezone_debug'] = timezone_debug
                    logger.info(f"Added timezone debug info to forecast response")
                    
                except Exception as e:
                    logger.warning(f"Failed to add timezone debug info: {e}")
                
                # NOTE: Timestamps remain in original timezone-aware format (UTC+offset) 
                # Let the frontend handle timezone conversion based on browser timezone
                # This prevents double timezone conversion issues
                
                # Convert all data to JSON-serializable types
                forecast = convert_numpy_types(forecast)
                logger.info(f"✅ API: Returning forecast data")
                return jsonify(forecast)
            else:
                logger.warning("No forecast available yet")
                return jsonify({'message': 'No forecast available yet'}), 404
                
        except Exception as e:
            logger.error(f"Error getting forecast: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/debug/timezone')
    def debug_timezone():
        """Debug timezone configuration and time handling"""
        try:
            import os, time, platform
            from datetime import datetime, timezone
            
            debug_info = {
                'timestamp': datetime.now().isoformat(),
                'container_info': {
                    'tz_env_var': os.environ.get('TZ', 'Not set'),
                    'system_timezone': time.tzname,
                    'utc_offset_hours': -time.timezone / 3600,
                    'dst_active': time.daylight,
                    'platform': platform.system()
                },
                'datetime_info': {
                    'utc_now': datetime.utcnow().isoformat() + 'Z',
                    'local_now': datetime.now().isoformat(),
                    'local_with_tz': datetime.now().astimezone().isoformat()
                },
                'homeforecast_config': {
                    'system_timezone': getattr(app.homeforecast, 'timezone', 'Not configured'),
                    'ha_timezone': getattr(app.homeforecast, 'ha_timezone', 'Not configured') if hasattr(app.homeforecast, 'ha_timezone') else 'Not configured'
                }
            }
            
            # Add HA client timezone if available
            if hasattr(app.homeforecast, 'ha_client'):
                debug_info['ha_client'] = {
                    'has_timezone': hasattr(app.homeforecast.ha_client, 'timezone'),
                    'timezone_str': str(getattr(app.homeforecast.ha_client, 'timezone', 'Not set'))
                }
            else:
                debug_info['ha_client'] = {'available': False}
                
            # Test forecast data timestamp to see what timezone it's using
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                forecast = loop.run_until_complete(app.homeforecast.data_store.get_latest_forecast())
                if forecast and 'data' in forecast:
                    forecast_data = forecast['data']
                    if 'timestamps' in forecast_data and forecast_data['timestamps']:
                        first_timestamp = forecast_data['timestamps'][0]
                        debug_info['forecast_sample'] = {
                            'first_timestamp': str(first_timestamp),
                            'first_timestamp_type': str(type(first_timestamp)),
                            'current_time_index': forecast_data.get('current_time_index', 'Not set')
                        }
                        
                        # Check if timestamps are timezone aware
                        if hasattr(first_timestamp, 'tzinfo'):
                            debug_info['forecast_sample']['timezone_aware'] = first_timestamp.tzinfo is not None
                            debug_info['forecast_sample']['tzinfo'] = str(first_timestamp.tzinfo) if first_timestamp.tzinfo else None
                        
                loop.close()
            except Exception as e:
                debug_info['forecast_sample'] = {'error': str(e)}
                
            return jsonify(debug_info)
            
        except Exception as e:
            logger.error(f"Error in timezone debug: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/measurements/recent')
    def get_recent_measurements():
        """Get recent sensor measurements"""
        try:
            hours = request.args.get('hours', 24, type=int)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            measurements = loop.run_until_complete(
                app.homeforecast.data_store.get_recent_measurements(hours)
            )
            
            response_data = {
                'measurements': measurements,
                'count': len(measurements),
                'hours': hours
            }
            
            # Convert all data to JSON-serializable types
            response_data = convert_numpy_types(response_data)
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error getting measurements: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/model/parameters')
    def get_model_parameters():
        """Get current thermal model parameters"""
        try:
            logger.info("🌐 API: Processing /api/model/parameters request")
            
            params = app.homeforecast.thermal_model.get_parameters()
            quality = app.homeforecast.thermal_model.get_model_quality_metrics()
            
            logger.info(f"Raw model parameters: {params}")
            logger.info(f"Model quality metrics: {quality}")
            
            # Convert numpy types for JSON serialization
            params_cleaned = convert_numpy_types(params)
            quality_cleaned = convert_numpy_types(quality) if quality else {}
            
            logger.info(f"Cleaned parameters: {params_cleaned}")
            
            ml_info = None
            if app.homeforecast.thermal_model.ml_corrector:
                ml_info = app.homeforecast.thermal_model.ml_corrector.get_model_info()
                ml_info = convert_numpy_types(ml_info) if ml_info else None
                
            response_data = {
                'thermal_model': params_cleaned,
                'model_quality': quality_cleaned,
                'ml_correction': ml_info
            }
            
            logger.info(f"✅ API: Returning model parameters: {response_data}")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error getting model parameters: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/comfort/analysis')
    def get_comfort_analysis():
        """Get latest comfort analysis"""
        try:
            logger.info("🌐 API: Processing /api/comfort/analysis request")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get latest forecast
            forecast = loop.run_until_complete(
                app.homeforecast.data_store.get_latest_forecast()
            )
            
            if not forecast:
                logger.warning("No forecast available for comfort analysis")
                return jsonify({'message': 'No analysis available yet'}), 404
                
            logger.info(f"Retrieved forecast for analysis with keys: {list(forecast.get('data', {}).keys())}")
                
            # Run comfort analysis
            analysis = loop.run_until_complete(
                app.homeforecast.comfort_analyzer.analyze(forecast['data'])
            )
            
            logger.info(f"Comfort analysis result keys: {list(analysis.keys())}")
            
            # Convert numpy types for JSON serialization
            analysis_cleaned = convert_numpy_types(analysis)
            logger.info(f"✅ API: Returning comfort analysis: {analysis_cleaned}")
            
            return jsonify(analysis_cleaned)
            
        except Exception as e:
            logger.error(f"Error getting comfort analysis: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/trigger/update')
    def trigger_update():
        """Manually trigger an update cycle"""
        try:
            # Run update in background
            def run_update():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app.homeforecast.update_cycle())
                
            thread = threading.Thread(target=run_update)
            thread.start()
            
            return jsonify({
                'message': 'Update triggered',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error triggering update: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/statistics')
    def get_statistics():
        """Get database and model statistics"""
        try:
            logger.info("🌐 API: Processing /api/statistics request")
            
            db_stats = app.homeforecast.data_store.get_statistics()
            logger.info(f"Database statistics: {db_stats}")
            
            model_stats = {
                'parameter_history_count': len(app.homeforecast.thermal_model.parameter_history),
                'last_update': app.homeforecast.thermal_model.last_update.isoformat() if app.homeforecast.thermal_model.last_update else None
            }
            logger.info(f"Model statistics: {model_stats}")
            
            response_data = {
                'database': db_stats,
                'model': model_stats
            }
            
            # Convert all data to JSON-serializable types
            response_data = convert_numpy_types(response_data)
            
            logger.info(f"✅ API: Returning statistics: {response_data}")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/model/reset', methods=['POST'])
    def reset_model():
        """Reset the thermal model and clear historical data"""
        try:
            logger.info("🌐 API: Processing /api/model/reset request")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Clear all historical data
            loop.run_until_complete(app.homeforecast.data_store.clear_all_data())
            logger.info("✅ Historical data cleared")
            
            # Reset thermal model
            app.homeforecast.thermal_model.reset_model()
            logger.info("✅ Thermal model reset")
            
            # Trigger a full update cycle to restart data collection
            loop.run_until_complete(app.homeforecast.update_cycle())
            logger.info("✅ Full update cycle completed")
            
            logger.info("✅ API: Model reset completed successfully")
            return jsonify({'success': True, 'message': 'Model reset and data collection restarted'})
            
        except Exception as e:
            logger.error(f"Error resetting model: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
            # Clear database tables
            loop.run_until_complete(app.homeforecast.data_store.clear_all_data())
            
            # Reset thermal model
            app.homeforecast.thermal_model.reset_model()
            
            # Reset ML corrector if available
            if (hasattr(app.homeforecast.thermal_model, 'ml_corrector') and 
                app.homeforecast.thermal_model.ml_corrector):
                app.homeforecast.thermal_model.ml_corrector.reset_model()
            
            logger.info("✅ API: Model reset complete")
            return jsonify({
                'success': True,
                'message': 'Model and historical data have been reset'
            })
            
        except Exception as e:
            logger.error(f"Error resetting model: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ml/training-info')
    def get_ml_training_info():
        """Get ML training information for confirmation dialog"""
        try:
            ml_corrector = getattr(app.homeforecast.thermal_model, 'ml_corrector', None)
            
            # Get available training data count
            data_points = 0
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    training_data = loop.run_until_complete(
                        app.homeforecast.data_store.get_training_data(30)
                    )
                    data_points = len(training_data) if training_data else 0
                finally:
                    loop.close()
            except Exception as e:
                logger.warning(f"Could not fetch training data count: {e}")
                data_points = 0
            
            # Estimate duration based on data size
            if data_points < 100:
                estimated_duration = "Insufficient data (need 100+ points)"
                training_possible = False
            elif data_points < 1000:
                estimated_duration = "5-10 seconds"
                training_possible = True
            else:
                estimated_duration = "10-30 seconds"
                training_possible = True
            
            # Get sample of available columns for debugging
            sample_columns = []
            if data_points > 0:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        sample_data = loop.run_until_complete(
                            app.homeforecast.data_store.get_training_data(1)  # Just 1 day for column inspection
                        )
                        if sample_data and len(sample_data) > 0:
                            sample_columns = list(sample_data[0].keys())
                    finally:
                        loop.close()
                except Exception as e:
                    logger.debug(f"Could not get sample columns: {e}")
            
            return jsonify({
                'model_type': app.homeforecast.config.get('ml_model_type', 'Random Forest'),
                'data_points': data_points,
                'training_period': f"{app.homeforecast.config.get('ml_training_days', 30)} days",
                'estimated_duration': estimated_duration,
                'training_possible': training_possible,
                'current_status': 'Trained' if (ml_corrector and ml_corrector.is_trained) else 'Not Trained',
                'available_columns': sample_columns[:10] if sample_columns else []  # First 10 columns for debugging
            })
            
        except Exception as e:
            logger.error(f"Error getting ML training info: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ml/train', methods=['POST'])
    def train_ml_model():
        """Manually trigger ML model training"""
        try:
            logger.info("🎯 Manual ML model training requested via web UI")
            
            # Check if ML correction is enabled
            if not app.homeforecast.config.get('enable_ml_correction', False):
                return jsonify({
                    'success': False,
                    'error': 'ML correction is disabled in configuration'
                }), 400
            
            # Check if ML corrector exists
            ml_corrector = getattr(app.homeforecast.thermal_model, 'ml_corrector', None)
            if not ml_corrector:
                return jsonify({
                    'success': False,
                    'error': 'ML corrector not available'
                }), 400
            
            # Trigger training
            logger.info("🚀 Starting manual ML model training...")
            import asyncio
            
            # Run the training
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(ml_corrector.retrain())
            finally:
                loop.close()
            
            # Check if training was successful
            if ml_corrector.is_trained and ml_corrector.performance_history:
                latest_performance = ml_corrector.performance_history[-1]
                logger.info("✅ Manual ML model training completed successfully")
                
                return jsonify({
                    'success': True,
                    'message': 'ML model trained successfully',
                    'metrics': {
                        'r2': latest_performance.get('r2'),
                        'mse': latest_performance.get('mse'),
                        'mae': latest_performance.get('mae'),
                        'training_samples': latest_performance.get('training_samples'),
                        'training_duration': latest_performance.get('training_duration')
                    }
                })
            else:
                # Provide more specific error messages based on the state
                if not ml_corrector.is_trained:
                    error_msg = 'Training failed - insufficient data quality. '
                    
                    # Get the actual data count to help diagnose
                    try:
                        loop = asyncio.new_event_loop() 
                        asyncio.set_event_loop(loop)
                        try:
                            training_data = loop.run_until_complete(
                                app.homeforecast.data_store.get_training_data(30)
                            )
                            data_count = len(training_data) if training_data else 0
                            if data_count < 100:
                                error_msg += f'Only {data_count} data points available (need 100+).'
                            else:
                                error_msg += 'Data preparation failed - check logs for feature issues.'
                        finally:
                            loop.close()
                    except:
                        error_msg += 'Unable to assess data quality.'
                else:
                    error_msg = 'Training completed but performance history is missing.'
                
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 500
                
        except Exception as e:
            logger.error(f"❌ Error training ML model: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Training failed: {str(e)}'
            }), 500
            
    # Web UI Routes
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('dashboard.html')
        
    @app.route('/static/<path:path>')
    def send_static(path):
        """Serve static files"""
        return send_from_directory('static', path)
        
    @app.route('/api/logs')
    def get_logs():
        """Get recent log entries"""
        try:
            # In Home Assistant addon containers, logs are typically not accessible
            # from within the container. Direct users to the supervisor logs instead.
            help_message = """
HomeForecast Addon Logs

For full logs, please use one of these methods:

1. Home Assistant UI:
   - Go to Settings > Add-ons
   - Click on HomeForecast
   - Click the "Log" tab

2. Home Assistant CLI:
   - ha addons logs homeforecast-local

3. Direct supervisor logs:
   - docker logs addon_homeforecast-local

Recent console output from this session is not accessible 
from within the addon container for security reasons.

Check the Home Assistant supervisor logs for detailed 
information about HomeForecast operation.
            """.strip()
            
            return jsonify({
                'logs': help_message,
                'lines': help_message.split('\n'),
                'redirect_message': 'Use Home Assistant > Settings > Add-ons > HomeForecast > Log tab for full logs'
            })
                
        except Exception as e:
            return jsonify({'error': f'Error: {str(e)}'}), 500
    
    @app.route('/api/health')
    def get_system_health():
        """Get comprehensive system health and data quality information"""
        try:
            logger.info("🌐 API: Processing /api/health request")
            
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'unknown',
                'sensor_status': {},
                'weather_status': {},
                'system_metrics': {},
                'data_quality': {},
                'recommendations': []
            }
            
            # Check sensor connectivity and data freshness
            sensor_health = check_sensor_health(app.homeforecast)
            health_data['sensor_status'] = sensor_health
            
            # Check weather data availability
            weather_health = check_weather_health(app.homeforecast)
            health_data['weather_status'] = weather_health
            
            # Get system performance metrics
            system_metrics = get_system_metrics(app.homeforecast)
            health_data['system_metrics'] = system_metrics
            
            # Analyze data quality
            data_quality = analyze_data_quality(app.homeforecast)
            health_data['data_quality'] = data_quality
            
            # Generate health recommendations
            recommendations = generate_health_recommendations(health_data)
            health_data['recommendations'] = recommendations
            
            # Calculate overall health status
            health_data['overall_status'] = calculate_overall_health(health_data)
            
            return jsonify(health_data)
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return jsonify({
                'error': str(e),
                'overall_status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/logs')
    def logs_page():
        """Logs viewer page"""
        return render_template('logs.html')
    
    @app.route('/v2/building-model-manager')
    def building_model_manager():
        """V2.0 Building Model Manager page"""
        return render_template('building_model_manager.html')
    
    # V2.0 API Test Endpoint
    @app.route('/api/v2/test', methods=['GET'])
    def test_v2_api():
        """Test endpoint to verify v2.0 API is working"""
        training_available = HAS_ENHANCED_TRAINING or (not HAS_ENHANCED_TRAINING and 'HAS_SIMPLE_TRAINING' in globals() and HAS_SIMPLE_TRAINING)
        return jsonify({
            'success': True,
            'message': 'V2.0 API is working',
            'version': '2.0.8',
            'enhanced_training_available': HAS_ENHANCED_TRAINING,
            'simple_training_available': globals().get('HAS_SIMPLE_TRAINING', False),
            'training_available': training_available,
            'training_system': 'Enhanced' if HAS_ENHANCED_TRAINING else ('Simple' if globals().get('HAS_SIMPLE_TRAINING', False) else 'None'),
            'enhanced_training_error': ENHANCED_TRAINING_ERROR if not HAS_ENHANCED_TRAINING else None,
            'timestamp': datetime.now().isoformat()
        })
    
    # V2.0 Building Model and Weather File Upload Endpoints
    @app.route('/api/v2/building-model/upload', methods=['POST'])
    def upload_building_model():
        """Upload and parse DOE IDF building model file"""
        try:
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No file provided'
                }), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
                
            if not file.filename.lower().endswith('.idf'):
                return jsonify({
                    'success': False,
                    'error': 'File must be a .idf file'
                }), 400
                
            # Check file size (IDF files should typically be under 10MB)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            # Convert to MB for user-friendly messages
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > 25:  # 25MB limit for IDF files
                return jsonify({
                    'success': False,
                    'error': f'Building model file is too large ({file_size_mb:.1f}MB). IDF files are typically under 10MB.',
                    'details': 'Please verify this is a valid IDF building model file.'
                }), 413
            
            logger.info(f"🏠 Uploading IDF file: {file.filename} ({file_size_mb:.1f}MB)")
                
            # Check if any training system is available
            if not TrainingSystemClass:
                error_msg = f'No training system available: {ENHANCED_TRAINING_ERROR or "Unknown import error"}'
                logger.error(error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'details': 'Both enhanced and simple training systems failed to import. Please check server logs.'
                }), 500
                
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.idf', delete=False) as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
                
            try:
                # Parse building model using available training system
                training_system = TrainingSystemClass(homeforecast_instance.thermal_model)
                building_model = training_system.load_building_model(tmp_path)
                
                # Store building model in homeforecast instance
                homeforecast_instance.building_model = building_model
                
                logger.info(f"✅ Building model uploaded and parsed: {building_model['building_type']}")
                
                return jsonify({
                    'success': True,
                    'building_model': {
                        'building_type': building_model.get('building_type', 'Unknown'),
                        'floor_area_sqft': building_model.get('geometry', {}).get('floor_area_sqft', 2000),
                        'time_constant_hours': building_model.get('rc_parameters', {}).get('time_constant_hours', 8.0),
                        'thermal_properties': {
                            'material_count': building_model.get('thermal_properties', {}).get('material_count', 5),
                            'thermal_time_constant_hours': building_model.get('thermal_properties', {}).get('thermal_time_constant_hours', 8.0)
                        },
                        'hvac_systems': building_model.get('hvac_systems', {'has_heating': True, 'has_cooling': True}),
                        'upload_timestamp': datetime.now().isoformat()
                    }
                })
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Error uploading building model: {e}")
            
            # Provide more helpful error messages
            error_msg = str(e)
            if "encoding" in error_msg.lower():
                error_msg = "File encoding issue. Please ensure your IDF file is saved in UTF-8 format."
            elif "empty" in error_msg.lower():
                error_msg = "The uploaded file appears to be empty. Please check your IDF file."
            elif "not a valid" in error_msg.lower():
                error_msg = "This doesn't appear to be a valid EnergyPlus IDF file. Please upload a proper .idf building model file."
            else:
                error_msg = f"Error parsing building model: {error_msg}. The system will use a default building model for training."
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': 'If you continue to have issues, the system can use a default building model for basic functionality.'
            }), 500
    
    @app.route('/api/v2/weather-dataset/upload', methods=['POST'])
    def upload_weather_dataset():
        """Upload and parse EPW weather dataset file"""
        try:
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No file provided'
                }), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
                
            if not file.filename.lower().endswith('.epw'):
                return jsonify({
                    'success': False,
                    'error': 'File must be a .epw file'
                }), 400
                
            # Get optional parameters
            limit_hours = request.form.get('limit_hours', type=int)
            
            # Check file size (EPW files should typically be 1-5MB)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            # Convert to MB for user-friendly messages
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > 20:  # 20MB limit for EPW files
                return jsonify({
                    'success': False,
                    'error': f'Weather dataset file is too large ({file_size_mb:.1f}MB). EPW files are typically 1-2MB.',
                    'details': 'Please verify this is a valid EPW weather file and not a different file type.'
                }), 413
            
            logger.info(f"📊 Uploading EPW file: {file.filename} ({file_size_mb:.1f}MB)")
            
            # Check if any training system is available
            if not TrainingSystemClass:
                error_msg = f'No training system available: {ENHANCED_TRAINING_ERROR or "Unknown import error"}'
                logger.error(error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'details': 'Both enhanced and simple training systems failed to import. Please check server logs.'
                }), 500
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.epw', delete=False) as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
                
            try:
                # Parse weather dataset using available training system
                training_system = TrainingSystemClass(homeforecast_instance.thermal_model)
                weather_dataset = training_system.load_weather_dataset(tmp_path, limit_hours)
                
                # Store weather dataset in homeforecast instance
                homeforecast_instance.weather_dataset = weather_dataset
                
                logger.info(f"✅ Weather dataset uploaded: {weather_dataset.get('location', {}).get('city', 'Unknown Location')}, {weather_dataset.get('data_points', 0)} hours")
                
                return jsonify({
                    'success': True,
                    'weather_dataset': {
                        'location': weather_dataset.get('location', {'city': 'Unknown', 'country': 'Unknown'}),
                        'data_points': weather_dataset.get('data_points', 0),
                        'summary_statistics': weather_dataset.get('summary_statistics', {}),
                        'upload_timestamp': datetime.now().isoformat()
                    }
                })
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Error uploading weather dataset: {e}")
            
            # Provide more helpful error messages for EPW files
            error_msg = str(e)
            if "file too large" in error_msg.lower() or "413" in error_msg:
                error_msg = "Weather dataset file is too large. EPW files are typically 1-2MB. Please check if this is a valid EPW file."
            elif "encoding" in error_msg.lower():
                error_msg = "File encoding issue. Please ensure your EPW file is saved in UTF-8 format."
            elif "empty" in error_msg.lower():
                error_msg = "The uploaded file appears to be empty. Please check your EPW file."
            elif "not a valid" in error_msg.lower() or "header" in error_msg.lower():
                error_msg = "This doesn't appear to be a valid EnergyPlus EPW weather file. Please upload a proper .epw weather dataset."
            else:
                error_msg = f"Error parsing weather dataset: {error_msg}. Please check that you've uploaded a valid EPW weather file."
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': 'EPW files should be from EnergyPlus weather datasets and typically range from 1-2MB in size.'
            }), 500
    
    @app.route('/api/v2/training/run', methods=['POST'])
    def run_enhanced_training():
        """Run enhanced training using uploaded building model and weather data"""
        try:
            # Check prerequisites
            if not hasattr(homeforecast_instance, 'building_model') or not homeforecast_instance.building_model:
                return jsonify({
                    'success': False,
                    'error': 'Building model must be uploaded first'
                }), 400
                
            if not hasattr(homeforecast_instance, 'weather_dataset') or not homeforecast_instance.weather_dataset:
                return jsonify({
                    'success': False,
                    'error': 'Weather dataset must be uploaded first'  
                }), 400
                
            # Get training parameters
            data = request.get_json() or {}
            training_duration_hours = data.get('training_duration_hours', 168)  # 1 week default
            hvac_scenarios = data.get('hvac_scenarios', ['heating', 'cooling', 'off', 'mixed'])
            comfort_min = data.get('comfort_min', 68.0)
            comfort_max = data.get('comfort_max', 76.0)
            
            # Check if any training system is available
            if not TrainingSystemClass:
                error_msg = f'No training system available: {ENHANCED_TRAINING_ERROR or "Unknown import error"}'
                logger.error(error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'details': 'Both enhanced and simple training systems failed to import. Please check server logs.'
                }), 500
            
            # Run training using available system
            training_system = TrainingSystemClass(homeforecast_instance.thermal_model)
            training_system.building_model = homeforecast_instance.building_model
            training_system.weather_dataset = homeforecast_instance.weather_dataset
            
            logger.info(f"🎯 Starting enhanced training - Duration: {training_duration_hours}h, Scenarios: {hvac_scenarios}")
            
            training_results = training_system.run_enhanced_training(
                training_duration_hours=training_duration_hours,
                hvac_scenarios=hvac_scenarios,
                comfort_min=comfort_min,
                comfort_max=comfort_max
            )
            
            # Store training results
            homeforecast_instance.training_results = training_results
            
            logger.info(f"✅ Enhanced training completed - Accuracy: {training_results.get('accuracy_score', 0):.3f}")
            
            return jsonify({
                'success': True,
                'training_results': training_results,
                'training_timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error running enhanced training: {e}")
            return jsonify({
                'success': False,
                'error': f'Error running enhanced training: {str(e)}'
            }), 500
    
    @app.route('/api/v2/model/status')
    def get_v2_model_status():
        """Get comprehensive v2.0 model status including building model and training state"""
        try:
            building_model_loaded = hasattr(homeforecast_instance, 'building_model') and homeforecast_instance.building_model
            weather_dataset_loaded = hasattr(homeforecast_instance, 'weather_dataset') and homeforecast_instance.weather_dataset
            training_completed = hasattr(homeforecast_instance, 'training_results') and homeforecast_instance.training_results
            
            status = {
                'version': '2.0.3',
                'building_model_loaded': building_model_loaded,
                'weather_dataset_loaded': weather_dataset_loaded,
                'training_completed': training_completed,
                'system_ready': building_model_loaded and weather_dataset_loaded and training_completed
            }
            
            if building_model_loaded:
                bm = homeforecast_instance.building_model
                status['building_model'] = {
                    'building_type': bm['building_type'],
                    'floor_area_sqft': bm['geometry']['floor_area_sqft'],
                    'time_constant_hours': bm['rc_parameters']['time_constant_hours']
                }
                
            if weather_dataset_loaded:
                wd = homeforecast_instance.weather_dataset
                status['weather_dataset'] = {
                    'location': wd['location']['city'],
                    'data_points': wd['data_points']
                }
                
            if training_completed:
                tr = homeforecast_instance.training_results
                status['training_results'] = {
                    'accuracy_score': tr.get('accuracy_score', 0),
                    'physics_compliance': tr.get('physics_compliance', 0),
                    'total_samples': tr.get('total_samples', 0)
                }
                
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"❌ Error getting v2.0 model status: {e}")
            return jsonify({
                'version': '2.0.3',
                'error': f'Error getting model status: {str(e)}',
                'building_model_loaded': False,
                'weather_dataset_loaded': False,
                'training_completed': False,
                'system_ready': False
            }), 500
        
    return app


def check_sensor_health(homeforecast_instance):
    """Check the health and connectivity of all sensors"""
    sensor_status = {
        'connected_sensors': [],
        'disconnected_sensors': [],
        'stale_sensors': [],
        'sensor_quality': {}
    }
    
    try:
        # Get last sensor data if available
        last_data = getattr(homeforecast_instance, 'last_sensor_data', {})
        data_quality = last_data.get('data_quality', {})
        
        # Required sensors
        required_sensors = ['indoor_temp', 'indoor_humidity', 'hvac_thermostat']
        optional_sensors = ['outdoor_temp', 'outdoor_humidity', 'solar_irradiance']
        
        # Check required sensors
        missing_sensors = data_quality.get('missing_sensors', [])
        failed_sensors = data_quality.get('failed_sensors', [])
        
        for sensor in required_sensors:
            if sensor in missing_sensors:
                sensor_status['disconnected_sensors'].append(sensor)
                sensor_status['sensor_quality'][sensor] = 'missing'
            elif sensor in failed_sensors:
                sensor_status['disconnected_sensors'].append(sensor)
                sensor_status['sensor_quality'][sensor] = 'failed'
            else:
                sensor_status['connected_sensors'].append(sensor)
                sensor_status['sensor_quality'][sensor] = 'good'
        
        # Check optional sensors
        for sensor in optional_sensors:
            if sensor in missing_sensors:
                sensor_status['sensor_quality'][sensor] = 'not_configured'
            elif sensor in failed_sensors:
                sensor_status['sensor_quality'][sensor] = 'failed'
            elif last_data.get(sensor) is not None:
                sensor_status['connected_sensors'].append(sensor)
                sensor_status['sensor_quality'][sensor] = 'good'
            else:
                sensor_status['sensor_quality'][sensor] = 'not_configured'
                
    except Exception as e:
        logger.error(f"Error checking sensor health: {e}")
        sensor_status['error'] = str(e)
        
    return sensor_status


def check_weather_health(homeforecast_instance):
    """Check the health and availability of weather data sources"""
    weather_status = {
        'accuweather_available': False,
        'data_source': 'unknown',
        'last_successful_fetch': None,
        'api_issues': [],
        'forecast_coverage': 0
    }
    
    try:
        # Check if we have weather cache or recent data
        ha_client = getattr(homeforecast_instance, 'ha_client', None)
        if ha_client and hasattr(ha_client, '_weather_cache'):
            cache = ha_client._weather_cache
            if cache:
                cache_age = (datetime.now() - cache['timestamp']).total_seconds() / 3600
                weather_status['last_successful_fetch'] = cache['timestamp'].isoformat()
                
                # Check cache data quality
                cached_data = cache.get('data', {})
                quality = cached_data.get('data_quality', {})
                weather_status['data_source'] = quality.get('source', 'cached')
                weather_status['api_issues'] = quality.get('issues', [])
                
                # Check forecast coverage
                hourly_forecast = cached_data.get('hourly_forecast', [])
                weather_status['forecast_coverage'] = len(hourly_forecast)
                
                if cache_age < 2:  # Fresh data
                    weather_status['accuweather_available'] = quality.get('source') == 'accuweather'
                else:
                    weather_status['api_issues'].append('Stale weather cache')
        
        # Check API credentials
        config = getattr(homeforecast_instance, 'config', None)
        if config:
            api_key = config.get('accuweather_api_key')
            location_key = config.get('accuweather_location_key')
            
            if not api_key:
                weather_status['api_issues'].append('AccuWeather API key not configured')
            if not location_key:
                weather_status['api_issues'].append('AccuWeather location key not configured')
                
    except Exception as e:
        logger.error(f"Error checking weather health: {e}")
        weather_status['error'] = str(e)
        
    return weather_status


def get_system_metrics(homeforecast_instance):
    """Get system performance and operational metrics"""
    metrics = {
        'uptime_hours': 0,
        'total_updates': 0,
        'successful_updates': 0,
        'model_accuracy': None,
        'data_points_collected': 0,
        'ml_model_trained': False
    }
    
    try:
        # Calculate uptime
        start_time = getattr(homeforecast_instance, 'start_time', datetime.now())
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        metrics['uptime_hours'] = round(uptime_seconds / 3600, 1)
        
        # Get thermal model statistics
        thermal_model = getattr(homeforecast_instance, 'thermal_model', None)
        if thermal_model:
            params = thermal_model.get_parameters()
            metrics['total_updates'] = len(getattr(thermal_model, 'parameter_history', []))
            
            # Check ML model status
            ml_corrector = getattr(thermal_model, 'ml_corrector', None)
            if ml_corrector:
                metrics['ml_model_trained'] = getattr(ml_corrector, 'is_trained', False)
                performance = getattr(ml_corrector, 'performance_history', [])
                if performance:
                    latest_perf = performance[-1]
                    metrics['model_accuracy'] = round(latest_perf.get('accuracy', 0) * 100, 1)
        
        # Get data store statistics
        data_store = getattr(homeforecast_instance, 'data_store', None)
        if data_store:
            try:
                stats = data_store.get_statistics()
                metrics['data_points_collected'] = stats.get('total_measurements', 0)
            except Exception:
                pass  # Data store statistics optional
                
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        metrics['error'] = str(e)
        
    return metrics


def analyze_data_quality(homeforecast_instance):
    """Analyze overall data quality and consistency"""
    quality = {
        'overall_score': 0,
        'sensor_reliability': 0,
        'weather_reliability': 0,
        'data_freshness': 0,
        'issues_detected': []
    }
    
    try:
        # Analyze sensor data quality
        last_data = getattr(homeforecast_instance, 'last_sensor_data', {})
        data_quality = last_data.get('data_quality', {})
        
        missing_count = len(data_quality.get('missing_sensors', []))
        failed_count = len(data_quality.get('failed_sensors', []))
        warning_count = len(data_quality.get('warnings', []))
        
        # Calculate sensor reliability (0-100)
        total_sensors = 6  # Expected total sensors
        working_sensors = total_sensors - missing_count - failed_count
        quality['sensor_reliability'] = max(0, (working_sensors / total_sensors) * 100)
        
        # Check data freshness
        if last_data.get('timestamp'):
            data_age = (datetime.now() - last_data['timestamp']).total_seconds() / 60
            if data_age < 10:
                quality['data_freshness'] = 100
            elif data_age < 30:
                quality['data_freshness'] = 80
            elif data_age < 60:
                quality['data_freshness'] = 60
            else:
                quality['data_freshness'] = 30
                quality['issues_detected'].append(f'Sensor data is {data_age:.0f} minutes old')
        
        # Check weather data quality
        ha_client = getattr(homeforecast_instance, 'ha_client', None)
        if ha_client and hasattr(ha_client, '_weather_cache'):
            cache = ha_client._weather_cache
            if cache:
                cache_age_hours = (datetime.now() - cache['timestamp']).total_seconds() / 3600
                if cache_age_hours < 1:
                    quality['weather_reliability'] = 100
                elif cache_age_hours < 3:
                    quality['weather_reliability'] = 80
                elif cache_age_hours < 6:
                    quality['weather_reliability'] = 60
                else:
                    quality['weather_reliability'] = 30
                    quality['issues_detected'].append('Weather data is stale')
            else:
                quality['weather_reliability'] = 0
                quality['issues_detected'].append('No weather data available')
        else:
            quality['weather_reliability'] = 0
            quality['issues_detected'].append('Weather service not configured')
        
        # Calculate overall score
        quality['overall_score'] = round(
            (quality['sensor_reliability'] * 0.4 + 
             quality['weather_reliability'] * 0.3 + 
             quality['data_freshness'] * 0.3)
        )
        
        # Add specific issues
        if missing_count > 0:
            quality['issues_detected'].append(f'{missing_count} sensors not configured')
        if failed_count > 0:
            quality['issues_detected'].append(f'{failed_count} sensors failed')
        if warning_count > 0:
            quality['issues_detected'].append(f'{warning_count} data validation warnings')
            
    except Exception as e:
        logger.error(f"Error analyzing data quality: {e}")
        quality['error'] = str(e)
        
    return quality


def generate_health_recommendations(health_data):
    """Generate actionable recommendations based on health analysis"""
    recommendations = []
    
    try:
        sensor_status = health_data.get('sensor_status', {})
        weather_status = health_data.get('weather_status', {})
        data_quality = health_data.get('data_quality', {})
        
        # Sensor recommendations
        disconnected = sensor_status.get('disconnected_sensors', [])
        if 'indoor_temp' in disconnected:
            recommendations.append({
                'priority': 'high',
                'category': 'sensors',
                'message': 'Indoor temperature sensor is disconnected - check entity configuration',
                'action': 'Verify indoor_temp_entity setting in addon configuration'
            })
            
        if 'hvac_thermostat' in disconnected:
            recommendations.append({
                'priority': 'high', 
                'category': 'sensors',
                'message': 'HVAC thermostat is unreachable - check device connectivity',
                'action': 'Verify hvac_entity setting and thermostat network connection'
            })
            
        # Weather recommendations
        if not weather_status.get('accuweather_available', False):
            api_issues = weather_status.get('api_issues', [])
            if 'API key not configured' in str(api_issues):
                recommendations.append({
                    'priority': 'medium',
                    'category': 'weather',
                    'message': 'AccuWeather API not configured - using fallback weather data',
                    'action': 'Configure AccuWeather API credentials for accurate forecasts'
                })
            else:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'weather', 
                    'message': 'AccuWeather API experiencing issues - check API limits',
                    'action': 'Verify API key validity and check usage quotas'
                })
                
        # Data quality recommendations
        overall_score = data_quality.get('overall_score', 0)
        if overall_score < 70:
            recommendations.append({
                'priority': 'medium',
                'category': 'data_quality',
                'message': f'Data quality is suboptimal ({overall_score}%) - system accuracy may be reduced',
                'action': 'Address sensor connectivity and weather API issues'
            })
            
        # Performance recommendations
        metrics = health_data.get('system_metrics', {})
        if not metrics.get('ml_model_trained', False):
            uptime = metrics.get('uptime_hours', 0)
            if uptime > 24:
                recommendations.append({
                    'priority': 'low',
                    'category': 'performance',
                    'message': 'ML correction model not yet trained - accuracy will improve over time',
                    'action': 'Allow system to collect more data for automatic ML training'
                })
                
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        
    return recommendations


def calculate_overall_health(health_data):
    """Calculate overall system health status"""
    try:
        sensor_status = health_data.get('sensor_status', {})
        weather_status = health_data.get('weather_status', {})
        data_quality = health_data.get('data_quality', {})
        
        # Check for critical issues
        disconnected = sensor_status.get('disconnected_sensors', [])
        critical_sensors = ['indoor_temp', 'hvac_thermostat']
        
        if any(sensor in disconnected for sensor in critical_sensors):
            return 'critical'
            
        # Check overall data quality score
        quality_score = data_quality.get('overall_score', 0)
        
        if quality_score >= 80:
            return 'excellent'
        elif quality_score >= 60:
            return 'good' 
        elif quality_score >= 40:
            return 'fair'
        else:
            return 'poor'
            
    except Exception as e:
        logger.error(f"Error calculating overall health: {e}")
        return 'unknown'