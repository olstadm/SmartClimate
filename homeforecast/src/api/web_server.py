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
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


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
    
    try:
        now = datetime.now()
        current_hvac_on = False
        hvac_start_time = None
        hvac_end_time = None
        
        # Analyze trajectory for HVAC activity patterns
        for i, point in enumerate(trajectory):
            hvac_state = point.get('hvac_state', 'idle')
            timestamp = point.get('timestamp', now)
            
            # Check if HVAC is active (heating or cooling)
            is_active = hvac_state in ['heating', 'cooling', 'heat', 'cool']
            
            # Detect HVAC start
            if is_active and not current_hvac_on:
                hvac_start_time = timestamp
                current_hvac_on = True
                # Determine action type
                if hvac_state in ['cooling', 'cool']:
                    insights['recommended_action'] = 'COOL SOON'
                elif hvac_state in ['heating', 'heat']:
                    insights['recommended_action'] = 'HEAT SOON'
                    
            # Detect HVAC end
            elif not is_active and current_hvac_on:
                hvac_end_time = timestamp
                current_hvac_on = False
                
                # Calculate runtime
                if hvac_start_time:
                    runtime_hours = (hvac_end_time - hvac_start_time).total_seconds() / 3600
                    insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                    
                break  # Use first HVAC cycle found
        
        # Set times based on findings
        if hvac_start_time:
            insights['next_action_time'] = format_time_consistent(hvac_start_time)
            
        if hvac_end_time:
            insights['action_off_time'] = format_time_consistent(hvac_end_time)
        elif current_hvac_on and hvac_start_time:
            # Still running at end of forecast
            insights['action_off_time'] = "Beyond forecast"
            
    except Exception as e:
        logger.warning(f"Error analyzing HVAC schedule: {e}")
        
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
        
        # Try to get forecast-based insights from comfort analyzer
        forecast_insights = None
        if comfort_analyzer and hasattr(comfort_analyzer, 'homeforecast'):
            try:
                # Check if we have recent forecast data stored in comfort analyzer
                if hasattr(comfort_analyzer, 'latest_forecast_data') and comfort_analyzer.latest_forecast_data:
                    forecast_data = comfort_analyzer.latest_forecast_data
                    if 'controlled_trajectory' in forecast_data:
                        trajectory = forecast_data['controlled_trajectory']
                        forecast_insights = _analyze_hvac_schedule_from_trajectory(trajectory, current_temp, hvac_mode)
                        logger.debug(f"Found {len(trajectory)} trajectory points for insights")
            except Exception as e:
                logger.debug(f"Could not get forecast insights: {e}")
        
        # If we have forecast insights, use them
        if forecast_insights:
            insights.update(forecast_insights)
            logger.info(f"Using forecast-based climate insights: {insights['recommended_action']}")
        else:
            # Fallback to current state analysis
            logger.info("Using current state for climate insights")
        
        # HVAC performance estimates (°F/hour)
        heating_rate = 3.5
        cooling_rate = 4.0
        drift_rate = 0.5  # Natural temperature drift when HVAC is off
        
        logger.info(f"Climate insights calculation - Current: {current_temp}°F, Target: {target_temp}°F, " +
                   f"Mode: {hvac_mode}, Action: {hvac_action}, Comfort: {comfort_min}-{comfort_max}°F")
        
        now = datetime.now()
        
        # Context-aware insights: Only show relevant timing based on current HVAC state
        is_currently_heating = hvac_heating or hvac_action in ['heating', 'heat'] or (hvac_mode == 'heat' and hvac_action != 'idle')
        is_currently_cooling = hvac_cooling or hvac_action in ['cooling', 'cool'] or (hvac_mode == 'cool' and hvac_action != 'idle')
        
        logger.info(f"HVAC Context - Currently heating: {is_currently_heating}, Currently cooling: {is_currently_cooling}, " +
                   f"HVAC flags - Heat: {hvac_heating}, Cool: {hvac_cooling}, Action: {hvac_action}")
        
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
                    insights['recommended_action'] = 'OFF'
                    insights['action_off_time'] = 'N/A (Currently off)'
                    
                    # Predict when action will be needed based on setpoint and mode
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
                'addon_version': '1.8.16',
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
                'version': '1.8.15',
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
                    logger.info(f"✅ Enhanced chart data structure available")
                    logger.info(f"Historical data keys: {list(forecast_data['historical_data'].keys()) if forecast_data['historical_data'] else 'None'}")
                    logger.info(f"Forecast data keys: {list(forecast_data['forecast_data'].keys()) if forecast_data['forecast_data'] else 'None'}")
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