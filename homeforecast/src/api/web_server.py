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


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif HAS_NUMPY and isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def calculate_climate_insights(current_data, thermostat_data, config, comfort_analyzer=None):
    """Calculate climate action insights for dashboard"""
    try:
        insights = {
            'recommended_action': 'NONE',
            'action_off_time': None,
            'next_action_time': None,
            'estimated_runtime': None
        }
        
        # Get current conditions
        current_temp = current_data.get('indoor_temp', 70.0)
        target_temp = thermostat_data.get('target_temperature', 72.0)
        hvac_mode = thermostat_data.get('hvac_mode', 'off')
        hvac_action = thermostat_data.get('hvac_action', 'idle')
        
        comfort_min = config.get('comfort_min_temp', 68.0)
        comfort_max = config.get('comfort_max_temp', 74.0)
        
        logger.info(f"Climate insights calculation - Current: {current_temp}°F, Target: {target_temp}°F, " +
                   f"Mode: {hvac_mode}, Action: {hvac_action}")
        
        # Determine recommended action based on current conditions
        if hvac_mode == 'off':
            insights['recommended_action'] = 'OFF'
        elif hvac_action in ['heating', 'heat']:
            insights['recommended_action'] = 'HEATING'
            # Estimate when heating will stop (when target temp is reached)
            if current_temp < target_temp:
                temp_diff = target_temp - current_temp
                heating_rate = 3.5  # °F/hour (default estimate)
                runtime_hours = temp_diff / heating_rate
                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                insights['action_off_time'] = (datetime.now() + 
                                             timedelta(hours=runtime_hours)).strftime("%H:%M")
        elif hvac_action in ['cooling', 'cool']:
            insights['recommended_action'] = 'COOLING'
            # Estimate when cooling will stop (when target temp is reached)
            if current_temp > target_temp:
                temp_diff = current_temp - target_temp
                cooling_rate = 4.0  # °F/hour (default estimate)
                runtime_hours = temp_diff / cooling_rate
                insights['estimated_runtime'] = f"{runtime_hours * 60:.0f} min"
                insights['action_off_time'] = (datetime.now() + 
                                             timedelta(hours=runtime_hours)).strftime("%H:%M")
        elif hvac_mode == 'heat' and current_temp < comfort_min:
            insights['recommended_action'] = 'HEAT'
            # Calculate when to start heating to maintain comfort
            temp_deficit = comfort_min - current_temp
            if temp_deficit > 1.0:  # Need heating soon
                insights['next_action_time'] = "Now"
            else:
                insights['next_action_time'] = "15 min"
        elif hvac_mode == 'cool' and current_temp > comfort_max:
            insights['recommended_action'] = 'COOL'
            # Calculate when to start cooling to maintain comfort
            temp_excess = current_temp - comfort_max
            if temp_excess > 1.0:  # Need cooling soon
                insights['next_action_time'] = "Now"
            else:
                insights['next_action_time'] = "15 min"
        else:
            insights['recommended_action'] = 'MONITOR'
        
        logger.info(f"Climate insights result: {insights}")
        return insights
        
    except Exception as e:
        logger.error(f"Error calculating climate insights: {e}")
        return {
            'recommended_action': 'UNKNOWN',
            'action_off_time': None,
            'next_action_time': None,
            'estimated_runtime': None
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

            response_data = {
                'status': 'running',
                'version': '1.3.0',
                'last_update': app.homeforecast.thermal_model.last_update.isoformat() if app.homeforecast.thermal_model.last_update else None,
                'current_data': current_data,
                'thermostat_data': thermostat_data,
                'climate_insights': climate_insights,
                'model_parameters': app.homeforecast.thermal_model.get_parameters(),
                'config': {
                    'comfort_min': app.homeforecast.config.get('comfort_min_temp'),
                    'comfort_max': app.homeforecast.config.get('comfort_max_temp'),
                    'update_interval': app.homeforecast.config.get('update_interval_minutes'),
                    'ml_enabled': app.homeforecast.config.get('enable_ml_correction'),
                    'smart_hvac_enabled': app.homeforecast.config.is_smart_hvac_enabled()
                }
            }
            
            logger.info(f"✅ API: Returning status response: {response_data}")
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
                            'controlled_trajectory': forecast_data.get('controlled_trajectory', [])
                        }
                        forecast['chart_data'] = chart_data
                        logger.info(f"Added chart data with {len(chart_data['timestamps'])} points")
                
                logger.info(f"✅ API: Returning forecast data")
                return jsonify(forecast)
            else:
                logger.warning("No forecast available yet")
                return jsonify({'message': 'No forecast available yet'}), 404
                
        except Exception as e:
            logger.error(f"Error getting forecast: {e}")
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
            
            return jsonify({
                'measurements': measurements,
                'count': len(measurements),
                'hours': hours
            })
            
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
            
            logger.info(f"✅ API: Returning statistics: {response_data}")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return jsonify({'error': str(e)}), 500
            
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
    
    @app.route('/logs')
    def logs_page():
        """Logs viewer page"""
        return render_template('logs.html')
        
    return app