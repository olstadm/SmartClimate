"""
Web server and API for HomeForecast addon
Provides REST API and web interface for Home Assistant integration
"""
import logging
import json
from datetime import datetime
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import asyncio
import threading
import os

logger = logging.getLogger(__name__)


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
            # Get current sensor data for display
            current_data = {}
            try:
                # This will be synchronous, so we need to be careful about blocking
                # For now, we'll just get the model state which should have the latest data
                model_params = app.homeforecast.thermal_model.get_parameters()
                current_data = {
                    'indoor_temp': getattr(app.homeforecast.thermal_model, 'current_indoor_temp', None),
                    'outdoor_temp': getattr(app.homeforecast.thermal_model, 'current_outdoor_temp', None),
                    'hvac_state': getattr(app.homeforecast.thermal_model, 'current_hvac_state', 'unknown')
                }
            except Exception as e:
                logger.warning(f"Could not get current sensor data: {e}")
                current_data = {}
            
            return jsonify({
                'status': 'running',
                'version': '1.0.9',
                'last_update': app.homeforecast.thermal_model.last_update.isoformat() if app.homeforecast.thermal_model.last_update else None,
                'current_data': current_data,
                'model_parameters': app.homeforecast.thermal_model.get_parameters(),
                'config': {
                    'comfort_min': app.homeforecast.config.get('comfort_min_temp'),
                    'comfort_max': app.homeforecast.config.get('comfort_max_temp'),
                    'update_interval': app.homeforecast.config.get('update_interval_minutes'),
                    'ml_enabled': app.homeforecast.config.get('enable_ml_correction'),
                    'smart_hvac_enabled': app.homeforecast.config.is_smart_hvac_enabled()
                }
            })
        except Exception as e:
            logger.error(f"Error in get_status: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/forecast/latest')
    def get_latest_forecast():
        """Get the most recent forecast"""
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            forecast = loop.run_until_complete(
                app.homeforecast.data_store.get_latest_forecast()
            )
            
            if forecast:
                return jsonify(forecast)
            else:
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
            params = app.homeforecast.thermal_model.get_parameters()
            quality = app.homeforecast.thermal_model.get_model_quality_metrics()
            
            ml_info = None
            if app.homeforecast.thermal_model.ml_corrector:
                ml_info = app.homeforecast.thermal_model.ml_corrector.get_model_info()
                
            return jsonify({
                'thermal_model': params,
                'model_quality': quality,
                'ml_correction': ml_info
            })
            
        except Exception as e:
            logger.error(f"Error getting model parameters: {e}")
            return jsonify({'error': str(e)}), 500
            
    @app.route('/api/comfort/analysis')
    def get_comfort_analysis():
        """Get latest comfort analysis"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get latest forecast
            forecast = loop.run_until_complete(
                app.homeforecast.data_store.get_latest_forecast()
            )
            
            if not forecast:
                return jsonify({'message': 'No analysis available yet'}), 404
                
            # Run comfort analysis
            analysis = loop.run_until_complete(
                app.homeforecast.comfort_analyzer.analyze(forecast['data'])
            )
            
            return jsonify(analysis)
            
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
            db_stats = app.homeforecast.data_store.get_statistics()
            
            return jsonify({
                'database': db_stats,
                'model': {
                    'parameter_history_count': len(app.homeforecast.thermal_model.parameter_history),
                    'last_update': app.homeforecast.thermal_model.last_update.isoformat() if app.homeforecast.thermal_model.last_update else None
                }
            })
            
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
            log_file = '/proc/1/fd/1'  # Docker container stdout
            if not os.path.exists(log_file):
                # Fallback to supervisor logs
                log_file = '/proc/self/fd/1'
            
            # Read last 100 lines of logs
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    return jsonify({
                        'logs': ''.join(recent_lines),
                        'lines': recent_lines
                    })
            except:
                # If can't read logs, return a helpful message
                return jsonify({
                    'logs': 'Log file not accessible. Check Home Assistant Supervisor logs for HomeForecast addon.',
                    'lines': []
                })
                
        except Exception as e:
            return jsonify({'error': f'Error reading logs: {str(e)}'}), 500
    
    @app.route('/logs')
    def logs_page():
        """Logs viewer page"""
        return render_template('logs.html')
        
    return app