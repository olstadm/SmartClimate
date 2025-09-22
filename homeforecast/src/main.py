#!/usr/bin/env python3
"""
HomeForecast: Smart Thermal Forecasting for Home Assistant
Main application entry point
"""
import asyncio
import json
import logging
import os
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from flask import Flask
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Add local modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.ha_client import HomeAssistantClient
from utils.data_store import DataStore
from models.thermal_model import ThermalModel
from models.forecast_engine import ForecastEngine
from models.comfort_analyzer import ComfortAnalyzer
from api.web_server import create_app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/homeforecast/homeforecast.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HomeForecast:
    """Main application class"""
    
    def __init__(self):
        self.config = None
        self.ha_client = None
        self.data_store = None
        self.thermal_model = None
        self.forecast_engine = None
        self.comfort_analyzer = None
        self.timezone = 'UTC'  # Default timezone
        self.scheduler = None
        self.web_app = None
        self.running = False
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing HomeForecast...")
        
        # Load configuration
        self.config = Config()
        await self.config.load()
        
        # Initialize data store
        self.data_store = DataStore(self.config)
        await self.data_store.initialize()
        
        # Connect to Home Assistant
        self.ha_client = HomeAssistantClient(self.config)
        await self.ha_client.connect()
        
        # Get and store timezone from Home Assistant
        self.timezone = self.ha_client.get_timezone()
        logger.info(f"🌍 Using timezone: {self.timezone}")
        
        # Initialize thermal model
        self.thermal_model = ThermalModel(self.config, self.data_store)
        await self.thermal_model.initialize()
        
        # Initialize forecast engine
        self.forecast_engine = ForecastEngine(
            self.config, 
            self.thermal_model,
            self.data_store
        )
        
        # Initialize comfort analyzer
        self.comfort_analyzer = ComfortAnalyzer(
            self.config,
            self.forecast_engine
        )
        
        # Setup scheduler
        self.scheduler = AsyncIOScheduler()
        self.setup_scheduled_tasks()
        
        # Create web app
        self.web_app = create_app(self)
        
        logger.info("HomeForecast initialization complete")
        
    def setup_scheduled_tasks(self):
        """Configure scheduled tasks"""
        update_interval = self.config.get('update_interval_minutes', 5)
        
        # Main update cycle
        self.scheduler.add_job(
            self.update_cycle,
            'interval',
            minutes=update_interval,
            id='main_update',
            replace_existing=True
        )
        
        # ML model retraining (if enabled)
        if self.config.get('enable_ml_correction', False):
            # More frequent initial training attempts (every 2 hours for first few days)
            self.scheduler.add_job(
                self.check_ml_training_readiness,
                'interval',
                hours=2,
                id='ml_training_check',
                replace_existing=True
            )
            
            # Regular retraining (monthly)
            retrain_days = self.config.get('ml_retrain_days', 30)
            self.scheduler.add_job(
                self.retrain_ml_model,
                'interval',
                days=retrain_days,
                id='ml_retrain',
                replace_existing=True
            )
        
        # Data cleanup
        self.scheduler.add_job(
            self.cleanup_old_data,
            'cron',
            hour=2,
            minute=0,
            id='data_cleanup',
            replace_existing=True
        )
        
    async def update_cycle(self):
        """Main update cycle with robust error handling"""
        cycle_status = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'steps_failed': [],
            'data_quality_issues': [],
            'total_success': False
        }
        
        try:
            logger.info("=" * 60)
            logger.info("🔄 STARTING UPDATE CYCLE")
            logger.info("=" * 60)
            
            # Step 1: Collect current sensor data (critical)
            logger.info("📊 Step 1: Collecting sensor data...")
            try:
                sensor_data = await self.ha_client.get_sensor_data()
                
                # Evaluate data quality
                quality_issues = sensor_data.get('data_quality', {})
                if quality_issues.get('missing_sensors') or quality_issues.get('failed_sensors'):
                    cycle_status['data_quality_issues'].extend([
                        f"Missing sensors: {quality_issues.get('missing_sensors', [])}",
                        f"Failed sensors: {quality_issues.get('failed_sensors', [])}"
                    ])
                    logger.warning(f"⚠️ Sensor data quality issues detected")
                
                self.last_sensor_data = sensor_data
                cycle_status['steps_completed'].append('sensor_data_collection')
                logger.info(f"✅ Sensor data collected with {len(quality_issues.get('warnings', []))} warnings")
                
            except Exception as e:
                logger.error(f"❌ Critical failure collecting sensor data: {e}")
                cycle_status['steps_failed'].append(f'sensor_data_collection: {str(e)}')
                # Cannot continue without sensor data
                await self._publish_system_health(cycle_status)
                return
            
            # Step 2: Get weather forecast (important but not critical)
            logger.info("🌤️ Step 2: Getting weather forecast...")
            weather_forecast = None
            try:
                weather_forecast = await self.ha_client.get_weather_forecast()
                
                # Check weather data quality
                weather_quality = weather_forecast.get('data_quality', {})
                if weather_quality.get('issues'):
                    cycle_status['data_quality_issues'].extend(
                        [f"Weather: {issue}" for issue in weather_quality['issues']]
                    )
                    
                cycle_status['steps_completed'].append('weather_forecast')
                logger.info(f"✅ Weather forecast retrieved from {weather_quality.get('source', 'unknown')}")
                
            except Exception as e:
                logger.warning(f"⚠️ Weather forecast failed, using fallback: {e}")
                cycle_status['steps_failed'].append(f'weather_forecast: {str(e)}')
                # Continue with fallback weather data
                weather_forecast = {
                    'hourly_forecast': [],
                    'current_outdoor': {'temperature': sensor_data.get('indoor_temp', 70), 'humidity': 50},
                    'data_quality': {'source': 'fallback', 'issues': ['Weather API unavailable']}
                }
            
            # Step 3: Store measurement data (important for learning)
            logger.info("💾 Step 3: Storing measurement data...")
            try:
                await self.data_store.store_measurement(sensor_data)
                cycle_status['steps_completed'].append('data_storage')
                logger.info("✅ Measurement data stored")
            except Exception as e:
                logger.warning(f"⚠️ Data storage failed: {e}")
                cycle_status['steps_failed'].append(f'data_storage: {str(e)}')
                # Continue without storing (affects learning but not immediate operation)
            
            # Step 4: Update thermal model (critical for forecasting)
            logger.info("🏠 Step 4: Updating thermal model...")
            try:
                await self.thermal_model.update(sensor_data)
                cycle_status['steps_completed'].append('thermal_model_update')
                logger.info("✅ Thermal model updated")
            except Exception as e:
                logger.error(f"❌ Thermal model update failed: {e}")
                cycle_status['steps_failed'].append(f'thermal_model_update: {str(e)}')
                # Continue with existing model state
            
            # Step 5: Generate forecast (critical for system function)
            logger.info("🔮 Step 5: Generating forecast...")
            forecast_result = None
            try:
                forecast_result = await self.forecast_engine.generate_forecast(
                    sensor_data,
                    weather_forecast,
                    getattr(self, 'timezone', 'UTC')
                )
                cycle_status['steps_completed'].append('forecast_generation')
                logger.info(f"✅ Forecast generated with keys: {list(forecast_result.keys())}")
            except Exception as e:
                logger.error(f"❌ Forecast generation failed: {e}")
                cycle_status['steps_failed'].append(f'forecast_generation: {str(e)}')
                # Use simplified forecast as fallback
                forecast_result = self._generate_simple_forecast(sensor_data, weather_forecast)
                logger.warning("Using simplified fallback forecast")
            
            # Step 6: Analyze comfort (important for recommendations)
            logger.info("🏡 Step 6: Analyzing comfort...")
            comfort_analysis = None
            try:
                comfort_analysis = await self.comfort_analyzer.analyze(forecast_result)
                cycle_status['steps_completed'].append('comfort_analysis')
                logger.info(f"✅ Comfort analysis completed: {comfort_analysis.get('recommended_mode', 'N/A')} mode recommended")
            except Exception as e:
                logger.warning(f"⚠️ Comfort analysis failed: {e}")
                cycle_status['steps_failed'].append(f'comfort_analysis: {str(e)}')
                # Use safe defaults
                comfort_analysis = {
                    'recommended_mode': 'off',
                    'confidence': 0.0,
                    'reasoning': 'Analysis failed - using safe default'
                }
            
            # Step 7: Publish results (important for HA integration)
            logger.info("📡 Step 7: Publishing results to Home Assistant...")
            try:
                await self.publish_results(forecast_result, comfort_analysis)
                cycle_status['steps_completed'].append('result_publication')
                logger.info("✅ Results published to Home Assistant")
            except Exception as e:
                logger.warning(f"⚠️ Publishing results failed: {e}")
                cycle_status['steps_failed'].append(f'result_publication: {str(e)}')
            
            # Determine overall success
            critical_steps = ['sensor_data_collection', 'forecast_generation']
            critical_failures = [step for step in cycle_status['steps_failed'] 
                               if any(critical in step for critical in critical_steps)]
            
            cycle_status['total_success'] = len(critical_failures) == 0
            
            if cycle_status['total_success']:
                logger.info("🎉 UPDATE CYCLE COMPLETE - All critical steps successful")
            else:
                logger.warning(f"⚠️ UPDATE CYCLE COMPLETE with issues - Critical failures: {len(critical_failures)}")
            
            # Publish system health status
            await self._publish_system_health(cycle_status)
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in update cycle: {e}", exc_info=True)
            cycle_status['steps_failed'].append(f'unexpected_error: {str(e)}')
            cycle_status['total_success'] = False
            await self._publish_system_health(cycle_status)

    def _generate_simple_forecast(self, sensor_data: Dict, weather_forecast: Dict) -> Dict:
        """Generate a simple fallback forecast when the main engine fails"""
        logger.info("Generating simple fallback forecast")
        
        current_temp = sensor_data.get('indoor_temp', 70.0)
        outdoor_temp = weather_forecast.get('current_outdoor', {}).get('temperature', current_temp)
        
        # Simple prediction: indoor temperature will slowly drift toward outdoor temperature
        temp_diff = outdoor_temp - current_temp
        drift_rate = temp_diff * 0.1  # 10% of difference per hour
        
        return {
            'initial_conditions': {
                'indoor_temp': current_temp,
                'outdoor_temp': outdoor_temp,
                'hvac_state': sensor_data.get('hvac_state', 'off')
            },
            'hourly_predictions': [{
                'hour': i,
                'temperature': current_temp + (drift_rate * i),
                'confidence': 0.3  # Low confidence for fallback
            } for i in range(1, 13)],
            'forecast_type': 'simple_fallback'
        }

    async def _publish_system_health(self, cycle_status: Dict):
        """Publish system health and data quality metrics to Home Assistant"""
        try:
            # Calculate cycle success rate
            total_steps = len(cycle_status['steps_completed']) + len(cycle_status['steps_failed'])
            success_rate = (len(cycle_status['steps_completed']) / total_steps * 100) if total_steps > 0 else 0
            
            # Overall system health score
            health_score = success_rate
            if cycle_status['data_quality_issues']:
                health_score -= len(cycle_status['data_quality_issues']) * 5  # Deduct 5% per issue
            health_score = max(0, min(100, health_score))
            
            # Publish health metrics
            await self.ha_client.update_sensor(
                'homeforecast.system_health',
                round(health_score),
                unit='%',
                friendly_name='HomeForecast System Health'
            )
            
            await self.ha_client.update_sensor(
                'homeforecast.cycle_success_rate',
                round(success_rate, 1),
                unit='%',
                friendly_name='HomeForecast Cycle Success Rate'
            )
            
            # Data quality summary
            quality_summary = "Good"
            if cycle_status['data_quality_issues']:
                if len(cycle_status['data_quality_issues']) >= 3:
                    quality_summary = "Poor"
                else:
                    quality_summary = "Fair"
                    
            await self.ha_client.update_sensor(
                'homeforecast.data_quality',
                quality_summary,
                friendly_name='HomeForecast Data Quality'
            )
            
            # Last successful update time
            if cycle_status['total_success']:
                await self.ha_client.update_sensor(
                    'homeforecast.last_successful_update',
                    cycle_status['start_time'].isoformat(),
                    friendly_name='HomeForecast Last Successful Update'
                )
                
            logger.debug(f"System health published: {health_score}% health, {success_rate}% success rate")
            
        except Exception as e:
            logger.warning(f"Failed to publish system health metrics: {e}")
            
    async def publish_results(self, forecast_result, comfort_analysis):
        """Publish results as Home Assistant sensors"""
        try:
            logger.info("Publishing thermal model parameters...")
            # Thermal model parameters
            params = self.thermal_model.get_parameters()
            logger.info(f"Thermal parameters: {params}")
            
            await self.ha_client.update_sensor(
                'homeforecast.thermal_time_constant',
                params['time_constant'],
                unit='hours',
                friendly_name='Thermal Time Constant'
            )
            await self.ha_client.update_sensor(
                'homeforecast.heating_rate',
                params['heating_rate'],
                unit='°F/hour',
                friendly_name='Heating Rate'
            )
            await self.ha_client.update_sensor(
                'homeforecast.cooling_rate',
                params['cooling_rate'],
                unit='°F/hour',
                friendly_name='Cooling Rate'
            )
            
            # Current forecast
            await self.ha_client.update_sensor(
                'homeforecast.forecast_12h',
                forecast_result['indoor_forecast'][-1],
                unit='°F',
                friendly_name='12 Hour Temperature Forecast'
            )
            
            # Comfort analysis
            await self.ha_client.update_sensor(
                'homeforecast.time_to_upper_limit',
                comfort_analysis['time_to_upper'],
                unit='minutes',
                friendly_name='Time to Upper Comfort Limit'
            )
            await self.ha_client.update_sensor(
                'homeforecast.time_to_lower_limit',
                comfort_analysis['time_to_lower'],
                unit='minutes',
                friendly_name='Time to Lower Comfort Limit'
            )
            await self.ha_client.update_sensor(
                'homeforecast.recommended_mode',
                comfort_analysis['recommended_mode'],
                friendly_name='Recommended HVAC Mode'
            )
            await self.ha_client.update_sensor(
                'homeforecast.smart_hvac_enabled',
                comfort_analysis.get('smart_hvac_enabled', False),
                friendly_name='Smart HVAC Control Enabled'
            )
            
            # HVAC recommendations
            if comfort_analysis['hvac_start_time']:
                await self.ha_client.update_sensor(
                    'homeforecast.hvac_start_time',
                    comfort_analysis['hvac_start_time'].isoformat(),
                    friendly_name='Recommended HVAC Start Time'
                )
            if comfort_analysis['hvac_stop_time']:
                await self.ha_client.update_sensor(
                    'homeforecast.hvac_stop_time',
                    comfort_analysis['hvac_stop_time'].isoformat(),
                    friendly_name='Recommended HVAC Stop Time'
                )
                
            # Store forecast for visualization
            await self.data_store.store_forecast(forecast_result)
            
        except Exception as e:
            logger.error(f"Error publishing results: {e}", exc_info=True)
            
    async def check_ml_training_readiness(self):
        """Check if ML model is ready to train or needs retraining"""
        try:
            if not self.config.get('enable_ml_correction', False):
                return
                
            # Check if ML model exists and is trained
            if (hasattr(self.thermal_model, 'ml_corrector') and 
                self.thermal_model.ml_corrector and 
                not self.thermal_model.ml_corrector.is_trained):
                
                logger.info("🤖 ML model not yet trained, checking data availability...")
                
                # Check how much training data we have
                training_data = await self.data_store.get_training_data(7)  # Last 7 days
                data_count = len(training_data) if training_data else 0
                
                logger.info(f"📊 Available training data points: {data_count}/100 minimum needed")
                
                if data_count >= 100:
                    logger.info("✅ Sufficient data available, attempting ML model training...")
                    await self.retrain_ml_model()
                else:
                    hours_needed = max(1, (100 - data_count) / 12)  # Estimate based on 5-min intervals
                    logger.info(f"⏳ Need ~{hours_needed:.1f} more hours of data collection for ML training")
                    
        except Exception as e:
            logger.error(f"Error checking ML training readiness: {e}", exc_info=True)

    async def retrain_ml_model(self):
        """Retrain ML correction model"""
        try:
            logger.info("🚀 Starting ML model retraining...")
            if hasattr(self.thermal_model, 'retrain_ml_correction'):
                await self.thermal_model.retrain_ml_correction()
            logger.info("✅ ML model retraining complete")
        except Exception as e:
            logger.error(f"❌ Error retraining ML model: {e}", exc_info=True)
            
    async def cleanup_old_data(self):
        """Clean up old data from storage"""
        try:
            retention_days = self.config.get('data_retention_days', 90)
            await self.data_store.cleanup_old_data(retention_days)
            logger.info(f"Cleaned up data older than {retention_days} days")
        except Exception as e:
            logger.error(f"Error cleaning up data: {e}", exc_info=True)
            
    async def start(self):
        """Start the application"""
        self.running = True
        
        # Start scheduler
        self.scheduler.start()
        
        # Run initial update
        await self.update_cycle()
        
        # Start web server in background
        from threading import Thread
        web_thread = Thread(
            target=self.web_app.run,
            kwargs={'host': '0.0.0.0', 'port': 5000}
        )
        web_thread.daemon = True
        web_thread.start()
        
        logger.info("HomeForecast is running")
        
        # Keep running until stopped
        while self.running:
            await asyncio.sleep(1)
            
    async def stop(self):
        """Stop the application"""
        logger.info("Stopping HomeForecast...")
        self.running = False
        
        if self.scheduler:
            self.scheduler.shutdown()
            
        if self.ha_client:
            await self.ha_client.disconnect()
            
        if self.data_store:
            await self.data_store.close()
            
        logger.info("HomeForecast stopped")
        

async def main():
    """Main entry point"""
    app = HomeForecast()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(app.stop())
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize and start
        await app.initialize()
        await app.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
        

if __name__ == "__main__":
    asyncio.run(main())