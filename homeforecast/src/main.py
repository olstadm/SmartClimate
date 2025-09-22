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
        """Main update cycle"""
        try:
            logger.info("=" * 60)
            logger.info("🔄 STARTING UPDATE CYCLE")
            logger.info("=" * 60)
            
            # Collect current sensor data
            logger.info("📊 Step 1: Collecting sensor data...")
            sensor_data = await self.ha_client.get_sensor_data()
            logger.info(f"✅ Sensor data collected: {sensor_data}")
            
            # Store for API access
            self.last_sensor_data = sensor_data
            
            # Get weather forecast
            logger.info("🌤️ Step 2: Getting weather forecast...")
            weather_forecast = await self.ha_client.get_weather_forecast()
            logger.info(f"✅ Weather forecast retrieved with keys: {list(weather_forecast.keys())}")
            
            # Store current data
            logger.info("💾 Step 3: Storing measurement data...")
            await self.data_store.store_measurement(sensor_data)
            logger.info("✅ Measurement data stored")
            
            # Update thermal model
            logger.info("🏠 Step 4: Updating thermal model...")
            await self.thermal_model.update(sensor_data)
            logger.info("✅ Thermal model updated")
            
            # Generate forecast
            logger.info("🔮 Step 5: Generating forecast...")
            forecast_result = await self.forecast_engine.generate_forecast(
                sensor_data,
                weather_forecast,
                getattr(self, 'timezone', 'UTC')
            )
            logger.info(f"✅ Forecast generated with keys: {list(forecast_result.keys())}")
            
            # Analyze comfort and generate recommendations
            logger.info("🏡 Step 6: Analyzing comfort...")
            comfort_analysis = await self.comfort_analyzer.analyze(
                forecast_result
            )
            logger.info(f"✅ Comfort analysis completed: {comfort_analysis.get('recommended_mode', 'N/A')} mode recommended")
            
            # Publish results to Home Assistant
            logger.info("📡 Step 7: Publishing results to Home Assistant...")
            await self.publish_results(forecast_result, comfort_analysis)
            logger.info("✅ Results published to Home Assistant")
            
            logger.info("🎉 UPDATE CYCLE COMPLETE")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in update cycle: {e}", exc_info=True)
            
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
            
    async def retrain_ml_model(self):
        """Retrain ML correction model"""
        try:
            logger.info("Starting ML model retraining...")
            if hasattr(self.thermal_model, 'retrain_ml_correction'):
                await self.thermal_model.retrain_ml_correction()
            logger.info("ML model retraining complete")
        except Exception as e:
            logger.error(f"Error retraining ML model: {e}", exc_info=True)
            
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