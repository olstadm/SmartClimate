"""
Configuration management for HomeForecast addon
"""
import json
import os
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Manage addon configuration"""
    
    def __init__(self, config_path: str = "/data/options.json"):
        self.config_path = config_path
        self.config = {}
        self._defaults = {
            'update_interval_minutes': 5,
            'forecast_horizon_hours': 12,
            'comfort_min_temp': 68.0,
            'comfort_max_temp': 75.0,
            'enable_ml_correction': False,
            'enable_smart_hvac_control': False,
            'ml_retrain_days': 30,
            'data_retention_days': 90,
            'ml_training_days': 30,
            'ml_model_type': 'random_forest'
        }
        
    async def load(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                self.config = {}
                
            # Validate required fields
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}
            
    def _validate_config(self):
        """Validate configuration and ensure required fields exist"""
        required_fields = [
            'indoor_temp_entity',
            'indoor_humidity_entity', 
            'hvac_entity',
            'accuweather_api_key',
            'accuweather_location_key'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in self.config or not self.config[field]:
                missing_fields.append(field)
                
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        # Check if we have a default defined
        if default is None and key in self._defaults:
            default = self._defaults[key]
            
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        
    async def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            
    def get_comfort_range(self) -> tuple:
        """Get comfort temperature range in Fahrenheit"""
        min_temp = self.get('comfort_min_temp', 68.0)
        max_temp = self.get('comfort_max_temp', 75.0)
        return (min_temp, max_temp)
        
    def is_smart_hvac_enabled(self) -> bool:
        """Check if smart HVAC control is enabled"""
        return self.get('enable_smart_hvac_control', False)
        
    def get_ha_url(self) -> str:
        """Get Home Assistant API URL"""
        # When running as addon, use supervisor API to access Home Assistant
        return "http://supervisor/core"
        
    def get_ha_token(self) -> Optional[str]:
        """Get Home Assistant access token"""
        # When running as addon, token is in environment
        return os.environ.get('SUPERVISOR_TOKEN')
        
    def to_dict(self) -> Dict:
        """Get full configuration as dictionary"""
        return self.config.copy()