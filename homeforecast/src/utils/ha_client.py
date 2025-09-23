"""
Home Assistant API client for sensor data collection and publishing
"""
import aiohttp
import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
try:
    import pytz
    from zoneinfo import ZoneInfo
    HAS_TIMEZONE = True
except ImportError:
    HAS_TIMEZONE = False

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Client for interacting with Home Assistant API"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = config.get_ha_url()
        self.token = config.get_ha_token()
        self.session = None
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        # Initialize weather cache with unified format
        self._weather_cache = {}
        
    async def connect(self):
        """Establish connection to Home Assistant"""
        try:
            # Create session with timeout configuration
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            logger.info(f"Connecting to Home Assistant at {self.base_url}")
            logger.debug(f"Using token: {'***' + self.token[-4:] if self.token and len(self.token) > 4 else 'No token'}")
            
            # Test connection and get HA config
            async with self.session.get(
                f"{self.base_url}/",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Connected to Home Assistant {data.get('version')}")
                    
                    # Get Home Assistant configuration including timezone
                    ha_config = await self.get_ha_config()
                    if ha_config:
                        self.timezone = ha_config.get('time_zone', 'UTC')
                        logger.info(f"🌍 Home Assistant timezone: {self.timezone}")
                    else:
                        self.timezone = 'UTC'
                        logger.warning("Could not get HA timezone, defaulting to UTC")
                        
                else:
                    logger.error(f"Connection failed with status {resp.status}")
                    response_text = await resp.text()
                    logger.error(f"Response: {response_text}")
                    raise ConnectionError(f"Failed to connect: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Error connecting to Home Assistant: {e}")
            if self.session:
                await self.session.close()
            raise
            
    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()

    async def get_ha_config(self) -> Optional[Dict]:
        """Get Home Assistant configuration including timezone"""
        try:
            logger.info("🔧 Getting Home Assistant configuration...")
            
            async with self.session.get(
                f"{self.base_url}/config",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    config = await resp.json()
                    logger.info(f"HA Config - Location: {config.get('location_name')}, " +
                              f"Timezone: {config.get('time_zone')}, " +
                              f"Version: {config.get('version')}")
                    return config
                else:
                    logger.warning(f"Failed to get HA config: {resp.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting HA config: {e}")
            return None

    def get_timezone(self) -> str:
        """Get the configured timezone"""
        return getattr(self, 'timezone', 'UTC')

    def get_local_time(self, utc_datetime=None) -> datetime:
        """Convert UTC datetime to local timezone"""
        try:
            if utc_datetime is None:
                utc_datetime = datetime.utcnow()
            
            timezone_name = self.get_timezone()
            
            if HAS_TIMEZONE:
                try:
                    # Try using zoneinfo (Python 3.9+)
                    from zoneinfo import ZoneInfo
                    local_tz = ZoneInfo(timezone_name)
                    if utc_datetime.tzinfo is None:
                        utc_datetime = utc_datetime.replace(tzinfo=ZoneInfo('UTC'))
                    return utc_datetime.astimezone(local_tz)
                except ImportError:
                    try:
                        # Fallback to pytz
                        import pytz
                        local_tz = pytz.timezone(timezone_name)
                        utc_tz = pytz.UTC
                        if utc_datetime.tzinfo is None:
                            utc_datetime = utc_tz.localize(utc_datetime)
                        return utc_datetime.astimezone(local_tz)
                    except ImportError:
                        pass
            
            # Fallback: return UTC time with timezone name note
            logger.warning(f"Timezone conversion not available, using UTC for {timezone_name}")
            return utc_datetime
            
        except Exception as e:
            logger.error(f"Error converting timezone: {e}")
            return utc_datetime or datetime.utcnow()

    def format_local_time(self, utc_datetime=None, format_str="%Y-%m-%d %H:%M:%S") -> str:
        """Format UTC datetime as local time string"""
        local_time = self.get_local_time(utc_datetime)
        return local_time.strftime(format_str)

    def format_time_for_display(self, utc_datetime=None) -> str:
        """Format datetime for dashboard display (h:mm AM/PM format)"""
        import platform
        local_time = self.get_local_time(utc_datetime)
        
        # Format as h:mm AM/PM (handle leading zero differences between platforms)
        if platform.system() == 'Windows':
            return local_time.strftime("%#I:%M %p")
        else:
            return local_time.strftime("%-I:%M %p")

    def format_datetime_for_display(self, utc_datetime=None) -> str:
        """Format full datetime for dashboard display (m/d h:mm AM/PM format)"""
        import platform
        local_time = self.get_local_time(utc_datetime)
        
        # Format as m/d h:mm AM/PM (handle leading zero differences between platforms)
        if platform.system() == 'Windows':
            return local_time.strftime("%#m/%#d %#I:%M %p")
        else:
            return local_time.strftime("%-m/%-d %-I:%M %p")
    
    @staticmethod
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
            
    async def get_sensor_data(self) -> Dict:
        """Collect current sensor data with robust error handling"""
        logger.info("=== Collecting Local Sensor Data ===")
        data = {
            'timestamp': datetime.now(),
            'data_quality': {'missing_sensors': [], 'failed_sensors': [], 'warnings': []}
        }
        
        # Get indoor temperature (required sensor)
        indoor_temp_entity = self.config.get('indoor_temp_entity')
        logger.info(f"Reading indoor temperature from entity: {indoor_temp_entity}")
        
        indoor_temp = await self._get_state_robust(indoor_temp_entity, 'indoor_temp')
        if indoor_temp is not None:
            try:
                data['indoor_temp'] = float(indoor_temp)
                logger.info(f"✅ Indoor temperature: {data['indoor_temp']}°F")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid indoor temperature value '{indoor_temp}': {e}")
                data['indoor_temp'] = self._get_last_known_value('indoor_temp', 70.0)
                data['data_quality']['warnings'].append('Using last known indoor temperature')
        else:
            logger.warning("❌ Indoor temperature sensor unavailable - using fallback")
            data['indoor_temp'] = self._get_last_known_value('indoor_temp', 70.0)
            data['data_quality']['missing_sensors'].append('indoor_temp')
        
        # Get indoor humidity (required sensor)
        indoor_humidity_entity = self.config.get('indoor_humidity_entity')
        logger.info(f"Reading indoor humidity from entity: {indoor_humidity_entity}")
        
        indoor_humidity = await self._get_state_robust(indoor_humidity_entity, 'indoor_humidity')
        if indoor_humidity is not None:
            try:
                humidity_val = float(indoor_humidity)
                # Validate humidity range
                if 0 <= humidity_val <= 100:
                    data['indoor_humidity'] = humidity_val
                    logger.info(f"✅ Indoor humidity: {data['indoor_humidity']}%")
                else:
                    logger.warning(f"Indoor humidity out of range: {humidity_val}%")
                    data['indoor_humidity'] = self._get_last_known_value('indoor_humidity', 50.0)
                    data['data_quality']['warnings'].append('Indoor humidity out of range')
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid indoor humidity value '{indoor_humidity}': {e}")
                data['indoor_humidity'] = self._get_last_known_value('indoor_humidity', 50.0)
                data['data_quality']['warnings'].append('Invalid indoor humidity format')
        else:
            logger.warning("❌ Indoor humidity sensor unavailable - using fallback")
            data['indoor_humidity'] = self._get_last_known_value('indoor_humidity', 50.0)
            data['data_quality']['missing_sensors'].append('indoor_humidity')
        
        # Get outdoor temperature (optional sensor)
        outdoor_temp_entity = self.config.get('outdoor_temp_entity')
        if outdoor_temp_entity:
            logger.info(f"Reading outdoor temperature from entity: {outdoor_temp_entity}")
            outdoor_temp = await self._get_state_robust(outdoor_temp_entity, 'outdoor_temp')
            if outdoor_temp is not None:
                try:
                    data['outdoor_temp'] = float(outdoor_temp)
                    logger.info(f"✅ Local outdoor temperature: {data['outdoor_temp']}°F")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid outdoor temperature value '{outdoor_temp}': {e}")
                    data['outdoor_temp'] = None
                    data['data_quality']['warnings'].append('Invalid outdoor temperature format')
            else:
                logger.warning("❌ Outdoor temperature sensor unavailable")
                data['outdoor_temp'] = None
                data['data_quality']['missing_sensors'].append('outdoor_temp')
        else:
            data['outdoor_temp'] = None
            logger.info("No local outdoor temperature sensor configured - will use AccuWeather data")
        
        # Get outdoor humidity (optional sensor)
        outdoor_humidity_entity = self.config.get('outdoor_humidity_entity')
        if outdoor_humidity_entity:
            logger.info(f"Reading outdoor humidity from entity: {outdoor_humidity_entity}")
            outdoor_humidity = await self._get_state_robust(outdoor_humidity_entity, 'outdoor_humidity')
            if outdoor_humidity is not None:
                try:
                    humidity_val = float(outdoor_humidity)
                    if 0 <= humidity_val <= 100:
                        data['outdoor_humidity'] = humidity_val
                        logger.info(f"✅ Local outdoor humidity: {data['outdoor_humidity']}%")
                    else:
                        logger.warning(f"Outdoor humidity out of range: {humidity_val}%")
                        data['outdoor_humidity'] = None
                        data['data_quality']['warnings'].append('Outdoor humidity out of range')
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid outdoor humidity value '{outdoor_humidity}': {e}")
                    data['outdoor_humidity'] = None
                    data['data_quality']['warnings'].append('Invalid outdoor humidity format')
            else:
                logger.warning("❌ Outdoor humidity sensor unavailable")
                data['outdoor_humidity'] = None
                data['data_quality']['missing_sensors'].append('outdoor_humidity')
        else:
            data['outdoor_humidity'] = None
            logger.info("No local outdoor humidity sensor configured - will use AccuWeather data")
            
        # Get comprehensive thermostat data (critical for HVAC control)
        hvac_entity = self.config.get('hvac_entity')
        logger.info(f"Reading thermostat data from entity: {hvac_entity}")
        
        try:
            thermostat_data = await self.get_thermostat_data(hvac_entity)
            
            # Include thermostat data in sensor data with validation
            data['hvac_state'] = thermostat_data.get('hvac_state', 'unknown')
            data['hvac_mode'] = thermostat_data.get('hvac_mode', 'off')
            data['hvac_action'] = thermostat_data.get('hvac_action', 'idle')
            data['target_temperature'] = thermostat_data.get('target_temperature', data['indoor_temp'])
            data['thermostat_data'] = thermostat_data
            
            logger.info(f"✅ Thermostat info - State: {data['hvac_state']}, Mode: {data['hvac_mode']}, " +
                       f"Action: {data['hvac_action']}, Target: {data['target_temperature']}°F")
                       
            # Check for thermostat connection issues
            if thermostat_data.get('connection_status') == 'unavailable':
                data['data_quality']['warnings'].append('Thermostat connection unstable')
                
        except Exception as e:
            logger.error(f"❌ Failed to get thermostat data: {e}")
            # Use safe defaults for thermostat
            data['hvac_state'] = 'off'
            data['hvac_mode'] = 'off'  
            data['hvac_action'] = 'idle'
            data['target_temperature'] = data['indoor_temp']
            data['thermostat_data'] = self._get_default_thermostat_data()
            data['data_quality']['failed_sensors'].append('hvac_thermostat')
            logger.warning("Using default thermostat values due to connection failure")
        
        # Get solar irradiance if available
        solar_entity = self.config.get('solar_irradiance_entity')
        if solar_entity:
            logger.info(f"Reading solar irradiance from entity: {solar_entity}")
            solar = await self._get_state_robust(solar_entity, 'solar_irradiance')
            if solar is not None:
                try:
                    solar_val = float(solar)
                    data['solar_irradiance'] = max(0.0, solar_val)  # Ensure non-negative
                    logger.info(f"✅ Solar irradiance: {data['solar_irradiance']} W/m²")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid solar irradiance value '{solar}': {e}")
                    data['solar_irradiance'] = 0.0
                    data['data_quality']['warnings'].append('Invalid solar irradiance format')
            else:
                logger.warning("❌ Solar irradiance sensor unavailable")
                data['solar_irradiance'] = 0.0
                data['data_quality']['missing_sensors'].append('solar_irradiance')
        else:
            data['solar_irradiance'] = 0.0
            logger.info("No solar irradiance sensor configured")
            
        # Cache current values for future fallbacks
        self._cache_sensor_values(data)
        
        # Log data quality summary
        quality = data['data_quality']
        if quality['missing_sensors'] or quality['failed_sensors'] or quality['warnings']:
            logger.warning(f"📊 Data Quality Issues - Missing: {quality['missing_sensors']}, " +
                         f"Failed: {quality['failed_sensors']}, Warnings: {len(quality['warnings'])}")
        else:
            logger.info("📊 All sensor data collected successfully")
            
        return data
            
    async def _get_state(self, entity_id: str) -> Optional[str]:
        """Get state of an entity"""
        try:
            async with self.session.get(
                f"{self.base_url}/states/{entity_id}",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('state')
                else:
                    logger.warning(f"Failed to get state for {entity_id}: {resp.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting state for {entity_id}: {e}")
            return None

    async def _get_state_robust(self, entity_id: str, sensor_type: str, retries: int = 2) -> Optional[str]:
        """Get state of an entity with retry logic and staleness detection"""
        if not entity_id:
            logger.warning(f"No entity_id provided for {sensor_type}")
            return None
            
        for attempt in range(retries + 1):
            try:
                async with self.session.get(
                    f"{self.base_url}/states/{entity_id}",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)  # 10 second timeout
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        state = data.get('state')
                        
                        # Check for unavailable states
                        if state in ['unavailable', 'unknown', 'none', None, '']:
                            logger.warning(f"Entity {entity_id} state is '{state}' on attempt {attempt + 1}")
                            if attempt < retries:
                                await asyncio.sleep(1)  # Brief delay before retry
                                continue
                            return None
                            
                        # Check data freshness
                        last_changed = data.get('last_changed')
                        if last_changed and self._is_data_stale(last_changed, sensor_type):
                            logger.warning(f"Entity {entity_id} data is stale (last changed: {last_changed})")
                            # Continue anyway but log the staleness
                            
                        return state
                        
                    elif resp.status == 404:
                        logger.error(f"Entity {entity_id} not found (404)")
                        return None
                    else:
                        logger.warning(f"Failed to get state for {entity_id}: HTTP {resp.status} on attempt {attempt + 1}")
                        if attempt < retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting state for {entity_id} on attempt {attempt + 1}")
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return None
            except Exception as e:
                logger.warning(f"Error getting state for {entity_id} on attempt {attempt + 1}: {e}")
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return None
                
        return None

    def _is_data_stale(self, last_changed_str: str, sensor_type: str) -> bool:
        """Check if sensor data is too old to be reliable"""
        try:
            last_changed = datetime.fromisoformat(last_changed_str.replace('Z', '+00:00'))
            now = datetime.now(last_changed.tzinfo) if last_changed.tzinfo else datetime.now()
            age_minutes = (now - last_changed).total_seconds() / 60
            
            # Define staleness thresholds by sensor type
            thresholds = {
                'indoor_temp': 15,      # Indoor sensors should update frequently
                'indoor_humidity': 15,
                'outdoor_temp': 60,     # Outdoor sensors can be less frequent
                'outdoor_humidity': 60,
                'solar_irradiance': 30,
                'hvac': 5              # HVAC state should be very current
            }
            
            threshold = thresholds.get(sensor_type, 30)  # Default 30 minutes
            return age_minutes > threshold
            
        except Exception as e:
            logger.warning(f"Could not parse last_changed time '{last_changed_str}': {e}")
            return False

    def _get_last_known_value(self, sensor_type: str, default_value: float) -> float:
        """Get last known good value for a sensor type with fallback"""
        # Initialize cache if needed
        if not hasattr(self, '_sensor_cache'):
            self._sensor_cache = {}
            
        cached_value = self._sensor_cache.get(sensor_type)
        if cached_value is not None:
            logger.info(f"Using cached value for {sensor_type}: {cached_value}")
            return cached_value
            
        logger.info(f"No cached value for {sensor_type}, using default: {default_value}")
        return default_value

    def _cache_sensor_values(self, data: Dict):
        """Cache current sensor values for future fallback use"""
        if not hasattr(self, '_sensor_cache'):
            self._sensor_cache = {}
            
        # Cache only valid numeric values
        cache_mapping = {
            'indoor_temp': data.get('indoor_temp'),
            'indoor_humidity': data.get('indoor_humidity'), 
            'outdoor_temp': data.get('outdoor_temp'),
            'outdoor_humidity': data.get('outdoor_humidity'),
            'solar_irradiance': data.get('solar_irradiance')
        }
        
        for key, value in cache_mapping.items():
            if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
                self._sensor_cache[key] = value
                
        # Add timestamp to cache
        self._sensor_cache['last_update'] = datetime.now()
            
    async def _get_climate_state(self, entity_id: str) -> str:
        """Get HVAC operating state from climate entity (for backward compatibility)"""
        climate_data = await self.get_thermostat_data(entity_id)
        return climate_data.get('hvac_action', 'off')

    async def get_thermostat_data(self, entity_id: str) -> Dict:
        """Get comprehensive thermostat information from climate entity"""
        try:
            logger.info(f"📡 Collecting thermostat data from entity: {entity_id}")
            
            async with self.session.get(
                f"{self.base_url}/states/{entity_id}",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    attributes = data.get('attributes', {})
                    
                    # Extract comprehensive thermostat information
                    thermostat_data = {
                        # Current state
                        'hvac_mode': data.get('state', 'off').lower(),  # off, heat, cool, auto
                        'hvac_action': attributes.get('hvac_action', 'idle').lower(),  # heating, cooling, idle
                        
                        # Temperature settings
                        'current_temperature': attributes.get('current_temperature', 70.0),
                        'target_temperature': attributes.get('temperature', 72.0),  # setpoint
                        
                        # Humidity
                        'current_humidity': attributes.get('current_humidity', 50.0),
                        
                        # Fan settings
                        'fan_mode': attributes.get('fan_mode', 'auto'),
                        
                        # Available modes
                        'hvac_modes': attributes.get('hvac_modes', ['off', 'heat', 'cool']),
                        'fan_modes': attributes.get('fan_modes', ['auto', 'on']),
                        
                        # Temperature limits
                        'min_temp': attributes.get('min_temp', 50),
                        'max_temp': attributes.get('max_temp', 95),
                        'target_temp_step': attributes.get('target_temp_step', 1),
                        
                        # Additional attributes
                        'friendly_name': attributes.get('friendly_name', 'Thermostat'),
                        'supported_features': attributes.get('supported_features', 0)
                    }
                    
                    # Normalize hvac_action for our system
                    hvac_action = thermostat_data['hvac_action']
                    if hvac_action in ['heating', 'heat']:
                        thermostat_data['hvac_state'] = 'heat'
                    elif hvac_action in ['cooling', 'cool']:
                        thermostat_data['hvac_state'] = 'cool'
                    elif hvac_action in ['idle', 'off']:
                        thermostat_data['hvac_state'] = 'off'
                    else:
                        thermostat_data['hvac_state'] = 'off'
                    
                    logger.info(f"🌡️ Thermostat data: Mode={thermostat_data['hvac_mode']}, " +
                              f"Action={thermostat_data['hvac_action']}, " +
                              f"Current={thermostat_data['current_temperature']}°F, " +
                              f"Target={thermostat_data['target_temperature']}°F")
                    
                    return thermostat_data
                    
                else:
                    logger.warning(f"Failed to get climate state for {entity_id}: {resp.status}")
                    return self._get_default_thermostat_data()
                    
        except Exception as e:
            logger.error(f"Error getting thermostat data: {e}")
            return self._get_default_thermostat_data()

    def _get_default_thermostat_data(self) -> Dict:
        """Return default thermostat data when unable to connect"""
        return {
            'hvac_mode': 'off',
            'hvac_action': 'idle', 
            'hvac_state': 'off',
            'current_temperature': 70.0,
            'target_temperature': 72.0,
            'current_humidity': 50.0,
            'fan_mode': 'auto',
            'hvac_modes': ['off', 'heat', 'cool'],
            'fan_modes': ['auto', 'on'],
            'min_temp': 50,
            'max_temp': 95,
            'target_temp_step': 1,
            'friendly_name': 'Thermostat',
            'supported_features': 0
        }
            
    async def get_weather_forecast(self) -> Dict:
        """Get weather forecast from AccuWeather with robust error handling and fallbacks"""
        logger.info("=== Getting Weather Forecast from AccuWeather ===")
        
        # Check configuration
        api_key = self.config.get('accuweather_api_key')
        location_key = self.config.get('accuweather_location_key')
        
        if not api_key or not location_key:
            logger.error("❌ AccuWeather API credentials not configured")
            return self._get_fallback_weather_data("Missing API credentials")
            
        logger.info(f"AccuWeather API configured: Location key = {location_key[:8]}... API key = {'*' * len(api_key[:-4]) + api_key[-4:] if len(api_key) > 4 else 'Yes'}")
        
        forecast_data = {
            'hourly_forecast': [],
            'current_outdoor': {},
            'data_quality': {'source': 'accuweather', 'issues': []}
        }
        
        # Try to get current conditions with retry logic
        current_success = await self._get_accuweather_current(api_key, location_key, forecast_data)
        
        # Try to get forecast with retry logic  
        forecast_success = await self._get_accuweather_forecast(api_key, location_key, forecast_data)
        
        # Get historical data from internal cache for trend analysis  
        logger.info("🕰️ Fetching historical data from internal cache for trend analysis...")
        historical_data = await self.get_cached_historical_data(hours=6)
        if historical_data and 'historical_weather' in historical_data:
            forecast_data['historical_weather'] = historical_data['historical_weather']
            if historical_data['data_quality']['issues']:
                forecast_data['data_quality']['issues'].extend(historical_data['data_quality']['issues'])
            logger.info(f"✅ Cached historical data integrated: {len(historical_data['historical_weather'])} data points")
        else:
            logger.warning("⚠️ No cached historical data available - trend analysis will be limited")
            forecast_data['historical_weather'] = []
        
        # If both current and forecast failed, return fallback data
        if not current_success and not forecast_success:
            logger.error("❌ Complete AccuWeather API failure - using fallback weather data")
            fallback_data = self._get_fallback_weather_data("API completely unavailable")
            # Still include whatever historical data we got
            if forecast_data['historical_weather']:
                fallback_data['historical_weather'] = forecast_data['historical_weather']
            return fallback_data
        
        # If we got some data but not all, log the issue
        if not current_success:
            forecast_data['data_quality']['issues'].append('Current conditions unavailable')
        if not forecast_success:
            forecast_data['data_quality']['issues'].append('Forecast data unavailable')
            
        # Cache successful data for future fallbacks and historical reference
        if current_success or forecast_success:
            self._cache_weather_data(forecast_data)
            self.cache_current_weather_data(forecast_data)
            
        return forecast_data

    async def _get_accuweather_current(self, api_key: str, location_key: str, forecast_data: Dict, retries: int = 2) -> bool:
        """Get current conditions from AccuWeather with retry logic"""
        current_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}"
        
        for attempt in range(retries + 1):
            try:
                params = {'apikey': api_key, 'details': 'true'}
                timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout
                
                async with self.session.get(current_url, params=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) > 0:
                            current = data[0]
                            current_temp = current['Temperature']['Imperial']['Value']
                            current_humidity = current['RelativeHumidity']
                            
                            # Validate data ranges
                            if not (-50 <= current_temp <= 150):  # Reasonable temperature range
                                logger.warning(f"AccuWeather temperature out of range: {current_temp}°F")
                                if attempt < retries:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                return False
                                
                            if not (0 <= current_humidity <= 100):  # Humidity validation
                                logger.warning(f"AccuWeather humidity out of range: {current_humidity}%")
                                current_humidity = max(0, min(100, current_humidity))  # Clamp to valid range
                            
                            forecast_data['current_outdoor'] = {
                                'temperature': current_temp,
                                'humidity': current_humidity,
                                'solar_irradiance': self._estimate_solar_irradiance(current)
                            }
                            
                            logger.info(f"✅ AccuWeather current conditions: {current_temp}°F, {current_humidity}% humidity")
                            return True
                        else:
                            logger.warning(f"Empty AccuWeather current conditions response on attempt {attempt + 1}")
                            
                    elif resp.status == 401:
                        logger.error("❌ AccuWeather API authentication failed - check API key")
                        return False
                    elif resp.status == 403:
                        logger.error("❌ AccuWeather API quota exceeded or forbidden")
                        return False  
                    elif resp.status == 503:
                        logger.warning(f"AccuWeather service temporarily unavailable (503) - attempt {attempt + 1}")
                        if attempt < retries:
                            await asyncio.sleep(5 + (2 ** attempt))  # Longer delay for service issues
                            continue
                    else:
                        logger.warning(f"AccuWeather current conditions failed: HTTP {resp.status} on attempt {attempt + 1}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting AccuWeather current conditions on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error getting AccuWeather current conditions on attempt {attempt + 1}: {e}")
                
            # Wait before retry (exponential backoff)
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
                
        return False

    async def _get_accuweather_forecast(self, api_key: str, location_key: str, forecast_data: Dict, retries: int = 2) -> bool:
        """Get forecast from AccuWeather with retry logic"""
        forecast_url = f"http://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{location_key}"
        
        for attempt in range(retries + 1):
            try:
                params = {'apikey': api_key, 'metric': 'false'}
                timeout = aiohttp.ClientTimeout(total=20)  # Longer timeout for larger response
                
                async with self.session.get(forecast_url, params=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) > 0:
                            valid_hours = 0
                            for hour in data:
                                try:
                                    temp = hour['Temperature']['Value']
                                    humidity = hour.get('RelativeHumidity', 50)
                                    
                                    # Validate forecast data
                                    if not (-50 <= temp <= 150):
                                        logger.warning(f"Skipping invalid forecast temperature: {temp}°F")
                                        continue
                                        
                                    if not (0 <= humidity <= 100):
                                        humidity = max(0, min(100, humidity))  # Clamp humidity
                                        
                                    forecast_data['hourly_forecast'].append({
                                        'timestamp': datetime.fromisoformat(hour['DateTime'].replace('Z', '+00:00')),
                                        'temperature': temp,
                                        'humidity': humidity,
                                        'solar_irradiance': self._calculate_solar_irradiance(hour),
                                        'precipitation_probability': hour.get('PrecipitationProbability', 0)
                                    })
                                    valid_hours += 1
                                    
                                except Exception as e:
                                    logger.warning(f"Skipping invalid forecast hour: {e}")
                                    continue
                                    
                            if valid_hours > 0:
                                logger.info(f"✅ AccuWeather forecast: {valid_hours} valid hours retrieved")
                                return True
                            else:
                                logger.warning("No valid forecast hours found")
                                
                    elif resp.status == 401:
                        logger.error("❌ AccuWeather API authentication failed - check API key")
                        return False
                    elif resp.status == 403:
                        logger.error("❌ AccuWeather API quota exceeded or forbidden")
                        return False
                    elif resp.status == 503:
                        logger.warning(f"AccuWeather service temporarily unavailable (503) - attempt {attempt + 1}")
                        if attempt < retries:
                            await asyncio.sleep(5 + (2 ** attempt))
                            continue
                    else:
                        logger.warning(f"AccuWeather forecast failed: HTTP {resp.status} on attempt {attempt + 1}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting AccuWeather forecast on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error getting AccuWeather forecast on attempt {attempt + 1}: {e}")
                
            # Wait before retry
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
                
        return False

    async def get_cached_historical_data(self, hours: int = 6) -> Dict:
        """Get historical weather data from internal cache and database"""
        logger.info(f"=== Getting Cached Historical Data (past {hours} hours) ===")
        
        try:
            # Try to get data from database first (our recorded measurements and forecasts)
            if hasattr(self, 'data_store') and self.data_store:
                historical_data = await self._get_database_historical_data(hours)
                if historical_data and historical_data['historical_weather']:
                    logger.info(f"✅ Retrieved {len(historical_data['historical_weather'])} points from database")
                    return historical_data
            
            # Fallback to synthetic data based on current conditions
            return self._generate_synthetic_historical_data(hours)
            
        except Exception as e:
            logger.error(f"Error getting cached historical data: {e}")
            return self._generate_synthetic_historical_data(hours)

    async def _get_database_historical_data(self, hours: int) -> Dict:
        """Get historical data from database measurements and cached forecasts"""
        try:
            from ..utils.data_store import DataStore
            
            # Get recent measurements from database
            recent_measurements = await self.data_store.get_recent_measurements(hours)
            
            historical_weather = []
            
            for measurement in recent_measurements:
                if 'outdoor_temp' in measurement and 'timestamp' in measurement:
                    historical_weather.append({
                        'timestamp': measurement['timestamp'],
                        'temperature': measurement['outdoor_temp'],
                        'humidity': measurement.get('outdoor_humidity', 50.0),
                        'solar_irradiance': measurement.get('solar_irradiance', 0.0),
                        'source': 'database_measurement'
                    })
            
            # If we don't have enough database records, supplement with weather cache
            if len(historical_weather) < hours:
                logger.info(f"Only {len(historical_weather)} database records, supplementing with cached weather data")
                cached_weather = await self._get_cached_weather_predictions(hours - len(historical_weather))
                historical_weather.extend(cached_weather)
            
            # Sort by timestamp
            historical_weather.sort(key=lambda x: x['timestamp'])
            
            return {
                'historical_weather': historical_weather,
                'data_quality': {
                    'source': 'database_cache',
                    'issues': [] if len(historical_weather) >= hours // 2 else ['Limited historical data available']
                }
            }
            
        except Exception as e:
            logger.warning(f"Database historical data query failed: {e}")
            return {'historical_weather': [], 'data_quality': {'source': 'error', 'issues': [str(e)]}}

    async def _get_cached_weather_predictions(self, hours: int) -> List[Dict]:
        """Get cached weather predictions to supplement historical data"""
        cached_predictions = []
        
        try:
            # Get cached forecast data that's now "historical"
            cached_forecasts = await self.data_store.get_recent_forecasts(hours=hours + 2)  # Get a bit more to find overlaps
            
            now = datetime.now()
            cutoff_time = now - timedelta(hours=hours)
            
            for forecast in cached_forecasts:
                if 'outdoor_forecast' in forecast.get('data', {}):
                    forecast_data = forecast['data']
                    timestamps = forecast_data.get('timestamps', [])
                    outdoor_temps = forecast_data.get('outdoor_forecast', [])
                    
                    for i, timestamp in enumerate(timestamps):
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        
                        # Only include data within our historical window that's now past
                        if cutoff_time <= timestamp <= now and i < len(outdoor_temps):
                            cached_predictions.append({
                                'timestamp': timestamp,
                                'temperature': outdoor_temps[i],
                                'humidity': 50.0,  # Default
                                'solar_irradiance': 0.0,
                                'source': 'cached_prediction'
                            })
                            
        except Exception as e:
            logger.warning(f"Error getting cached weather predictions: {e}")
        
        return cached_predictions

    def _generate_synthetic_historical_data(self, hours: int) -> Dict:
        """Generate synthetic historical data when no cache is available"""
        logger.info(f"🔄 Generating synthetic historical data for past {hours} hours")
        
        # Use current conditions as baseline
        now = datetime.now()
        current_temp = 70.0  # Default temperature
        
        # Try to get current outdoor temperature from recent data
        try:
            if hasattr(self, 'current_outdoor_temp') and self.current_outdoor_temp:
                current_temp = self.current_outdoor_temp
        except:
            pass
        
        historical_data = {
            'historical_weather': [],
            'data_quality': {
                'source': 'synthetic',
                'issues': ['No cached data available', 'Using synthetic historical estimates']
            }
        }
        
        # Generate hourly data going backwards with realistic variation
        for i in range(hours):
            timestamp = now - timedelta(hours=hours - i)
            
            # Add realistic temperature variation based on time of day and randomness
            hour_of_day = timestamp.hour
            diurnal_variation = 5 * math.sin((hour_of_day - 6) * math.pi / 12)  # Peak at 2 PM
            random_variation = (i % 3 - 1) * 2  # ±2°F random variation
            estimated_temp = current_temp + diurnal_variation + random_variation
            
            historical_data['historical_weather'].append({
                'timestamp': timestamp,
                'temperature': estimated_temp,
                'humidity': 50.0 + (i % 30),  # Vary humidity 50-80%
                'solar_irradiance': self._estimate_solar_irradiance_simple(timestamp),
                'source': 'synthetic'
            })
        
        logger.info(f"✅ Generated {hours} hours of synthetic historical data")
        return historical_data

    def cache_current_weather_data(self, weather_data: Dict):
        """Cache current weather data for future historical reference"""
        try:
            # Validate weather_data is a dictionary
            if not isinstance(weather_data, dict):
                logger.debug(f"Skipping cache - weather_data is not a dict: {type(weather_data)}")
                return
            
            # This method can be called to cache current AccuWeather data
            # for use as "historical" data in future requests
            if not hasattr(self, '_weather_cache'):
                self._weather_cache = {}
            
            # Validate current_outdoor exists and is a dict
            current_outdoor = weather_data.get('current_outdoor', {})
            if not isinstance(current_outdoor, dict):
                logger.debug(f"Skipping cache - current_outdoor is not a dict: {type(current_outdoor)}")
                return
            
            # Initialize historical cache if needed
            if 'historical_entries' not in self._weather_cache:
                self._weather_cache['historical_entries'] = []
            
            # Add current conditions to historical entries with timestamp
            cache_entry = {
                'timestamp': datetime.now(),
                'outdoor_temp': current_outdoor.get('temperature'),
                'outdoor_humidity': current_outdoor.get('humidity'),
                'solar_irradiance': current_outdoor.get('solar_irradiance', 0),
                'source': 'accuweather_cached'
            }
            
            # Keep only last 24 hours of historical cache
            cutoff_time = datetime.now() - timedelta(hours=24)
            self._weather_cache['historical_entries'] = [
                entry for entry in self._weather_cache['historical_entries'] 
                if entry['timestamp'] > cutoff_time
            ]
            self._weather_cache['historical_entries'].append(cache_entry)
            
            logger.debug(f"Weather data cached - historical entries: {len(self._weather_cache['historical_entries'])}")
            
        except Exception as e:
            logger.warning(f"Error caching weather data: {e}")

    def _get_fallback_weather_data(self, reason: str) -> Dict:
        """Generate fallback weather data when AccuWeather is unavailable"""
        logger.info(f"🔄 Generating fallback weather data - Reason: {reason}")
        
        # Try to use cached weather data first
        cached_weather = self._get_cached_weather_data()
        if cached_weather:
            logger.info("Using cached weather data as fallback")
            cached_weather['data_quality'] = {
                'source': 'cached_accuweather',
                'issues': [f'AccuWeather unavailable: {reason}']
            }
            return cached_weather
            
        # If no cache, generate reasonable estimates based on season and time
        now = datetime.now()
        base_temp = self._estimate_seasonal_temperature(now)
        
        fallback_data = {
            'hourly_forecast': [],
            'current_outdoor': {
                'temperature': base_temp,
                'humidity': 50.0,  # Neutral humidity
                'solar_irradiance': self._estimate_solar_irradiance_simple(now)
            },
            'data_quality': {
                'source': 'estimated',
                'issues': [f'AccuWeather unavailable: {reason}', 'Using seasonal temperature estimates']
            }
        }
        
        # Generate simple 12-hour forecast
        for hour in range(12):
            forecast_time = now + timedelta(hours=hour + 1)
            temp_variation = self._estimate_temperature_variation(forecast_time, base_temp)
            
            fallback_data['hourly_forecast'].append({
                'timestamp': forecast_time,
                'temperature': base_temp + temp_variation,
                'humidity': 50.0,  # Static humidity estimate
                'solar_irradiance': self._estimate_solar_irradiance_simple(forecast_time),
                'precipitation_probability': 10  # Low probability estimate
            })
            
        logger.warning(f"Generated fallback weather data with base temperature {base_temp}°F")
        return fallback_data

    def _estimate_seasonal_temperature(self, dt: datetime) -> float:
        """Estimate reasonable outdoor temperature based on season"""
        # Simple seasonal temperature estimation (Northern hemisphere assumed)
        day_of_year = dt.timetuple().tm_yday
        
        # Rough seasonal cycle: coldest ~day 15 (mid-Jan), warmest ~day 196 (mid-July)
        seasonal_variation = 30 * math.sin((day_of_year - 15) * 2 * math.pi / 365)
        base_temp = 60 + seasonal_variation  # 60°F average with ±30°F seasonal swing
        
        # Daily temperature variation
        hour = dt.hour
        daily_variation = 10 * math.sin((hour - 6) * math.pi / 12)  # Peak at 2 PM, minimum at 6 AM
        
        return round(base_temp + daily_variation, 1)

    def _estimate_temperature_variation(self, forecast_time: datetime, base_temp: float) -> float:
        """Estimate temperature change from base temperature"""
        # Simple daily cycle variation
        hour = forecast_time.hour
        daily_cycle = 8 * math.sin((hour - 6) * math.pi / 12)  # ±8°F daily variation
        return daily_cycle

    def _estimate_solar_irradiance_simple(self, dt: datetime) -> float:
        """Simple solar irradiance estimation based on time of day"""
        hour = dt.hour
        
        # Solar irradiance follows roughly a sine curve during daylight hours
        if 6 <= hour <= 18:  # Daylight hours
            # Peak at noon (hour 12), zero at 6 AM/6 PM
            solar_angle = (hour - 6) * math.pi / 12
            return max(0, 800 * math.sin(solar_angle))  # Max 800 W/m²
        else:
            return 0.0

    def _cache_weather_data(self, weather_data: Dict):
        """Cache weather data for fallback use"""
        if not hasattr(self, '_weather_cache'):
            self._weather_cache = {}
            
        # Only cache if we have actual data
        if weather_data.get('current_outdoor') or weather_data.get('hourly_forecast'):
            # Preserve existing historical entries when updating main cache
            historical_entries = self._weather_cache.get('historical_entries', [])
            self._weather_cache = {
                'data': weather_data.copy(),
                'timestamp': datetime.now(),
                'historical_entries': historical_entries
            }
            logger.debug("Cached weather data for fallback use")

    def _get_cached_weather_data(self) -> Optional[Dict]:
        """Get cached weather data if recent enough"""
        if not hasattr(self, '_weather_cache') or not self._weather_cache:
            return None
        
        try:
            # Handle unified dict cache format
            if isinstance(self._weather_cache, dict):
                # Check for main weather data cache first
                if 'timestamp' in self._weather_cache and 'data' in self._weather_cache:
                    cache_age_hours = (datetime.now() - self._weather_cache['timestamp']).total_seconds() / 3600
                    
                    # Use cached data if less than 2 hours old
                    if cache_age_hours < 2:
                        logger.info(f"Using main weather cache from {cache_age_hours:.1f} hours ago")
                        return self._weather_cache['data'].copy()
                    else:
                        logger.info(f"Main weather cache too old ({cache_age_hours:.1f} hours)")
                
                # Fall back to historical entries if available
                historical_entries = self._weather_cache.get('historical_entries', [])
                if historical_entries:
                    recent_entry = max(historical_entries, key=lambda x: x.get('timestamp', datetime.min))
                    cache_age_hours = (datetime.now() - recent_entry['timestamp']).total_seconds() / 3600
                    
                    if cache_age_hours < 2:
                        logger.info(f"Using recent historical cache entry from {cache_age_hours:.1f} hours ago")
                        # Convert to expected format
                        return {
                            'current_outdoor': {
                                'temperature': recent_entry.get('outdoor_temp'),
                                'humidity': recent_entry.get('outdoor_humidity'),
                                'solar_irradiance': recent_entry.get('solar_irradiance', 0)
                            }
                        }
                    else:
                        logger.info(f"Historical cache entry too old ({cache_age_hours:.1f} hours)")
                        
            elif isinstance(self._weather_cache, list):
                # Handle legacy list format - migrate to dict format
                logger.info("Migrating legacy list cache format to unified dict format")
                if self._weather_cache:
                    recent_entry = max(self._weather_cache, key=lambda x: x.get('timestamp', datetime.min))
                    cache_age_hours = (datetime.now() - recent_entry['timestamp']).total_seconds() / 3600
                    
                    if cache_age_hours < 2:
                        logger.info(f"Using legacy cache entry from {cache_age_hours:.1f} hours ago")
                        return {
                            'current_outdoor': {
                                'temperature': recent_entry.get('outdoor_temp'),
                                'humidity': recent_entry.get('outdoor_humidity'),
                                'solar_irradiance': recent_entry.get('solar_irradiance', 0)
                            }
                        }
                # Migrate to new format
                self._weather_cache = {'historical_entries': self._weather_cache if isinstance(self._weather_cache, list) else []}
            else:
                logger.warning(f"Unknown weather cache format: {type(self._weather_cache)}")
                
        except Exception as e:
            logger.warning(f"Error accessing weather cache: {e}")
            
        return None
            
    def _estimate_solar_irradiance(self, current_data: Dict) -> float:
        """Estimate solar irradiance from current conditions"""
        # Simplified estimation based on time, cloud cover, and sun angle
        now = datetime.now()
        hour = now.hour
        
        # Basic diurnal pattern (peak at noon)
        if 6 <= hour <= 18:
            base_irradiance = 800 * math.sin(math.pi * (hour - 6) / 12)
        else:
            base_irradiance = 0
            
        # Adjust for cloud cover if available
        if 'CloudCover' in current_data:
            cloud_factor = 1 - (current_data['CloudCover'] / 100)
            base_irradiance *= cloud_factor
            
        return base_irradiance
        
    def _calculate_solar_irradiance(self, forecast_hour: Dict) -> float:
        """Calculate expected solar irradiance for forecast hour"""
        # Parse timestamp
        timestamp = datetime.fromisoformat(forecast_hour['DateTime'].replace('Z', '+00:00'))
        hour = timestamp.hour
        
        # Basic calculation
        if 6 <= hour <= 18:
            base_irradiance = 800 * math.sin(math.pi * (hour - 6) / 12)
        else:
            base_irradiance = 0
            
        # Adjust for weather conditions
        if forecast_hour.get('HasPrecipitation'):
            base_irradiance *= 0.2  # Heavy reduction for precipitation
        elif 'CloudCover' in forecast_hour:
            cloud_factor = 1 - (forecast_hour['CloudCover'] / 100)
            base_irradiance *= cloud_factor
            
        return base_irradiance
        
    async def update_sensor(self, entity_id: str, state: Any, 
                          unit: Optional[str] = None,
                          friendly_name: Optional[str] = None):
        """Update/create a sensor in Home Assistant"""
        try:
            # For Home Assistant addons, we need to use the REST API
            # to create/update sensors
            
            attributes = {}
            if unit:
                attributes['unit_of_measurement'] = unit
            if friendly_name:
                attributes['friendly_name'] = friendly_name
                
            # Update sensor state
            data = {
                'state': str(state),
                'attributes': attributes
            }
            
            async with self.session.post(
                f"{self.base_url}/states/{entity_id}",
                headers=self.headers,
                json=data
            ) as resp:
                if resp.status in [200, 201]:
                    logger.debug(f"Updated sensor {entity_id} = {state}")
                else:
                    logger.error(f"Failed to update sensor {entity_id}: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Error updating sensor {entity_id}: {e}")
            
    async def fire_event(self, event_type: str, event_data: Dict):
        """Fire an event in Home Assistant"""
        try:
            async with self.session.post(
                f"{self.base_url}/events/{event_type}",
                headers=self.headers,
                json=event_data
            ) as resp:
                if resp.status == 200:
                    logger.debug(f"Fired event {event_type}")
                else:
                    logger.error(f"Failed to fire event {event_type}: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Error firing event: {e}")
            
    async def call_service(self, domain: str, service: str, service_data: Dict):
        """Call a Home Assistant service"""
        try:
            async with self.session.post(
                f"{self.base_url}/services/{domain}/{service}",
                headers=self.headers,
                json=service_data
            ) as resp:
                if resp.status == 200:
                    logger.debug(f"Called service {domain}.{service}")
                else:
                    logger.error(f"Failed to call service: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Error calling service: {e}")