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
        """Format datetime for dashboard display"""
        local_time = self.get_local_time(utc_datetime)
        return local_time.strftime("%H:%M")

    def format_datetime_for_display(self, utc_datetime=None) -> str:
        """Format full datetime for dashboard display"""
        local_time = self.get_local_time(utc_datetime)
        return local_time.strftime("%m/%d %H:%M")
            
    async def get_sensor_data(self) -> Dict:
        """Collect current sensor data"""
        try:
            logger.info("=== Collecting Local Sensor Data ===")
            data = {}
            
            # Get indoor temperature (in Fahrenheit)
            indoor_temp_entity = self.config.get('indoor_temp_entity')
            logger.info(f"Reading indoor temperature from entity: {indoor_temp_entity}")
            indoor_temp = await self._get_state(indoor_temp_entity)
            data['indoor_temp'] = float(indoor_temp) if indoor_temp else 70.0
            logger.info(f"Indoor temperature: {data['indoor_temp']}°F")
            
            # Get indoor humidity
            indoor_humidity_entity = self.config.get('indoor_humidity_entity')
            logger.info(f"Reading indoor humidity from entity: {indoor_humidity_entity}")
            indoor_humidity = await self._get_state(indoor_humidity_entity)
            data['indoor_humidity'] = float(indoor_humidity) if indoor_humidity else 50.0
            logger.info(f"Indoor humidity: {data['indoor_humidity']}%")
            
            # Get outdoor temperature (optional, in Fahrenheit)
            outdoor_temp_entity = self.config.get('outdoor_temp_entity')
            if outdoor_temp_entity:
                logger.info(f"Reading outdoor temperature from entity: {outdoor_temp_entity}")
                outdoor_temp = await self._get_state(outdoor_temp_entity)
                data['outdoor_temp'] = float(outdoor_temp) if outdoor_temp else data['indoor_temp']
                logger.info(f"Local outdoor temperature: {data['outdoor_temp']}°F")
            else:
                # Will be filled from AccuWeather
                data['outdoor_temp'] = None
                logger.info("No local outdoor temperature sensor - will use AccuWeather data")
                
            # Get outdoor humidity (optional)
            outdoor_humidity_entity = self.config.get('outdoor_humidity_entity')
            if outdoor_humidity_entity:
                logger.info(f"Reading outdoor humidity from entity: {outdoor_humidity_entity}")
                outdoor_humidity = await self._get_state(outdoor_humidity_entity)
                data['outdoor_humidity'] = float(outdoor_humidity) if outdoor_humidity else 50.0
                logger.info(f"Local outdoor humidity: {data['outdoor_humidity']}%")
            else:
                data['outdoor_humidity'] = None
                logger.info("No local outdoor humidity sensor - will use AccuWeather data")
                
            # Get comprehensive thermostat data
            hvac_entity = self.config.get('hvac_entity')
            logger.info(f"Reading thermostat data from entity: {hvac_entity}")
            thermostat_data = await self.get_thermostat_data(hvac_entity)
            
            # Include thermostat data in sensor data
            data['hvac_state'] = thermostat_data['hvac_state']
            data['hvac_mode'] = thermostat_data['hvac_mode']
            data['hvac_action'] = thermostat_data['hvac_action']
            data['target_temperature'] = thermostat_data['target_temperature']
            data['thermostat_data'] = thermostat_data
            
            logger.info(f"Thermostat info - State: {data['hvac_state']}, Mode: {data['hvac_mode']}, " +
                       f"Action: {data['hvac_action']}, Target: {data['target_temperature']}°F")
            
            # Get solar irradiance if available
            solar_entity = self.config.get('solar_irradiance_entity')
            if solar_entity:
                solar = await self._get_state(solar_entity)
                data['solar_irradiance'] = float(solar) if solar else 0.0
            else:
                data['solar_irradiance'] = 0.0
                
            data['timestamp'] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting sensor data: {e}")
            raise
            
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
        """Get weather forecast from AccuWeather"""
        try:
            logger.info("=== Getting Weather Forecast from AccuWeather ===")
            
            # Get current conditions and 12-hour forecast from AccuWeather
            api_key = self.config.get('accuweather_api_key')
            location_key = self.config.get('accuweather_location_key')
            
            logger.info(f"AccuWeather API configured: Location key = {location_key[:8]}... API key = {'Yes' if api_key else 'No'}")
            
            forecast_data = {
                'hourly_forecast': [],
                'current_outdoor': {}
            }
            
            # Get current conditions
            current_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}"
            params = {'apikey': api_key, 'details': 'true'}
            
            async with self.session.get(current_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info("✅ Successfully retrieved AccuWeather current conditions")
                    if data and len(data) > 0:
                        current = data[0]
                        current_temp = current['Temperature']['Imperial']['Value']
                        current_humidity = current['RelativeHumidity']
                        logger.info(f"Current AccuWeather data: {current_temp}°F, {current_humidity}% humidity")
                        forecast_data['current_outdoor'] = {
                            'temperature': current_temp,
                            'humidity': current_humidity,
                            'solar_irradiance': self._estimate_solar_irradiance(current)
                        }
                else:
                    logger.error(f"❌ Failed to get AccuWeather current conditions: HTTP {resp.status}")
                        
            # Get 12-hour forecast
            forecast_url = f"http://dataservice.accuweather.com/forecasts/v1/hourly/12hour/{location_key}"
            
            async with self.session.get(forecast_url, params={'apikey': api_key, 'metric': 'false'}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Successfully retrieved AccuWeather 12-hour forecast ({len(data)} hours)")
                    
                    for hour in data:
                        forecast_data['hourly_forecast'].append({
                            'timestamp': datetime.fromisoformat(hour['DateTime'].replace('Z', '+00:00')),
                            'temperature': hour['Temperature']['Value'],
                            'humidity': hour.get('RelativeHumidity', 50),
                            'solar_irradiance': self._calculate_solar_irradiance(hour),
                            'precipitation_probability': hour.get('PrecipitationProbability', 0)
                        })
                    
                    # Log forecast range
                    if data:
                        first_hour = data[0]['DateTime']
                        last_hour = data[-1]['DateTime'] 
                        logger.info(f"Forecast range: {first_hour} to {last_hour}")
                else:
                    logger.error(f"❌ Failed to get AccuWeather forecast: HTTP {resp.status}")
                    
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error getting weather forecast: {e}")
            return {'hourly_forecast': [], 'current_outdoor': {}}
            
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