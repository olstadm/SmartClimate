# HomeForecast - Smart Thermal Forecasting for Home Assistant

HomeForecast is a Home Assistant addon that predicts your home's temperature changes over the next 12 hours, helping you optimize HVAC usage for comfort and efficiency.

## Features

- **Physics-based thermal modeling** - Uses RC (Resistance-Capacitance) model to simulate your home's thermal dynamics
- **Adaptive learning** - Continuously updates model parameters using Recursive Least Squares (RLS) algorithm
- **12-hour forecasting** - Predicts indoor temperature trajectories under different HVAC control scenarios
- **Smart HVAC recommendations** - Calculates optimal start/stop times to maintain comfort while minimizing runtime
- **Weather integration** - Incorporates AccuWeather forecasts for accurate predictions
- **Optional ML enhancement** - Machine learning corrects for patterns the physics model might miss
- **Local processing** - All calculations run on your hardware for privacy
- **Rich sensor data** - Publishes forecast data, model parameters, and recommendations as HA sensors

## How It Works

1. **Data Collection** - Monitors indoor/outdoor temperature, humidity, HVAC state, and weather forecasts
2. **Model Learning** - Updates thermal model parameters in real-time based on observed behavior
3. **Forecast Generation** - Simulates temperature evolution for idle, current, and smart control scenarios
4. **Comfort Analysis** - Determines when temperatures will exceed comfort bounds and optimal HVAC timing
5. **Sensor Publishing** - Makes all predictions and recommendations available in Home Assistant

## Installation

1. Add the HomeForecast repository to your Home Assistant addon store:
   - Navigate to **Supervisor** → **Add-on Store**
   - Click **⋮** → **Repositories**
   - Add: `https://github.com/olstadm/SmartClimate`

2. Find **HomeForecast** in the addon store and click **Install**

3. Configure the addon (see Configuration section)

4. Start the addon

## Configuration

```yaml
indoor_temp_entity: sensor.living_room_temperature
indoor_humidity_entity: sensor.living_room_humidity
outdoor_temp_entity: sensor.outdoor_temperature  # Optional
outdoor_humidity_entity: sensor.outdoor_humidity  # Optional
hvac_entity: climate.thermostat
accuweather_api_key: YOUR_API_KEY
accuweather_location_key: YOUR_LOCATION_KEY
comfort_min_temp: 20.0
comfort_max_temp: 24.0
forecast_horizon_hours: 12
update_interval_minutes: 5
enable_ml_correction: false
ml_retrain_days: 30
data_retention_days: 90
```

### Configuration Options

- **indoor_temp_entity** (required): Entity ID for indoor temperature sensor
- **indoor_humidity_entity** (required): Entity ID for indoor humidity sensor
- **outdoor_temp_entity** (optional): Entity ID for outdoor temperature sensor
- **outdoor_humidity_entity** (optional): Entity ID for outdoor humidity sensor
- **hvac_entity** (required): Entity ID for your HVAC/climate device
- **accuweather_api_key** (required): Your AccuWeather API key
- **accuweather_location_key** (required): AccuWeather location key for your area
- **comfort_min_temp**: Lower comfort temperature (default: 68°F)
- **comfort_max_temp**: Upper comfort temperature (default: 75°F)
- **forecast_horizon_hours**: How far to forecast (default: 12 hours)
- **update_interval_minutes**: Update frequency (default: 5 minutes)
- **enable_ml_correction**: Enable machine learning correction (default: false)
- **enable_smart_hvac_control**: Enable automated HVAC recommendations (default: false)
- **ml_retrain_days**: ML model retraining interval (default: 30 days)
- **data_retention_days**: How long to keep historical data (default: 90 days)

**Note:** All temperature readings and calculations use Fahrenheit throughout the system for consistency.

## Safety Features

### Smart HVAC Control
The `enable_smart_hvac_control` setting is **disabled by default** for safety. When disabled, the system operates in "learning mode":
- ✅ Collects data and builds thermal model
- ✅ Provides temperature forecasts  
- ✅ Shows comfort analysis
- ❌ No automated HVAC recommendations

**Recommendation:** Let the system learn for 1-2 weeks before enabling smart control to ensure accurate modeling.

## Getting AccuWeather API Credentials

1. Sign up for a free developer account at [developer.accuweather.com](https://developer.accuweather.com/)
2. Create a new app to get your API key
3. Use the [Location API](https://developer.accuweather.com/accuweather-locations-api/apis) to find your location key

## Published Sensors

HomeForecast creates the following sensors in Home Assistant:

### Thermal Model Parameters
- `sensor.homeforecast_thermal_time_constant` - How quickly your home gains/loses heat (hours)
- `sensor.homeforecast_heating_rate` - Temperature rise rate when heating (°F/hour)
- `sensor.homeforecast_cooling_rate` - Temperature drop rate when cooling (°F/hour)

### Forecast Data
- `sensor.homeforecast_forecast_12h` - Predicted temperature in 12 hours (°F)
- `sensor.homeforecast_time_to_upper_limit` - Minutes until upper comfort limit reached
- `sensor.homeforecast_time_to_lower_limit` - Minutes until lower comfort limit reached

### HVAC Recommendations
- `sensor.homeforecast_recommended_mode` - Suggested HVAC mode (heat/cool/off)
- `sensor.homeforecast_hvac_start_time` - Optimal time to start HVAC
- `sensor.homeforecast_hvac_stop_time` - Optimal time to stop HVAC

## Web Interface

Access the dashboard at: `http://homeassistant:8099`

The dashboard provides:
- Real-time system status and model parameters
- Interactive temperature forecast chart
- Comfort analysis and recommendations
- Manual update trigger

## Example Automations

### Smart HVAC Control
```yaml
automation:
  - alias: "HomeForecast Smart Heating"
    trigger:
      - platform: time_pattern
        minutes: "/5"
    condition:
      - condition: state
        entity_id: sensor.homeforecast_recommended_mode
        state: "heat"
    action:
      - service: climate.set_hvac_mode
        target:
          entity_id: climate.thermostat
        data:
          hvac_mode: heat

  - alias: "HomeForecast Stop HVAC"
    trigger:
      - platform: template
        value_template: "{{ now() >= as_datetime(states('sensor.homeforecast_hvac_stop_time')) }}"
    action:
      - service: climate.set_hvac_mode
        target:
          entity_id: climate.thermostat
        data:
          hvac_mode: "off"
```

### Comfort Warnings
```yaml
automation:
  - alias: "Temperature Warning"
    trigger:
      - platform: numeric_state
        entity_id: sensor.homeforecast_time_to_upper_limit
        below: 30
    action:
      - service: notify.mobile_app
        data:
          title: "Temperature Alert"
          message: "Home will be too warm in {{ states('sensor.homeforecast_time_to_upper_limit') }} minutes"
```

## Understanding the Model

HomeForecast uses a physics-inspired RC thermal model:

```
dT/dt = a*(T_out - T_in) + k_H*I_h + k_C*I_c + b + k_E*(h_out - h_in) + k_S*Solar
```

Where:
- `a` = 1/τ (inverse thermal time constant)
- `k_H` = Heating effectiveness
- `k_C` = Cooling effectiveness
- `b` = Baseline drift/internal gains
- `k_E` = Enthalpy (humidity) coupling
- `k_S` = Solar gain factor

The model learns these parameters from your home's actual behavior, adapting to seasonal changes and unique characteristics.

## Troubleshooting

### No forecast data
- Check sensor entities are correctly configured
- Verify AccuWeather API credentials
- Check addon logs for errors

### Poor prediction accuracy
- Allow 1-2 weeks for model to learn your home
- Ensure sensors update frequently (< 5 minutes)
- Consider enabling ML correction for complex homes

### High HVAC cycling
- Adjust comfort temperature range
- Check for sensor noise/instability
- Verify HVAC state reporting is accurate

## Support

- **Issues**: [GitHub Issues](https://github.com/olstadm/SmartClimate/issues)
- **Documentation**: [Wiki](https://github.com/olstadm/SmartClimate/wiki)
- **Community**: [Home Assistant Community](https://community.home-assistant.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.