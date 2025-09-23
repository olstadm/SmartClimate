# HomeForecast Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.9] - 2025-09-23

### Fixed
- **Critical Bug Fix**: Resolved "string indices must be integers, not 'str'" error in weather cache system
- Unified weather cache format to prevent dict/list conflicts between caching methods
- Added proper cache initialization in HomeAssistantClient constructor
- Fixed cache access errors that occurred when different caching methods created incompatible data structures

### Enhanced
- **Climate Insights Enhancement**: Climate insights now use forecast trajectory data for predictions
- Added `_analyze_hvac_schedule_from_trajectory()` function for HVAC timing analysis based on forecast data
- Climate insights now show future HVAC needs based on temperature forecast trajectories instead of current state only

### Changed
- Reordered Climate Action Insights UI to display "Next Climate On Time" before "Climate Action Off Time"
- Consolidated weather caching mechanisms to use unified dictionary format
- Added migration handling for legacy cache formats

## [1.8.1] - 2024-12-19

### Changed
- **BREAKING CHANGE**: Replaced AccuWeather historical API integration with internal caching system
- Moved from external historical weather API calls to internal database-backed cache storage
- Updated forecast engine to use cached historical data instead of external API calls

### Added
- Internal historical weather data caching system in `ha_client.py`
  - `get_cached_historical_data()` method for retrieving 6-hour historical context
  - `_get_database_historical_data()` method for database-backed historical retrieval  
  - `cache_current_weather_data()` method for storing current conditions for future historical reference
  - `_generate_synthetic_historical_data()` method for graceful fallback when insufficient cache available
- Enhanced dashboard chart to display historical cached data alongside current and future predictions
- Graceful handling for insufficient cache scenarios (less than 6 hours of data)
- Visual differentiation between historical (cached), current, and forecast data points on charts

### Fixed
- **CRITICAL**: Resolved datetime comparison errors in forecast engine 
  - Fixed "can't compare offset-naive and offset-aware datetime" errors
  - Updated `_find_current_time_index()` method to handle timezone-aware vs naive datetime objects
  - Added timezone-aware datetime handling in comfort violation time calculations
  - Ensured consistent timezone handling across forecast generation pipeline
- Enhanced error handling and fallback mechanisms for timezone comparison issues

### Technical Details
- **Architecture Change**: Eliminated dependency on AccuWeather's paid historical data tier
- **Data Flow**: Historical context now sourced from internal measurements and cached forecasts
- **Performance**: Reduced external API dependency and improved reliability
- **Robustness**: Added comprehensive error handling for datetime operations across forecast engine

### Developer Notes
- AccuWeather historical API integration removed due to paid tier requirement
- Internal caching provides better control and eliminates external API limitations  
- Synthetic data generation ensures system continues operating with limited historical context
- Enhanced chart visualization supports historical data display with proper visual separation
- All datetime operations now use consistent timezone handling to prevent comparison errors

### Migration Notes
- No user action required - system automatically switches to internal caching
- Historical context will rebuild gradually as system operates
- Full 6-hour historical display available after 6 hours of system operation
- Fallback synthetic data ensures immediate functionality during initial operation

---

## [1.8.0] - 2024-12-18

### Added
- Enhanced thermal model with comprehensive trend analysis and prediction validation
- 18-hour forecast timeline with extended outdoor weather series
- Historical weather data integration with AccuWeather API (later removed in 1.8.1)
- Advanced prediction confidence calculation with trend-based validation
- Temperature trend analysis with coupling factor calculations
- JSON serialization framework with comprehensive data type handling

### Fixed
- JSON serialization errors with boolean and numpy data types
- NumPy compatibility issues across all API endpoints
- Enhanced error handling for API responses

### Technical Details
- Implemented `analyze_temperature_trends()` in thermal model
- Added `validate_prediction_against_trends()` for intelligent prediction corrections
- Created comprehensive data sanitization system for API responses
- Enhanced forecast engine with trend validation and extended timeline support

---

## [1.7.1] - 2024-12-17

### Fixed
- Critical JSON serialization errors in API endpoints
- Boolean data type handling in status and forecast APIs  
- Enhanced error reporting and data validation

---

## [1.6.1] - 2024-12-16

### Added
- Initial thermal forecasting system with RC model
- Home Assistant integration for sensor data collection
- AccuWeather API integration for weather forecasts
- Basic comfort analysis and HVAC control recommendations
- Web dashboard with real-time temperature monitoring
- SQLite data storage for historical analysis

### Features
- Smart HVAC scheduling based on thermal predictions
- Indoor temperature forecasting with outdoor weather context
- Energy efficiency analysis and recommendations
- Real-time dashboard with interactive charts
- Historical data collection and trend analysis