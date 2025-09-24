# HomeForecast Changelog

All notable changes to HomeForecast will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.1] - 2025-09-23

### Critical Hotfixes
- **Home Assistant Client**: Fixed "Timeout context manager should be used inside a task" errors affecting all sensor data collection
- **DataStore Integration**: Fixed AttributeError in `retroactive_physics_correction` method - corrected to use `get_recent_measurements`
- **None Value Handling**: Comprehensive fixes for `None` temperature values causing TypeError in enthalpy and trend calculations
- **Session Management**: Enhanced aiohttp session validation and automatic recovery from connection issues

### Physics Engine Improvements  
- **Smart Baseline Correction**: Enhanced baseline drift handling to prevent physics violations in natural thermal predictions
- **Advanced Diagnostics**: Added detailed logging for natural physics calculations to aid in troubleshooting
- **Robust Error Recovery**: Improved fallback mechanisms when sensor data is unavailable or corrupted

### System Stability
- **Enhanced Validation**: Added comprehensive data type checking throughout thermal model calculations
- **Graceful Degradation**: System continues operating with fallback values when sensors are offline
- **Error Context**: Improved error messages with specific context about missing data and recovery actions

## [2.2.0] - 2025-09-23recast Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-09-23

### Major Physics Engine Overhaul - Natural Thermal Behavior Isolation
- **HVAC Contamination Elimination**: Completely isolated natural thermal predictions from HVAC-trained parameters 
- **Clean Natural Physics Vector**: Added `_build_clean_natural_vector()` method that zeros out HVAC coefficients for idle predictions
- **Enhanced Physics Constraints**: Implemented strict thermodynamic rules for no-control scenarios with proper directional validation
- **Retroactive Physics Correction**: Added automatic cleaning of existing HVAC-contaminated parameters using historical natural-only data
- **Thermal Time Constant Integration**: Applied building-specific thermal characteristics from enhanced training to improve natural drift rates

### Strict Physics Validation System
- **Direction Enforcement**: Natural thermal predictions now MUST follow temperature differential (indoor → outdoor approach)
- **Rate Limiting**: Maximum natural drift rates based on thermal time constants and temperature differences  
- **Enhanced Training Validation**: Upgraded physics constraints with stricter rules for heating/cooling/natural scenarios
- **Equilibrium Handling**: Special cases for near-equilibrium conditions with minimal drift allowances
- **Absolute Bounds**: Temperature-dependent maximum rates preventing unrealistic thermal behavior

### Forecast Engine Improvements
- **Idle Trajectory Isolation**: Added extra physics validation layer specifically for idle/no-control forecasts
- **Real-time Physics Corrections**: Dynamic correction of non-physical predictions during trajectory simulation
- **Enhanced Logging**: Detailed physics violation detection and correction reporting for diagnostics
- **Control Mode Enforcement**: Guaranteed HVAC state isolation between different prediction scenarios

### Thermal Model Architecture Changes
- **Parameter Separation**: Natural thermal coefficients now cleanly separated from HVAC-specific parameters
- **Thermal Characteristics Storage**: Integration of building-specific thermal properties (time constants, max rates)
- **Automatic Model Correction**: Retroactive correction runs on initialization to clean existing contamination
- **Physics-Validated Learning**: Enhanced training results application with strict thermal constraint validation

### Bug Fixes - Critical Physics Issues Resolved
- **Non-Physical Cooling**: Fixed idle predictions showing cooling when outdoor temperature is warmer  
- **HVAC Parameter Bleeding**: Eliminated heating/cooling coefficients affecting natural thermal predictions
- **Unrealistic Drift Rates**: Bounded natural temperature change rates to physically realistic values
- **Direction Violations**: Corrected predictions that moved away from thermal equilibrium direction
- **Training Data Contamination**: Filtered HVAC-influenced data from natural thermal parameter learning

### Technical Improvements
- **Backward Compatibility**: All changes preserve existing enhanced training and HVAC prediction functionality
- **Data Integrity**: Retroactive corrections maintain historical data while cleaning contaminated parameters  
- **Performance Optimization**: Natural physics constraints applied efficiently without affecting real-time performance
- **Diagnostic Enhancement**: Comprehensive physics violation logging and correction tracking

## [2.1.2] - 2025-09-23

### Enhanced Training System Integration & UI Improvements
- **Enhanced Training Results Integration**: Physics-validated parameters from enhanced training now properly applied to thermal model predictions
- **Thermal Model Parameter Application**: Added `apply_enhanced_training_results()` method to integrate training results into ongoing RC model calculations
- **Retrain UI Functionality**: Added dynamic retrain controls with enhanced duration options (12h, 24h, 48h, 72h, 168h)
- **Dynamic UI Visibility**: Training controls and progress indicators now show/hide based on model training state
- **Model Status Display**: Added real-time model status section showing training completion and system readiness

### Technical Improvements  
- **Background Training Integration**: Enhanced training completion now triggers parameter application via update cycle
- **Physics-Validated Parameter Integration**: Training results properly integrated into RLS theta parameters with accuracy validation
- **Enhanced Training Application Flag**: Added system flag to ensure training results are applied exactly once per completion
- **API Version Consistency**: Unified all version references to v2.1.2 across entire codebase
- **Real-time Progress Tracking**: Maintained 1-second polling with comprehensive training status endpoints

### Bug Fixes
- **Training Results Application**: Resolved issue where 99.1% training accuracy wasn't improving actual predictions
- **Parameter Integration Gap**: Fixed disconnect between successful enhanced training and ongoing thermal model predictions
- **Version Consistency**: Updated all version references from mixed 2.0.3/2.1.0/2.1.1 to consistent 2.1.2

## [1.8.22] - 2025-09-23

### Enhanced Chart Data Structure API Support
- **Fixed Missing Chart Data**: Resolved "No current trajectory data available for energy chart" error
- **Enhanced Data Structure Creation**: Added automatic conversion from legacy to new chart format in API
- **Timeline Separator API**: Added proper timeline_separator metadata for "NOW" line positioning  
- **Historical Data Integration**: Creates historical_data and forecast_data sections from legacy trajectory data
- **Improved Chart Population**: Ensures all chart sections (historical/forecast) receive proper data

### Technical Improvements
- **API Data Conversion**: Automatically creates enhanced chart structure when legacy format detected
- **Current Time Index Usage**: Properly separates historical from forecast data using current_time_index
- **Enhanced Logging**: Added detailed logging for chart data structure creation and validation
- **Data Structure Validation**: Added checks for enhanced vs legacy data format compatibility
- **Timeline Metadata**: Includes separator positioning information for visual "NOW" line placement

### Chart Data Structure
- **Historical Section**: actual_outdoor_temp, actual_indoor_temp, actual_hvac_mode from trajectory data
- **Forecast Section**: forecasted_outdoor_temp, projected_indoor_with_hvac, projected_indoor_no_hvac
- **Timeline Separator**: historical_end_index, forecast_start_index, separator_timestamp for "NOW" line

## [1.8.21] - 2025-09-23

### Dashboard Error Fixes & Data Structure Compatibility
- **Fixed Energy Chart Crash**: Resolved `TypeError: Cannot read properties of undefined (reading 'filter')` 
- **Enhanced Data Structure Handling**: Added compatibility for new chart data structure in energy chart
- **Improved Error Handling**: Added proper null checks and fallback data handling
- **Trajectory Data Synthesis**: Creates synthetic trajectory data from new forecast structure when needed
- **Runtime Calculation Fix**: Added safe division by zero handling for HVAC runtime calculations

### Technical Details
- Fixed `updateEnergyChart` method to handle both legacy and new data structures
- Added proper validation for `controlled_trajectory` and `current_trajectory` data
- Implemented fallback logic when trajectory data unavailable from new chart structure
- Enhanced error logging for better debugging of data structure issues

## [1.8.20] - 2025-09-23

### Complete Chart Data Visualization & Timeline Separation
- **Enhanced Historical Section Display**:
  - Historical Actual Outdoor Temperature (from sensors) - solid brown line
  - Historical Cached Forecasted Outdoor (from weather cache) - dashed gray line  
  - Historical Actual Indoor Temperature (from sensors) - solid blue line
  - HVAC State visualization (if applicable)

- **Enhanced Forecast Section Display**:
  - Forecasted Outdoor Temperature - dashed green line
  - Forecasted Indoor (Smart HVAC) - dashed blue line
  - Forecasted Indoor (No Control) - long-dashed orange line
  - Forecasted HVAC State visualization

- **Visual Timeline Separation**:
  - Added prominent vertical "NOW" line with 🕒 emoji indicator
  - Clear visual separation between historical facts and future predictions
  - Automatic positioning based on timeline_separator metadata

- **Improved Legacy Fallback**:
  - Enhanced support for historical weather data in legacy mode
  - Better data structure handling when new format unavailable
  - Maintains visual consistency across data formats

### Chart Visual Improvements
- **Distinct Line Styles**: Solid lines for actual data, various dashed patterns for predictions
- **Enhanced Colors**: Improved color scheme for better data distinction
- **Better Point Markers**: Different sized points for different data types
- **Timeline Annotations**: Clear "NOW" separator with descriptive labeling

## [1.8.19] - 2025-09-23

### Dashboard & Timestamp Parsing Fixes
- **Fixed JavaScript Syntax Error**: Resolved `Unexpected token ','` preventing dashboard from loading
- **Enhanced Timestamp Parsing**: Fixed parsing of timestamps with microseconds from database
- **Dashboard Chart Display**: Removed orphaned JavaScript code causing syntax errors
- **Historical Data Retrieval**: Improved timestamp format handling for sensor data integration

### Technical Details
- Cleaned up duplicate/orphaned JavaScript code in dashboard chart generation
- Added proper microsecond parsing support for database timestamps (`%Y-%m-%d %H:%M:%S.%f`)
- Fallback parsing for timestamps without microseconds for backward compatibility
- Eliminated "Failed to parse timestamp" warnings in forecast engine logs

## [1.8.18] - 2025-09-23

### Critical Bug Fix
- **Fixed Forecast Engine Crash**: Resolved `TypeError: object of type 'NoneType' has no len()` in forecast generation
- **Missing Return Statement**: Added missing `return filled_series` in `_fill_time_gaps()` method
- **Forecast Generation Recovery**: System now properly generates forecasts without crashing during time gap filling

### Technical Details
- The `_fill_time_gaps()` method was missing a return statement, causing it to return `None` 
- This caused the forecast engine to crash when trying to get the length of the time series
- Added proper return statement to ensure filled time series is returned correctly

## [1.8.17] - 2025-09-23

### Historical Sensor Data Integration & Visual Timeline Separation
- **Actual Historical Data Display**: Charts now show real sensor measurements from data store
  - Historical outdoor temperatures from actual weather sensors
  - Historical indoor temperatures from Home Assistant sensors
  - Historical HVAC states from actual thermostat operation
- **Enhanced Chart Data Structure**: Implemented proper historical_data/forecast_data separation
- **Timeline Visual Indicators**: Added clear separation markers between historical and forecast periods
- **Smart Data Alignment**: Historical sensor data intelligently matched to chart timestamps within 15-minute windows
- **Visual Chart Improvements**:
  - Solid lines for actual historical sensor data
  - Dashed lines for weather forecasts and temperature projections
  - Clear visual distinction between "what happened" vs "what's predicted"
- **Fallback Data Handling**: Graceful handling when historical sensor data unavailable

### Enhanced Backend Data Management
- **Historical Sensor Retrieval**: New `_get_historical_sensor_data()` method for accurate chart data
- **Timeline Separator Metadata**: Added timeline separation indicators for frontend rendering
- **12-Hour Historical Window**: Retrieves sufficient historical data to populate 6-hour chart display
- **Data Store Integration**: Leverages existing measurements table for actual temperature/HVAC history

### Fixed
- **Chart Historical Data**: Eliminated use of projected data in historical 6-hour timeline
- **Visual Confusion**: Clear indicators now separate "actual past" from "predicted future"
- **Data Accuracy**: Charts display real sensor measurements instead of interpolated projections

## [1.8.16] - 2025-09-23

### Major Chart & Control Logic Overhaul
- **Full Comfort Band Operation**: Now uses complete 62-80°F comfort range instead of centering on setpoint
- **Proper Historical/Forecast Chart Structure**: Restructured chart data for clear separation:
  
  **Historical Hours (Previous 6 hours):**
  - Projected Outdoor Temperature (historical weather data)
  - Actual Outdoor Temperature 
  - Actual Indoor Temperature
  - Actual HVAC Mode

  **Forecasted Hours (Next 12 hours):**
  - Forecasted Outdoor Temperature
  - Projected Indoor Temperature w/ Climate Control Running
  - Projected Indoor Temperature w/o Climate Control Running
  - Projected HVAC Mode

- **Enhanced HVAC Control Logic**: 
  - Heat when approaching 63°F (comfort_min + 1°F)
  - Cool when approaching 79°F (comfort_max - 1°F) 
  - Respects full comfort band for energy efficiency
  - No more predictions in historical 6-hour window

### Fixed
- **Comfort Band Usage**: System now properly uses 62-80°F range instead of tight setpoint control
- **Historical Data Contamination**: Eliminated predictions appearing in past 6-hour timeline
- **Energy Efficiency**: HVAC operates across comfort band rather than constantly targeting single setpoint

## [1.8.15] - 2025-09-23

### Enhanced Data Visualization & Accuracy
- **Historical vs Forecast Separation**: Charts now properly separate 6-hour historical data from predictions
- **Enhanced Climate Insights**: Improved accuracy for thermostat in 'cool' mode with idle action
- **Setpoint-Based Predictions**: Climate insights now use actual thermostat setpoint (80°F) instead of comfort ranges
- **Context-Aware HVAC Logic**: Better prediction of when cooling will activate based on setpoint + deadband
- **Historical Data Handling**: Historical weather data shown as actual data, not predictions
- **Enhanced HVAC State Detection**: Improved detection using hvac_state, hvac_action, and direct flags

### Fixed
- **Chart Timeline Issues**: Resolved predictions appearing in historical 6-hour window
- **Climate Insights Accuracy**: Fixed interpretation of thermostat data showing mode='cool', target=80°F, action='idle'
- **Setpoint Recognition**: System now properly recognizes when temperature approaches setpoint thresholds

## [1.8.14] - 2025-09-23

### Major Accuracy Improvements
- **Setpoint-Based Forecasting**: Forecast engine now uses actual thermostat setpoint instead of hardcoded ranges
- **Smart HVAC Trajectory Validation**: "Projected Indoor Smart HVAC" line now respects thermostat setpoint ±5% tolerance
- **Proper Temperature Control**: Smart HVAC control prevents temperature from exceeding setpoint by more than specified tolerance
- **Accurate "No Control" Baseline**: "No control" line properly shows temperature without any HVAC intervention
- **Setpoint-Aware HVAC Logic**: HVAC control decisions based on actual setpoint with tighter hysteresis (0.5-1°F)
- **Enhanced Logging**: Added detailed logging for HVAC control decisions and setpoint validation

### Fixed
- **Chart Contradictions**: Resolved issue where "Smart HVAC" line trended higher than "no control" line
- **Setpoint Compliance**: HVAC now properly maintains temperature around actual thermostat setpoint
- **Temperature Overshooting**: Added safety caps to prevent unrealistic temperature excursions

## [1.8.13] - 2025-09-23

### Enhanced
- **Thermal Model Constraints**: Constrained thermal time constant to realistic 12-16 hour range
- **Context-Aware Climate Insights**: Climate insights now adapt based on current HVAC state
  - When HVAC is active: Focus on "Off Time" and remaining runtime (hide redundant "Next On Time")
  - When HVAC is idle: Focus on "Next On Time" and expected duration
  - Dynamic UI labels that change based on current context ("HVAC Status", "Action Needed", etc.)
- **Improved HVAC Detection**: Enhanced HVAC state detection using direct heating/cooling flags
- **Accurate Runtime Estimates**: Better runtime calculations based on actual temperature differences

## [1.8.12] - 2025-09-23

### Fixed
- **ML Training Data Preparation**: Fixed deprecated pandas `fillna(method='bfill')` syntax
- Improved ML data cleaning to preserve training samples with selective NaN handling
- Only drop rows with NaN in essential columns (indoor_temp, outdoor_temp, HVAC states)
- Fill missing optional features with reasonable defaults
- Added safety check to prevent training with zero samples
- Enhanced logging for ML feature preparation and data cleaning steps

## [1.8.11] - 2025-09-23

### Added
- **Manual ML Training Interface**: Added "Train ML Model" button to web UI
- Interactive confirmation dialog with training details:
  * Model type (Random Forest/Gradient Boosting)
  * Available training data points
  * Training period (30 days default)
  * Estimated training duration
- Real-time training progress indicators and success/error notifications
- Keyboard shortcuts: ESC key closes training dialog
- Click-outside-to-close modal functionality

### Enhanced
- **API Endpoints**: New `/api/ml/training-info` and `/api/ml/train` endpoints
- Training information pre-validation before starting training process
- Comprehensive error handling with user-friendly error messages
- Training success feedback with performance metrics (R², MSE, MAE)
- Automatic dashboard refresh after successful training

### User Experience
- Confirmation dialog prevents accidental training triggers
- Loading states and progress indicators during training
- Clear status messages throughout training process
- Modal interface with professional styling and animations

## [1.8.10] - 2025-09-23

### Fixed
- **Database Connection Fix**: Resolved "Cannot operate on a closed database" error in ML training data retrieval
- Fixed duplicate cursor.fetchall() call that caused database connection issues during ML model training
- Improved database connection management in get_training_data() method

### Enhanced
- **Comprehensive ML Logging**: Added detailed logging throughout ML model lifecycle
- Training process logging: data preparation, feature engineering, model training duration, performance metrics
- Prediction logging: ML correction applications, feature importance analysis, training readiness checks
- Added progress tracking for ML training data collection (shows X/100 samples needed)
- Enhanced error handling with diagnostic information for ML training failures
- Deferred initial ML training to scheduled tasks to avoid initialization conflicts

### Added
- Detailed feature importance analysis with percentage contributions
- ML correction application logging in thermal model predictions
- Training data range and statistics logging for debugging
- Enhanced training readiness checks with progress indicators

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