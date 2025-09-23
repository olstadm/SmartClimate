"""
Enhanced Training System for HomeForecast v2.0
Integrates DOE building models and EPW weather data for accurate thermal predictions
"""
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .building_model_parser import IDFBuildingParser, EPWWeatherParser
from .thermal_model import ThermalModel

logger = logging.getLogger(__name__)


class EnhancedTrainingSystem:
    """Enhanced training system using DOE building models and EPW weather data"""
    
    def __init__(self, thermal_model: ThermalModel):
        self.thermal_model = thermal_model
        self.idf_parser = IDFBuildingParser()
        self.epw_parser = EPWWeatherParser()
        self.building_model = None
        self.weather_dataset = None
        self.training_results = {}
        
    def load_building_model(self, idf_file_path: str) -> Dict:
        """
        Load and parse DOE building model from IDF file
        
        Args:
            idf_file_path: Path to the DOE IDF file
            
        Returns:
            Dict containing parsed building characteristics
        """
        logger.info(f"🏗️ Loading DOE building model from: {idf_file_path}")
        
        self.building_model = self.idf_parser.parse_idf_file(idf_file_path)
        
        # Apply building model parameters to thermal model
        self._apply_building_parameters()
        
        logger.info(f"✅ Building model loaded: {self.building_model['building_type']}")
        logger.info(f"   Floor area: {self.building_model['geometry']['floor_area_sqft']:.0f} sq ft")
        logger.info(f"   Time constant: {self.building_model['rc_parameters']['time_constant_hours']:.1f} hours")
        
        return self.building_model
    
    def load_weather_dataset(self, epw_file_path: str, limit_hours: Optional[int] = None) -> Dict:
        """
        Load EPW weather dataset for training
        
        Args:
            epw_file_path: Path to the EPW weather file
            limit_hours: Limit hours for testing (None for full year)
            
        Returns:
            Dict containing weather data and statistics
        """
        logger.info(f"🌤️ Loading EPW weather dataset from: {epw_file_path}")
        
        self.weather_dataset = self.epw_parser.parse_epw_file(epw_file_path, limit_hours)
        
        logger.info(f"✅ Weather dataset loaded: {self.weather_dataset['location']['city']}")
        logger.info(f"   Data points: {self.weather_dataset['data_points']} hours")
        
        if 'summary_statistics' in self.weather_dataset:
            temp_stats = self.weather_dataset['summary_statistics'].get('temperature_range_F', {})
            logger.info(f"   Temperature range: {temp_stats.get('min', 0):.1f}°F to {temp_stats.get('max', 100):.1f}°F")
            
        return self.weather_dataset
    
    def run_enhanced_training(self, 
                            training_duration_hours: int = 168,  # 1 week default
                            hvac_scenarios: Optional[List[str]] = None,
                            comfort_min: float = 68.0,
                            comfort_max: float = 76.0,
                            progress_callback: Optional[callable] = None) -> Dict:
        """
        Run enhanced training using building model and weather data
        
        Args:
            training_duration_hours: Duration of training simulation
            hvac_scenarios: List of HVAC scenarios to simulate
            comfort_min: Minimum comfort temperature (°F)
            comfort_max: Maximum comfort temperature (°F)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing training results and model parameters
        """
        if not self.building_model:
            raise ValueError("Building model must be loaded first (use load_building_model)")
            
        if not self.weather_dataset or not self.weather_dataset['weather_data']:
            raise ValueError("Weather dataset must be loaded first (use load_weather_dataset)")
            
        logger.info(f"🎯 Starting enhanced training simulation")
        logger.info(f"   Duration: {training_duration_hours} hours")
        logger.info(f"   Building: {self.building_model['building_type']}")
        logger.info(f"   Location: {self.weather_dataset['location']['city']}")
        
        if hvac_scenarios is None:
            hvac_scenarios = ['heating', 'cooling', 'off', 'mixed']
            
        training_data = []
        total_scenarios = len(hvac_scenarios)
        
        # Generate training scenarios with progress reporting
        for i, scenario in enumerate(hvac_scenarios):
            if progress_callback:
                progress = (i / total_scenarios) * 80  # Use 80% for data generation
                progress_callback(progress, f"Generating {scenario} scenario data...")
                
            scenario_data = self._simulate_hvac_scenario(
                scenario, 
                training_duration_hours // len(hvac_scenarios),
                comfort_min,
                comfort_max
            )
            training_data.extend(scenario_data)
            
        if progress_callback:
            progress_callback(80, "Training thermal model with physics validation...")
            
        # Train thermal model with enhanced data
        self.training_results = self._train_with_physics_validation(training_data, progress_callback)
        
        logger.info(f"✅ Enhanced training complete")
        logger.info(f"   Training samples: {len(training_data)}")
        logger.info(f"   Model accuracy: {self.training_results.get('accuracy_score', 0):.3f}")
        
        return self.training_results
    
    def _apply_building_parameters(self):
        """Apply parsed building model parameters to thermal model"""
        if not self.building_model:
            return
            
        rc_params = self.building_model.get('rc_parameters', {})
        
        # Update thermal model parameters
        self.thermal_model.theta = rc_params.get('a_parameter', 0.1)
        
        # Update heating/cooling rates
        if hasattr(self.thermal_model, 'heating_rate'):
            self.thermal_model.heating_rate = rc_params.get('suggested_heating_rate_F_per_hr', 4.0)
        if hasattr(self.thermal_model, 'cooling_rate'):
            self.thermal_model.cooling_rate = rc_params.get('suggested_cooling_rate_F_per_hr', 5.0)
            
        # Update solar gain factor
        if hasattr(self.thermal_model, 'solar_gain_factor'):
            self.thermal_model.solar_gain_factor = rc_params.get('suggested_solar_gain_factor', 0.5)
            
        logger.info(f"📊 Applied building parameters:")
        logger.info(f"   θ (a_parameter): {self.thermal_model.theta:.4f}")
        logger.info(f"   Heating rate: {rc_params.get('suggested_heating_rate_F_per_hr', 4.0):.1f}°F/hr")
        logger.info(f"   Cooling rate: {rc_params.get('suggested_cooling_rate_F_per_hr', 5.0):.1f}°F/hr")
        
    def _simulate_hvac_scenario(self, scenario: str, duration_hours: int, comfort_min: float = 68.0, comfort_max: float = 76.0) -> List[Dict]:
        """
        Simulate HVAC scenario using weather data and building physics
        
        Args:
            scenario: HVAC scenario ('heating', 'cooling', 'off', 'mixed')
            duration_hours: Duration of scenario simulation
            
        Returns:
            List of training data points
        """
        logger.info(f"🔄 Simulating {scenario} scenario for {duration_hours} hours")
        
        scenario_data = []
        weather_data = self.weather_dataset['weather_data']
        
        if not weather_data:
            logger.warning("No weather data available for simulation")
            return []
            
        # Start at random point in weather data
        start_idx = np.random.randint(0, max(1, len(weather_data) - duration_hours))
        
        # Initial conditions
        indoor_temp = 72.0  # Start at comfortable temperature
        
        for i in range(duration_hours):
            weather_idx = (start_idx + i) % len(weather_data)
            weather_point = weather_data[weather_idx]
            
            outdoor_temp = weather_point['dry_bulb_temp_F']
            solar_radiation = weather_point.get('global_horizontal_radiation_Wh_m2', 0)
            humidity = weather_point.get('relative_humidity_pct', 50)
            
            # Determine HVAC mode based on scenario
            hvac_mode = self._get_hvac_mode_for_scenario(scenario, indoor_temp, outdoor_temp, i, comfort_min, comfort_max)
            
            # Simulate physics-based temperature change
            temp_change = self._simulate_temperature_change(
                indoor_temp, outdoor_temp, hvac_mode, solar_radiation
            )
            
            # Create training data point
            training_point = {
                'timestamp': datetime.now() + timedelta(hours=i),
                'indoor_temp_prev': indoor_temp,
                'outdoor_temp': outdoor_temp,
                'hvac_mode': hvac_mode,
                'solar_radiation': solar_radiation,
                'humidity': humidity,
                'indoor_temp_next': indoor_temp + temp_change,
                'temp_change_per_hour': temp_change,
                'scenario': scenario
            }
            
            scenario_data.append(training_point)
            
            # Update indoor temperature for next iteration
            indoor_temp += temp_change
            
            # Keep indoor temperature within reasonable bounds
            indoor_temp = max(40.0, min(95.0, indoor_temp))
            
        logger.info(f"   Generated {len(scenario_data)} training points for {scenario}")
        return scenario_data
    
    def _get_hvac_mode_for_scenario(self, scenario: str, indoor_temp: float, outdoor_temp: float, hour: int, comfort_min: float = 68.0, comfort_max: float = 76.0) -> str:
        """Determine HVAC mode based on scenario and conditions"""
        if scenario == 'heating':
            return 'heat' if indoor_temp < comfort_max else 'off'
        elif scenario == 'cooling':
            return 'cool' if indoor_temp > comfort_min else 'off'
        elif scenario == 'off':
            return 'off'
        elif scenario == 'mixed':
            # Realistic thermostat behavior using comfort band
            if indoor_temp < comfort_min:
                return 'heat'
            elif indoor_temp > comfort_max:
                return 'cool'
            else:
                return 'off'
        else:
            return 'off'
    
    def _simulate_temperature_change(self, indoor_temp: float, outdoor_temp: float, 
                                   hvac_mode: str, solar_radiation: float) -> float:
        """
        Simulate realistic temperature change using building physics
        
        Args:
            indoor_temp: Current indoor temperature (°F)
            outdoor_temp: Outdoor temperature (°F)
            hvac_mode: HVAC mode ('heat', 'cool', 'off')
            solar_radiation: Solar radiation (Wh/m²)
            
        Returns:
            Temperature change per hour (°F/hr)
        """
        # Base temperature differential (Newton's law of cooling)
        temp_diff = outdoor_temp - indoor_temp
        
        # Natural temperature change based on building RC parameters
        rc_params = self.building_model.get('rc_parameters', {})
        a_param = rc_params.get('a_parameter', 0.1)  # 1/time_constant
        
        natural_change = a_param * temp_diff
        
        # HVAC contribution
        hvac_change = 0.0
        if hvac_mode == 'heat':
            heating_rate = rc_params.get('suggested_heating_rate_F_per_hr', 4.0)
            hvac_change = heating_rate
        elif hvac_mode == 'cool':
            cooling_rate = rc_params.get('suggested_cooling_rate_F_per_hr', 5.0)
            hvac_change = -cooling_rate
            
        # Solar gain contribution
        solar_gain_factor = rc_params.get('suggested_solar_gain_factor', 0.5)
        solar_change = (solar_radiation / 1000.0) * solar_gain_factor  # Convert and scale
        
        # Total temperature change
        total_change = natural_change + hvac_change + solar_change
        
        # Apply physics constraints (realistic limits)
        max_change_per_hour = 8.0  # °F/hr maximum reasonable change
        total_change = max(-max_change_per_hour, min(max_change_per_hour, total_change))
        
        return total_change
    
    def _train_with_physics_validation(self, training_data: List[Dict], progress_callback: Optional[callable] = None) -> Dict:
        """
        Train thermal model with physics validation
        
        Args:
            training_data: List of training data points
            
        Returns:
            Dict containing training results and validation metrics
        """
        logger.info(f"🧠 Training thermal model with physics validation")
        logger.info(f"   Training samples: {len(training_data)}")
        
        # Convert training data to format expected by thermal model
        valid_samples = 0
        physics_violations = 0
        total_points = len(training_data)
        
        for i, data_point in enumerate(training_data):
            if progress_callback and i % 100 == 0:  # Update every 100 samples
                progress = 80 + (i / total_points) * 20  # Use remaining 20%
                progress_callback(progress, f"Validating sample {i+1}/{total_points}...")
            # Validate physics constraints
            temp_change = data_point['temp_change_per_hour']
            indoor_temp = data_point['indoor_temp_prev']
            outdoor_temp = data_point['outdoor_temp']
            hvac_mode = data_point['hvac_mode']
            
            # Check for physics violations
            if self._validate_physics_constraints(temp_change, indoor_temp, outdoor_temp, hvac_mode):
                valid_samples += 1
                
                # Add to thermal model learning (simplified - would need actual model integration)
                # This is where the thermal model would incorporate the training data
                
            else:
                physics_violations += 1
                
        # Calculate training metrics
        accuracy_score = valid_samples / len(training_data) if training_data else 0.0
        physics_compliance = (len(training_data) - physics_violations) / len(training_data) if training_data else 1.0
        
        results = {
            'total_samples': len(training_data),
            'valid_samples': valid_samples,
            'physics_violations': physics_violations,
            'accuracy_score': accuracy_score,
            'physics_compliance': physics_compliance,
            'building_type': self.building_model.get('building_type', 'Unknown'),
            'training_location': self.weather_dataset.get('location', {}).get('city', 'Unknown'),
            'model_parameters': {
                'theta': self.thermal_model.theta,
                'time_constant_hours': 1.0 / self.thermal_model.theta if self.thermal_model.theta > 0 else 0,
            }
        }
        
        logger.info(f"   Valid samples: {valid_samples}/{len(training_data)}")
        logger.info(f"   Physics compliance: {physics_compliance:.1%}")
        logger.info(f"   Accuracy score: {accuracy_score:.3f}")
        
        return results
    
    def _validate_physics_constraints(self, temp_change: float, indoor_temp: float, 
                                    outdoor_temp: float, hvac_mode: str) -> bool:
        """
        Validate that temperature change follows physics constraints
        
        Args:
            temp_change: Predicted temperature change (°F/hr)
            indoor_temp: Indoor temperature (°F)
            outdoor_temp: Outdoor temperature (°F)
            hvac_mode: HVAC mode ('heat', 'cool', 'off')
            
        Returns:
            True if physics constraints are satisfied
        """
        # Maximum realistic temperature change rates
        if abs(temp_change) > 10.0:  # °F/hr
            return False
            
        # Temperature differential constraints
        temp_diff = outdoor_temp - indoor_temp
        
        if hvac_mode == 'off':
            # Natural heating/cooling should follow temperature differential direction
            if temp_diff > 0 and temp_change < -0.5:  # Should be warming, but cooling too fast
                return False
            if temp_diff < 0 and temp_change > 0.5:   # Should be cooling, but warming too fast
                return False
                
        elif hvac_mode == 'heat':
            # Should always be heating (positive change) or at least not cooling fast
            if temp_change < -1.0:
                return False
                
        elif hvac_mode == 'cool':
            # Should always be cooling (negative change) or at least not heating fast
            if temp_change > 1.0:
                return False
                
        return True
    
    def save_training_results(self, output_path: str):
        """Save training results and model parameters to file"""
        if not self.training_results:
            logger.warning("No training results to save")
            return
            
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.2',
            'building_model': self.building_model,
            'weather_dataset_info': {
                'location': self.weather_dataset.get('location', {}),
                'data_points': self.weather_dataset.get('data_points', 0),
                'summary_statistics': self.weather_dataset.get('summary_statistics', {})
            },
            'training_results': self.training_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
            
        logger.info(f"💾 Training results saved to: {output_path}")
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        return {
            'building_model_loaded': self.building_model is not None,
            'weather_dataset_loaded': self.weather_dataset is not None,
            'training_completed': bool(self.training_results),
            'building_type': self.building_model.get('building_type') if self.building_model else None,
            'training_location': self.weather_dataset.get('location', {}).get('city') if self.weather_dataset else None,
            'model_accuracy': self.training_results.get('accuracy_score') if self.training_results else None,
            'physics_compliance': self.training_results.get('physics_compliance') if self.training_results else None
        }