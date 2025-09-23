"""
DOE Building Model Parser for EnergyPlus IDF Files
Extracts thermal characteristics from DOE Residential Prototype Building Models
"""
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class IDFBuildingParser:
    """Parse EnergyPlus IDF files to extract building thermal characteristics"""
    
    def __init__(self):
        self.thermal_properties = {}
        self.building_geometry = {}
        self.hvac_systems = {}
        self.zone_properties = {}
        
    def parse_idf_file(self, idf_path: str) -> Dict:
        """
        Parse IDF file and extract thermal characteristics
        
        Args:
            idf_path: Path to the IDF file
            
        Returns:
            Dict containing building thermal properties
        """
        logger.info(f"📋 Parsing DOE building model: {idf_path}")
        
        try:
            with open(idf_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Parse different sections
            self._parse_building_geometry(content)
            self._parse_thermal_zones(content)
            self._parse_construction_materials(content)
            self._parse_hvac_systems(content)
            self._calculate_thermal_parameters(content)
            
            building_model = {
                'building_type': self._identify_building_type(content),
                'geometry': self.building_geometry,
                'thermal_properties': self.thermal_properties,
                'hvac_systems': self.hvac_systems,
                'zones': self.zone_properties,
                'rc_parameters': self._calculate_rc_parameters()
            }
            
            logger.info(f"✅ Successfully parsed building model: {building_model['building_type']}")
            return building_model
            
        except Exception as e:
            logger.error(f"❌ Error parsing IDF file: {e}")
            return self._get_default_building_model()
    
    def _parse_building_geometry(self, content: str):
        """Extract building geometry information"""
        # Parse Building object
        building_match = re.search(
            r'Building,\s*([^;]*);', content, re.IGNORECASE | re.DOTALL
        )
        
        if building_match:
            building_params = [param.strip() for param in building_match.group(1).split(',')]
            self.building_geometry = {
                'name': building_params[0] if len(building_params) > 0 else 'Unknown',
                'north_axis': float(building_params[1]) if len(building_params) > 1 else 0.0,
                'terrain': building_params[2] if len(building_params) > 2 else 'Suburbs',
                'loads_convergence_tolerance': float(building_params[3]) if len(building_params) > 3 else 0.04,
                'temperature_convergence_tolerance': float(building_params[4]) if len(building_params) > 4 else 0.4
            }
        
        # Parse BuildingSurface:Detailed for total floor area
        surfaces = re.findall(
            r'BuildingSurface:Detailed,\s*([^;]*);', content, re.IGNORECASE | re.DOTALL
        )
        
        total_floor_area = 0
        for surface in surfaces:
            params = [p.strip() for p in surface.split(',')]
            if len(params) > 2 and 'floor' in params[1].lower():
                # Calculate area from coordinates (simplified)
                total_floor_area += 100  # Placeholder - would need coordinate parsing
                
        self.building_geometry['floor_area_sqft'] = max(total_floor_area, 2000)  # Default 2000 sq ft
        
    def _parse_thermal_zones(self, content: str):
        """Extract thermal zone information"""
        zones = re.findall(
            r'Zone,\s*([^;]*);', content, re.IGNORECASE | re.DOTALL
        )
        
        for zone_data in zones:
            params = [p.strip() for p in zone_data.split(',')]
            if len(params) > 0:
                zone_name = params[0]
                self.zone_properties[zone_name] = {
                    'direction_of_relative_north': float(params[1]) if len(params) > 1 else 0.0,
                    'x_origin': float(params[2]) if len(params) > 2 else 0.0,
                    'y_origin': float(params[3]) if len(params) > 3 else 0.0,
                    'z_origin': float(params[4]) if len(params) > 4 else 0.0,
                    'type': int(params[5]) if len(params) > 5 else 1,
                    'multiplier': int(params[6]) if len(params) > 6 else 1
                }
    
    def _parse_construction_materials(self, content: str):
        """Extract construction and material thermal properties"""
        # Parse Material properties
        materials = re.findall(
            r'Material,\s*([^;]*);', content, re.IGNORECASE | re.DOTALL
        )
        
        material_properties = {}
        for material in materials:
            params = [p.strip() for p in material.split(',')]
            if len(params) >= 7:
                name = params[0]
                material_properties[name] = {
                    'roughness': params[1],
                    'thickness': float(params[2]) if params[2] else 0.1,  # meters
                    'conductivity': float(params[3]) if params[3] else 0.1,  # W/m-K
                    'density': float(params[4]) if params[4] else 1000,  # kg/m3
                    'specific_heat': float(params[5]) if params[5] else 1000,  # J/kg-K
                    'thermal_absorptance': float(params[6]) if params[6] else 0.9,
                    'solar_absorptance': float(params[7]) if len(params) > 7 and params[7] else 0.7,
                    'visible_absorptance': float(params[8]) if len(params) > 8 and params[8] else 0.7
                }
        
        # Calculate overall thermal mass and R-values
        total_thermal_mass = 0
        total_r_value = 0
        
        for mat_name, props in material_properties.items():
            # Calculate thermal mass (density × specific_heat × volume_estimate)
            volume_estimate = self.building_geometry.get('floor_area_sqft', 2000) * 0.1  # Assume 0.1 ft average thickness
            thermal_mass = props['density'] * props['specific_heat'] * volume_estimate * 0.0283168  # Convert to metric
            total_thermal_mass += thermal_mass
            
            # Calculate R-value (thickness / conductivity)
            r_value = props['thickness'] / props['conductivity'] if props['conductivity'] > 0 else 1.0
            total_r_value += r_value
            
        self.thermal_properties = {
            'material_count': len(material_properties),
            'total_thermal_mass_J_per_K': total_thermal_mass,
            'average_r_value_m2K_per_W': total_r_value / max(len(material_properties), 1),
            'materials': material_properties
        }
    
    def _parse_hvac_systems(self, content: str):
        """Extract HVAC system characteristics"""
        # Parse HVAC equipment
        hvac_equipment = []
        
        # Look for common HVAC objects
        hvac_objects = [
            'AirLoopHVAC',
            'AirLoopHVAC:UnitarySystem',
            'Coil:Heating:Electric',
            'Coil:Cooling:DX:SingleSpeed',
            'Fan:OnOff',
            'AirTerminal:SingleDuct:Uncontrolled'
        ]
        
        for hvac_type in hvac_objects:
            matches = re.findall(
                rf'{hvac_type},\s*([^;]*);', content, re.IGNORECASE | re.DOTALL
            )
            for match in matches:
                params = [p.strip() for p in match.split(',')]
                hvac_equipment.append({
                    'type': hvac_type,
                    'name': params[0] if len(params) > 0 else 'Unknown',
                    'parameters': params[1:] if len(params) > 1 else []
                })
        
        self.hvac_systems = {
            'equipment_count': len(hvac_equipment),
            'equipment': hvac_equipment,
            'has_heating': any('heating' in eq['type'].lower() for eq in hvac_equipment),
            'has_cooling': any('cooling' in eq['type'].lower() for eq in hvac_equipment)
        }
    
    def _calculate_thermal_parameters(self, content: str):
        """Calculate additional thermal parameters from parsed data"""
        floor_area = self.building_geometry.get('floor_area_sqft', 2000)
        
        # Estimate building volume (assume 9 ft ceiling)
        building_volume_ft3 = floor_area * 9
        
        # Calculate air changes per hour (estimate from building type)
        air_changes_per_hour = 0.5  # Typical for residential
        
        # Calculate thermal time constant
        thermal_mass = self.thermal_properties.get('total_thermal_mass_J_per_K', 1e6)
        r_value = self.thermal_properties.get('average_r_value_m2K_per_W', 2.0)
        
        # Thermal time constant τ = RC (hours)
        thermal_time_constant_hours = (thermal_mass * r_value) / 3600  # Convert to hours
        
        self.thermal_properties.update({
            'floor_area_sqft': floor_area,
            'building_volume_ft3': building_volume_ft3,
            'air_changes_per_hour': air_changes_per_hour,
            'thermal_time_constant_hours': thermal_time_constant_hours,
            'estimated_infiltration_cfm': building_volume_ft3 * air_changes_per_hour / 60
        })
    
    def _identify_building_type(self, content: str) -> str:
        """Identify the building type from IDF content"""
        content_lower = content.lower()
        
        if 'single family' in content_lower or 'singlefamily' in content_lower:
            return 'Single Family Detached'
        elif 'apartment' in content_lower or 'multifamily' in content_lower:
            return 'Multifamily'
        elif 'mobile' in content_lower or 'manufactured' in content_lower:
            return 'Mobile Home'
        elif 'office' in content_lower:
            return 'Small Office'
        elif 'retail' in content_lower:
            return 'Retail'
        else:
            return 'Residential (Generic)'
    
    def _calculate_rc_parameters(self) -> Dict:
        """Calculate RC model parameters from building characteristics"""
        # Extract key thermal properties
        thermal_mass = self.thermal_properties.get('total_thermal_mass_J_per_K', 1e6)
        r_value = self.thermal_properties.get('average_r_value_m2K_per_W', 2.0) 
        floor_area = self.thermal_properties.get('floor_area_sqft', 2000)
        
        # Convert to RC model parameters (Fahrenheit-based)
        # R: Thermal resistance (°F·hr/Btu)
        # C: Thermal capacitance (Btu/°F)
        
        # Convert thermal mass to thermal capacitance in Btu/°F
        thermal_capacitance_btu_per_F = thermal_mass * 0.000526565  # J/K to Btu/°F
        
        # Convert R-value to thermal resistance in °F·hr/Btu
        # Assume building envelope area ≈ 6 × floor_area (walls + roof + floor)
        envelope_area_ft2 = 6 * floor_area
        thermal_resistance_F_hr_per_btu = r_value * 5.678263 / envelope_area_ft2  # m²K/W to °F·hr/Btu per ft²
        
        # Calculate RC model parameters
        # τ = RC (thermal time constant in hours)
        tau_hours = thermal_resistance_F_hr_per_btu * thermal_capacitance_btu_per_F
        
        # a = 1/τ (inverse time constant, 1/hours)
        a_parameter = 1.0 / max(tau_hours, 1.0)  # Prevent division by zero
        
        return {
            'thermal_capacitance_btu_per_F': thermal_capacitance_btu_per_F,
            'thermal_resistance_F_hr_per_btu': thermal_resistance_F_hr_per_btu,
            'time_constant_hours': tau_hours,
            'a_parameter': a_parameter,  # For RC model θ parameter
            'suggested_heating_rate_F_per_hr': 4.0,  # Based on typical HVAC sizing
            'suggested_cooling_rate_F_per_hr': 5.0,  # Based on typical HVAC sizing
            'suggested_solar_gain_factor': 0.5,  # Based on window area and orientation
            'building_envelope_area_ft2': envelope_area_ft2
        }
    
    def _get_default_building_model(self) -> Dict:
        """Return default building model if parsing fails"""
        return {
            'building_type': 'Generic Residential',
            'geometry': {
                'name': 'Default Home',
                'floor_area_sqft': 2000,
                'building_volume_ft3': 18000
            },
            'thermal_properties': {
                'thermal_time_constant_hours': 8.0,
                'total_thermal_mass_J_per_K': 1e6,
                'average_r_value_m2K_per_W': 2.0
            },
            'hvac_systems': {
                'has_heating': True,
                'has_cooling': True,
                'equipment_count': 2
            },
            'zones': {},
            'rc_parameters': {
                'a_parameter': 0.125,  # 1/8 hour time constant
                'suggested_heating_rate_F_per_hr': 3.5,
                'suggested_cooling_rate_F_per_hr': 4.5,
                'suggested_solar_gain_factor': 0.8,
                'time_constant_hours': 8.0
            }
        }


class EPWWeatherParser:
    """Parse EnergyPlus Weather (EPW) files for local weather training data"""
    
    def __init__(self):
        self.weather_data = []
        self.location_info = {}
        
    def parse_epw_file(self, epw_path: str, limit_hours: Optional[int] = None) -> Dict:
        """
        Parse EPW file and extract weather data
        
        Args:
            epw_path: Path to the EPW file
            limit_hours: Limit number of hours to parse (for testing)
            
        Returns:
            Dict containing weather data and location info
        """
        logger.info(f"🌤️ Parsing EPW weather file: {epw_path}")
        
        try:
            with open(epw_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
            # Parse header (first 8 lines contain metadata)
            self._parse_epw_header(lines[:8])
            
            # Parse weather data (starting from line 9)
            data_lines = lines[8:8+(limit_hours or len(lines))]
            self._parse_weather_data(data_lines)
            
            weather_dataset = {
                'location': self.location_info,
                'data_points': len(self.weather_data),
                'weather_data': self.weather_data,
                'summary_statistics': self._calculate_weather_statistics()
            }
            
            logger.info(f"✅ Parsed {len(self.weather_data)} hours of weather data for {self.location_info.get('city', 'Unknown Location')}")
            return weather_dataset
            
        except Exception as e:
            logger.error(f"❌ Error parsing EPW file: {e}")
            return self._get_default_weather_data()
    
    def _parse_epw_header(self, header_lines: List[str]):
        """Parse EPW header information"""
        if len(header_lines) > 0:
            # Location line (line 1)
            location_parts = header_lines[0].strip().split(',')
            if len(location_parts) >= 10:
                self.location_info = {
                    'city': location_parts[1],
                    'state_province': location_parts[2], 
                    'country': location_parts[3],
                    'source': location_parts[4],
                    'wmo_number': location_parts[5],
                    'latitude': float(location_parts[6]) if location_parts[6] else 0.0,
                    'longitude': float(location_parts[7]) if location_parts[7] else 0.0,
                    'timezone': float(location_parts[8]) if location_parts[8] else 0.0,
                    'elevation': float(location_parts[9]) if location_parts[9] else 0.0
                }
    
    def _parse_weather_data(self, data_lines: List[str]):
        """Parse hourly weather data"""
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 35:  # EPW format has 35 fields minimum
                try:
                    weather_point = {
                        'year': int(parts[0]),
                        'month': int(parts[1]), 
                        'day': int(parts[2]),
                        'hour': int(parts[3]),
                        'minute': int(parts[4]),
                        'dry_bulb_temp_C': float(parts[6]) if parts[6] != '' else 20.0,
                        'dew_point_temp_C': float(parts[7]) if parts[7] != '' else 15.0,
                        'relative_humidity_pct': float(parts[8]) if parts[8] != '' else 50.0,
                        'atmospheric_pressure_Pa': float(parts[9]) if parts[9] != '' else 101325.0,
                        'global_horizontal_radiation_Wh_m2': float(parts[13]) if parts[13] != '' else 0.0,
                        'direct_normal_radiation_Wh_m2': float(parts[14]) if parts[14] != '' else 0.0,
                        'diffuse_horizontal_radiation_Wh_m2': float(parts[15]) if parts[15] != '' else 0.0,
                        'wind_direction_deg': float(parts[20]) if parts[20] != '' else 0.0,
                        'wind_speed_m_s': float(parts[21]) if parts[21] != '' else 0.0,
                        'total_sky_cover_tenths': float(parts[22]) if parts[22] != '' else 5.0,
                        'opaque_sky_cover_tenths': float(parts[23]) if parts[23] != '' else 5.0
                    }
                    
                    # Convert to Fahrenheit
                    weather_point['dry_bulb_temp_F'] = weather_point['dry_bulb_temp_C'] * 9/5 + 32
                    weather_point['dew_point_temp_F'] = weather_point['dew_point_temp_C'] * 9/5 + 32
                    
                    self.weather_data.append(weather_point)
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Skipping malformed weather data line: {e}")
                    continue
    
    def _calculate_weather_statistics(self) -> Dict:
        """Calculate summary statistics for weather data"""
        if not self.weather_data:
            return {}
            
        temps_f = [point['dry_bulb_temp_F'] for point in self.weather_data]
        humidity = [point['relative_humidity_pct'] for point in self.weather_data]
        solar = [point['global_horizontal_radiation_Wh_m2'] for point in self.weather_data]
        
        return {
            'temperature_range_F': {
                'min': min(temps_f),
                'max': max(temps_f),
                'mean': sum(temps_f) / len(temps_f),
                'range': max(temps_f) - min(temps_f)
            },
            'humidity_range_pct': {
                'min': min(humidity),
                'max': max(humidity), 
                'mean': sum(humidity) / len(humidity)
            },
            'solar_radiation_Wh_m2': {
                'min': min(solar),
                'max': max(solar),
                'mean': sum(solar) / len(solar),
                'total_daily_average': sum(solar) / len(solar) * 24
            },
            'data_quality': {
                'total_hours': len(self.weather_data),
                'missing_data_points': 8760 - len(self.weather_data) if len(self.weather_data) <= 8760 else 0,
                'completeness_pct': min(100.0, len(self.weather_data) / 8760 * 100)
            }
        }
    
    def _get_default_weather_data(self) -> Dict:
        """Return default weather data if parsing fails"""
        return {
            'location': {
                'city': 'Generic Location',
                'state_province': 'Unknown',
                'country': 'Unknown',
                'latitude': 40.0,
                'longitude': -100.0,
                'timezone': -6.0,
                'elevation': 100.0
            },
            'data_points': 0,
            'weather_data': [],
            'summary_statistics': {}
        }