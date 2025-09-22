"""
Data storage and persistence for HomeForecast
Handles storing sensor data, forecasts, and model parameters
"""
import json
import logging
import os
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class DataStore:
    """Handle data persistence for HomeForecast"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path("/data/homeforecast")
        self.db_path = self.data_dir / "homeforecast.db"
        self.models_dir = self.data_dir / "models"
        self.conn = None
        
    async def initialize(self):
        """Initialize data storage"""
        try:
            # Create directories
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            await self._init_database()
            
            logger.info("Data store initialized")
            
        except Exception as e:
            logger.error(f"Error initializing data store: {e}")
            raise
            
    async def _init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        cursor = self.conn.cursor()
        
        # Sensor measurements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                indoor_temp REAL,
                outdoor_temp REAL,
                indoor_humidity REAL,
                outdoor_humidity REAL,
                hvac_state TEXT,
                solar_irradiance REAL,
                residual_error REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Forecasts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                forecast_data TEXT NOT NULL,
                accuracy_metrics TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model parameters history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                performance_metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_timestamp ON forecasts(timestamp)")
        
        self.conn.commit()
        
    async def store_measurement(self, sensor_data: Dict):
        """Store a sensor measurement"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO measurements (
                    timestamp, indoor_temp, outdoor_temp,
                    indoor_humidity, outdoor_humidity,
                    hvac_state, solar_irradiance
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sensor_data['timestamp'],
                sensor_data.get('indoor_temp'),
                sensor_data.get('outdoor_temp'),
                sensor_data.get('indoor_humidity'),
                sensor_data.get('outdoor_humidity'),
                sensor_data.get('hvac_state'),
                sensor_data.get('solar_irradiance')
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing measurement: {e}")
            
    async def store_forecast(self, forecast_result: Dict):
        """Store a forecast result"""
        try:
            cursor = self.conn.cursor()
            
            # Convert forecast data to JSON
            forecast_json = json.dumps({
                'indoor_forecast': forecast_result.get('indoor_forecast', []),
                'outdoor_forecast': forecast_result.get('outdoor_forecast', []),
                'idle_trajectory': forecast_result.get('idle_trajectory', []),
                'controlled_trajectory': forecast_result.get('controlled_trajectory', []),
                'timestamps': [dt.isoformat() if hasattr(dt, 'isoformat') else str(dt) 
                              for dt in forecast_result.get('timestamps', [])]
            })
            
            # Store accuracy metrics if available
            accuracy_json = json.dumps(forecast_result.get('accuracy_metrics', {}))
            
            cursor.execute("""
                INSERT INTO forecasts (timestamp, forecast_data, accuracy_metrics)
                VALUES (?, ?, ?)
            """, (
                datetime.now(),
                forecast_json,
                accuracy_json
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing forecast: {e}")
            
    async def get_recent_measurements(self, hours: int = 24) -> List[Dict]:
        """Get recent measurements"""
        try:
            cursor = self.conn.cursor()
            
            since = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT * FROM measurements
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (since,))
            
            rows = cursor.fetchall()
            
            measurements = []
            for row in rows:
                measurements.append(dict(row))
                
            return measurements
            
        except Exception as e:
            logger.error(f"Error getting measurements: {e}")
            return []
            
    async def get_training_data(self, days: int) -> List[Dict]:
        """Get training data for ML model"""
        try:
            cursor = self.conn.cursor()
            
            since = datetime.now() - timedelta(days=days)
            
            # Get measurements with calculated residual errors
            cursor.execute("""
                SELECT 
                    m.*,
                    LAG(indoor_temp, 1) OVER (ORDER BY timestamp) as prev_temp,
                    (indoor_temp - LAG(indoor_temp, 1) OVER (ORDER BY timestamp)) /
                    ((JULIANDAY(timestamp) - JULIANDAY(LAG(timestamp, 1) OVER (ORDER BY timestamp))) * 24) as actual_rate
                FROM measurements m
                WHERE timestamp > ?
                ORDER BY timestamp
            """, (since,))
            
            rows = cursor.fetchall()
            
            training_data = []
            for row in rows:
                data = dict(row)
                # Calculate residual if we have model predictions stored
                # For now, this would come from comparing with RC model predictions
                training_data.append(data)
                
            return training_data
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return []
            
    async def save_model_parameters(self, model_name: str, parameters: Any):
        """Save model parameters"""
        try:
            # For complex objects, use pickle
            model_path = self.models_dir / f"{model_name}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(parameters, f)
                
            # Also store in database for history
            cursor = self.conn.cursor()
            
            # Extract basic info for database storage
            if isinstance(parameters, dict):
                params_json = json.dumps({
                    k: v for k, v in parameters.items()
                    if isinstance(v, (str, int, float, list, dict))
                })
            else:
                params_json = json.dumps({'type': str(type(parameters))})
                
            cursor.execute("""
                INSERT INTO model_parameters (model_name, parameters)
                VALUES (?, ?)
            """, (model_name, params_json))
            
            self.conn.commit()
            
            logger.info(f"Saved parameters for model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving model parameters: {e}")
            
    async def load_model_parameters(self, model_name: str) -> Optional[Any]:
        """Load model parameters"""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    parameters = pickle.load(f)
                    
                logger.info(f"Loaded parameters for model: {model_name}")
                return parameters
            else:
                logger.info(f"No saved parameters found for model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model parameters: {e}")
            return None
            
    async def cleanup_old_data(self, retention_days: int):
        """Clean up data older than retention period"""
        try:
            cursor = self.conn.cursor()
            
            cutoff = datetime.now() - timedelta(days=retention_days)
            
            # Clean up measurements
            cursor.execute("DELETE FROM measurements WHERE timestamp < ?", (cutoff,))
            measurements_deleted = cursor.rowcount
            
            # Clean up forecasts
            cursor.execute("DELETE FROM forecasts WHERE timestamp < ?", (cutoff,))
            forecasts_deleted = cursor.rowcount
            
            # Clean up old model parameters (keep last 10 per model)
            cursor.execute("""
                DELETE FROM model_parameters
                WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY timestamp DESC) as rn
                        FROM model_parameters
                    ) WHERE rn <= 10
                )
            """)
            params_deleted = cursor.rowcount
            
            self.conn.commit()
            
            logger.info(f"Cleanup complete: {measurements_deleted} measurements, "
                       f"{forecasts_deleted} forecasts, {params_deleted} parameter sets deleted")
                       
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    async def get_latest_forecast(self) -> Optional[Dict]:
        """Get the most recent forecast"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT * FROM forecasts
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            
            if row:
                forecast_data = json.loads(row['forecast_data'])
                return {
                    'timestamp': row['timestamp'],
                    'data': forecast_data,
                    'accuracy_metrics': json.loads(row['accuracy_metrics'] or '{}')
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest forecast: {e}")
            return None
            
    async def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            cursor = self.conn.cursor()
            
            stats = {}
            
            # Count measurements
            cursor.execute("SELECT COUNT(*) as count FROM measurements")
            stats['measurements_count'] = cursor.fetchone()['count']
            
            # Count forecasts
            cursor.execute("SELECT COUNT(*) as count FROM forecasts")
            stats['forecasts_count'] = cursor.fetchone()['count']
            
            # Get date range
            cursor.execute("""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date
                FROM measurements
            """)
            row = cursor.fetchone()
            stats['data_range'] = {
                'start': row['min_date'],
                'end': row['max_date']
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}