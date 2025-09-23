// HomeForecast Dashboard JavaScript

class HomeForecastDashboard {
    constructor() {
        this.charts = {};
        this.updateInterval = 30000; // 30 seconds
        this.updateTimer = null;
        this.init();
    }

    init() {
        // Initialize event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.loadAllData();
        
        // Start auto-refresh
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Manual update button
        const updateBtn = document.getElementById('updateBtn');
        if (updateBtn) {
            updateBtn.addEventListener('click', () => this.triggerUpdate());
        }

        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Settings form
        const settingsForm = document.getElementById('settingsForm');
        if (settingsForm) {
            settingsForm.addEventListener('submit', (e) => this.saveSettings(e));
        }

        // ML Training Modal - close when clicking outside
        const modal = document.getElementById('mlTrainingModal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideMLTrainingDialog();
                }
            });
        }

        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modal = document.getElementById('mlTrainingModal');
                if (modal && modal.style.display === 'block') {
                    this.hideMLTrainingDialog();
                }
            }
        });
    }

    async loadAllData() {
        try {
            // Show loading indicators
            this.showLoading(true);

            // Fetch data with individual error handling to prevent one failed API from breaking everything
            const statusPromise = this.fetchStatus().catch(e => {
                console.error('Status API failed:', e);
                this.showError('Status data unavailable');
                return null;
            });
            const parametersPromise = this.fetchModelParameters().catch(e => {
                console.error('Parameters API failed:', e);
                this.showError('Model parameters unavailable');
                return null;
            });
            const statisticsPromise = this.fetchStatistics().catch(e => {
                console.error('Statistics API failed:', e);
                this.showError('Statistics unavailable');
                return null;
            });

            // Fetch optional data (forecast, comfort analysis)
            const forecastPromise = this.fetchForecast().catch(e => {
                console.warn('Forecast not available:', e);
                return null;
            });
            const comfortPromise = this.fetchComfortAnalysis().catch(e => {
                console.warn('Comfort analysis not available:', e);
                return null;
            });

            // Wait for all promises
            const [status, parameters, statistics, forecast, comfort] = await Promise.all([
                statusPromise,
                parametersPromise, 
                statisticsPromise,
                forecastPromise,
                comfortPromise
            ]);

            // Update UI with fetched data
            console.log('API responses:', { status, forecast, comfort, parameters, statistics });
            
            // Update each section individually with error handling and fallback displays
            if (status) {
                try { 
                    this.updateStatus(status); 
                } catch (e) { 
                    console.error('Error updating status:', e);
                    this.showDataError('status', 'Status display error');
                }
            } else {
                this.showDataError('status', 'Status unavailable');
            }
            
            if (parameters) {
                try { 
                    this.updateParameters(parameters); 
                } catch (e) { 
                    console.error('Error updating parameters:', e);
                    this.showDataError('parameters', 'Parameters display error');
                }
            } else {
                this.showDataError('parameters', 'Parameters unavailable');
            }
            
            if (statistics) {
                try { 
                    this.updateStatistics(statistics); 
                } catch (e) { 
                    console.error('Error updating statistics:', e);
                    this.showDataError('statistics', 'Statistics display error');
                }
            } else {
                this.showDataError('statistics', 'Statistics unavailable');
            }
            
            if (forecast) {
                try { this.updateForecast(forecast); } catch (e) { console.error('Error updating forecast:', e); }
            }
            
            if (comfort) {
                try { this.updateComfort(comfort); } catch (e) { console.error('Error updating comfort:', e); }
            }

        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load data. Please check the logs.');
        } finally {
            this.showLoading(false);
        }
    }

    async fetchStatus() {
        try {
            const response = await fetch('/api/status');
            if (!response.ok) {
                throw new Error(`Status fetch failed: ${response.status}`);
            }
            const data = await response.json();
            console.log('Status data:', data);
            return data;
        } catch (error) {
            console.error('Error fetching status:', error);
            throw error;
        }
    }

    async fetchForecast() {
        try {
            const response = await fetch('/api/forecast/latest');
            if (!response.ok) {
                console.warn(`Forecast fetch failed: ${response.status}`);
                return null;
            }
            const data = await response.json();
            console.log('Forecast data:', data);
            return data;
        } catch (error) {
            console.error('Error fetching forecast:', error);
            return null;
        }
    }

    async fetchComfortAnalysis() {
        try {
            const response = await fetch('/api/comfort/analysis');
            if (!response.ok) {
                console.warn(`Comfort analysis fetch failed: ${response.status}`);
                return null;
            }
            const data = await response.json();
            console.log('Comfort analysis data:', data);
            return data;
        } catch (error) {
            console.error('Error fetching comfort analysis:', error);
            return null;
        }
    }

    async fetchModelParameters() {
        try {
            const response = await fetch('/api/model/parameters');
            if (!response.ok) {
                throw new Error(`Model parameters fetch failed: ${response.status}`);
            }
            const data = await response.json();
            console.log('Model parameters data:', data);
            return data;
        } catch (error) {
            console.error('Error fetching model parameters:', error);
            throw error;
        }
    }

    async fetchStatistics() {
        try {
            const response = await fetch('/api/statistics');
            if (!response.ok) {
                throw new Error(`Statistics fetch failed: ${response.status}`);
            }
            const data = await response.json();
            console.log('Statistics data:', data);
            return data;
        } catch (error) {
            console.error('Error fetching statistics:', error);
            throw error;
        }
    }

    updateStatus(status) {
        console.log('updateStatus called with:', status);
        
        // Update status indicator
        const statusEl = document.getElementById('status');
        if (statusEl) {
            statusEl.textContent = status.status || 'Unknown';
            statusEl.className = 'status status-' + (status.status === 'running' ? 'ok' : 'error');
        }

        // Update last update time
        const lastUpdateEl = document.getElementById('lastUpdate');
        if (lastUpdateEl) {
            if (status.last_update_display) {
                lastUpdateEl.textContent = status.last_update_display;
            } else if (status.last_update) {
                const date = new Date(status.last_update);
                lastUpdateEl.textContent = this.formatRelativeTime(date);
            } else {
                lastUpdateEl.textContent = 'just now';
            }
        }

        // Store timezone for use in other components
        if (status.timezone) {
            this.timezone = status.timezone;
            console.log(`🌍 Using timezone: ${this.timezone}`);
        }

        // Update system information in Debug tab
        this.updateSystemInfo(status);

        // Update thermal model section
        this.updateThermalModel(status);

        // Update ML performance section  
        this.updateMLPerformance(status);

        // Update thermostat configuration
        this.updateThermostatConfiguration(status);

        // Update current temperature from current sensor data
        const currentTempEl = document.getElementById('currentTemp');
        if (currentTempEl && status.current_data && status.current_data.indoor_temp) {
            currentTempEl.textContent = status.current_data.indoor_temp.toFixed(1) + '°F';
        } else if (currentTempEl) {
            currentTempEl.textContent = '-';
        }

        // Update configuration display
        if (status.config) {
            const comfortMinEl = document.getElementById('comfortMin');
            const comfortMaxEl = document.getElementById('comfortMax');
            const mlEnabledEl = document.getElementById('mlEnabled');
            const smartHvacEnabledEl = document.getElementById('smartHvacEnabled');
            const updateIntervalEl = document.getElementById('updateInterval');
            
            if (comfortMinEl) comfortMinEl.textContent = status.config.comfort_min + '°F';
            if (comfortMaxEl) comfortMaxEl.textContent = status.config.comfort_max + '°F';
            if (mlEnabledEl) mlEnabledEl.textContent = status.config.ml_enabled ? 'Enabled' : 'Disabled';
            if (smartHvacEnabledEl) smartHvacEnabledEl.textContent = status.config.smart_hvac_enabled ? 'Enabled' : 'Disabled';
            if (updateIntervalEl) updateIntervalEl.textContent = status.config.update_interval + ' min';
        }

        // Update climate action insights
        if (status.climate_insights) {
            this.updateClimateInsights(status.climate_insights);
        }

        // Update thermostat data in Analysis tab
        if (status.thermostat_data) {
            this.updateThermostatData(status.thermostat_data);
        }
    }

    updateClimateInsights(insights) {
        console.log('updateClimateInsights called with:', insights);
        
        const recommendedActionEl = document.getElementById('recommendedAction');
        if (recommendedActionEl) {
            recommendedActionEl.textContent = insights.recommended_action || 'UNKNOWN';
            // Add CSS class for styling based on action
            recommendedActionEl.className = 'metric-value large action-status action-' + 
                                           (insights.recommended_action || 'unknown').toLowerCase();
        }

        // Context-aware label updates
        const isCurrentlyActive = insights.next_action_time === 'Currently Active';
        const isCurrentlyOff = insights.action_off_time && insights.action_off_time.includes('Currently off');
        
        // Update "Next Climate On Time" label based on context
        const nextActionLabel = document.querySelector('#nextActionTime').previousElementSibling;
        if (nextActionLabel && nextActionLabel.classList.contains('metric-label')) {
            if (isCurrentlyActive) {
                nextActionLabel.textContent = 'HVAC Status:';
            } else if (insights.recommended_action && insights.recommended_action.includes('NOW')) {
                nextActionLabel.textContent = 'Action Needed:';
            } else {
                nextActionLabel.textContent = 'Next Climate On Time:';
            }
        }

        // Update "Climate Action Off Time" label based on context  
        const actionOffLabel = document.querySelector('#actionOffTime').previousElementSibling;
        if (actionOffLabel && actionOffLabel.classList.contains('metric-label')) {
            if (isCurrentlyOff) {
                actionOffLabel.textContent = 'HVAC Status:';
            } else if (isCurrentlyActive) {
                actionOffLabel.textContent = 'Estimated Off Time:';
            } else {
                actionOffLabel.textContent = 'Climate Action Off Time:';
            }
        }

        const actionOffTimeEl = document.getElementById('actionOffTime');
        if (actionOffTimeEl) {
            actionOffTimeEl.textContent = insights.action_off_time || 'N/A';
        }

        const nextActionTimeEl = document.getElementById('nextActionTime');
        if (nextActionTimeEl) {
            nextActionTimeEl.textContent = insights.next_action_time || 'N/A';
        }

        const estimatedRuntimeEl = document.getElementById('estimatedRuntime');
        if (estimatedRuntimeEl) {
            estimatedRuntimeEl.textContent = insights.estimated_runtime || 'N/A';
        }
    }

    updateThermostatData(thermostat_data) {
        console.log('updateThermostatData called with:', thermostat_data);
        
        const currentSetpointEl = document.getElementById('currentSetpoint');
        if (currentSetpointEl && thermostat_data.target_temperature) {
            currentSetpointEl.textContent = thermostat_data.target_temperature.toFixed(1) + '°F';
        }

        const hvacModeEl = document.getElementById('hvacMode');
        if (hvacModeEl) {
            hvacModeEl.textContent = (thermostat_data.hvac_mode || 'unknown').toUpperCase();
        }

        const hvacActionEl = document.getElementById('hvacAction');
        if (hvacActionEl) {
            hvacActionEl.textContent = (thermostat_data.hvac_action || 'unknown').toUpperCase();
        }
    }

    updateSystemInfo(status) {
        // Update version information
        const addonVersionEl = document.getElementById('addonVersion');
        if (addonVersionEl && status.system_info && status.system_info.addon_version) {
            addonVersionEl.textContent = status.system_info.addon_version;
        }

        // Update timezone information
        const systemTimezoneEl = document.getElementById('systemTimezone');
        if (systemTimezoneEl && status.timezone) {
            systemTimezoneEl.textContent = status.timezone;
        }

        // Update current local time
        const currentLocalTimeEl = document.getElementById('currentLocalTime');
        if (currentLocalTimeEl && status.current_time) {
            currentLocalTimeEl.textContent = status.current_time;
        }

        // Update Python version
        const pythonVersionEl = document.getElementById('pythonVersion');
        if (pythonVersionEl && status.system_info && status.system_info.python_version) {
            pythonVersionEl.textContent = status.system_info.python_version;
        }

        // Update log level
        const logLevelEl = document.getElementById('logLevel');
        if (logLevelEl && status.system_info && status.system_info.log_level) {
            const logLevels = {10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'};
            logLevelEl.textContent = logLevels[status.system_info.log_level] || 'INFO';
        }
    }

    updateThermalModel(status) {
        // Update thermal model MAE
        const thermalMaeEl = document.getElementById('thermalModelMae');
        if (thermalMaeEl && status.thermal_metrics && status.thermal_metrics.mae !== null) {
            thermalMaeEl.textContent = status.thermal_metrics.mae.toFixed(3) + '°F';
        } else if (thermalMaeEl) {
            thermalMaeEl.textContent = 'N/A';
        }
    }

    updateMLPerformance(status) {
        // Update ML model status
        const mlStatusEl = document.getElementById('mlModelStatus');
        if (mlStatusEl && status.ml_performance) {
            const statusMap = {
                'trained': 'Trained',
                'not_trained': 'Not Trained',
                'disabled': 'Disabled',
                'error': 'Error'
            };
            mlStatusEl.textContent = statusMap[status.ml_performance.status] || 'Unknown';
        }

        // Update ML accuracy (R²)
        const mlAccuracyEl = document.getElementById('mlModelAccuracy');
        if (mlAccuracyEl && status.ml_performance && status.ml_performance.r2 !== null) {
            mlAccuracyEl.textContent = (status.ml_performance.r2 * 100).toFixed(1) + '%';
        } else if (mlAccuracyEl) {
            mlAccuracyEl.textContent = 'N/A';
        }

        // Update training data points
        const trainingDataEl = document.getElementById('trainingDataPoints');
        if (trainingDataEl && status.ml_performance) {
            trainingDataEl.textContent = status.ml_performance.training_samples || '0';
        }

        // Update last model update
        const lastUpdateEl = document.getElementById('lastModelUpdate');
        if (lastUpdateEl && status.ml_performance && status.ml_performance.last_update) {
            const updateDate = new Date(status.ml_performance.last_update);
            lastUpdateEl.textContent = updateDate.toLocaleDateString() + ' ' + updateDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', hour12: true});
        } else if (lastUpdateEl) {
            lastUpdateEl.textContent = 'Never';
        }
    }

    updateThermostatConfiguration(status) {
        // Update comfort range
        const comfortRangeEl = document.getElementById('comfortRange');
        if (comfortRangeEl && status.config) {
            const min = status.config.comfort_min || 62;
            const max = status.config.comfort_max || 80;
            comfortRangeEl.textContent = `${min}°F - ${max}°F`;
        }
    }

    updateForecast(forecast) {
        if (!forecast || !forecast.data) return;

        const data = forecast.data;
        
        // Update forecast chart
        this.updateForecastChart(data);
        
        // Update energy analysis chart
        this.updateEnergyChart(data);
        
        // Update current temperature
        if (data.initial_conditions) {
            document.getElementById('currentTemp').textContent = 
                data.initial_conditions.indoor_temp.toFixed(1) + '°F';
        }
    }

    updateForecastChart(data) {
        const ctx = document.getElementById('forecastChart');
        if (!ctx) return;

        console.log('Updating forecast chart with enhanced historical/forecast separation:', data);
        
        // 🔍 DEBUG: Browser timezone information
        const browserTz = Intl.DateTimeFormat().resolvedOptions().timeZone;
        const currentTime = new Date();
        console.log('🌍 Browser timezone:', browserTz);
        console.log('🕒 Current browser time:', currentTime.toLocaleString());
        console.log('🕒 Current browser time (ISO):', currentTime.toISOString());
        console.log('🕒 Current browser time (local):', currentTime.toString());

        // Process HVAC operation periods for annotations
        const hvacPeriods = this.extractHvacPeriods(data);
        console.log('Extracted HVAC periods:', hvacPeriods);

        // Prepare chart data with proper historical/forecast separation
        const datasets = [];
        let allLabels = [];
        
        // Check if we have the new separated data structure
        if (data.historical_data && data.forecast_data) {
            console.log(`📊 Using separated data structure - ${data.historical_data.timestamps?.length || 0} historical + ${data.forecast_data.timestamps?.length || 0} forecast points`);
            
            // Create combined timeline with proper timezone handling
            console.log('📊 DEBUG: Sample timestamps from API:');
            console.log('Historical sample:', data.historical_data.timestamps?.[0]);
            console.log('Forecast sample:', data.forecast_data.timestamps?.[0]);
            console.log('Timeline separator:', data.timeline_separator?.separator_timestamp);
            
            const historicalLabels = (data.historical_data.timestamps || []).map(ts => {
                // Parse timezone-aware timestamp properly
                const date = new Date(ts);
                console.log(`Historical: ${ts} -> ${date.toLocaleString()} (browser interprets as ${date.toISOString()})`);
                return date.toLocaleTimeString([], { hour: '2-digit', minute:'2-digit', hour12: true });
            });
            const forecastLabels = (data.forecast_data.timestamps || []).map(ts => {
                // Parse timezone-aware timestamp properly
                const date = new Date(ts);
                return date.toLocaleTimeString([], { hour: '2-digit', minute:'2-digit', hour12: true });
            });
            allLabels = [...historicalLabels, ...forecastLabels];

            // === HISTORICAL SECTION ===
            
            // 1. Historical Actual Outdoor Temperature (from sensors)
            if (data.historical_data.actual_outdoor_temp) {
                datasets.push({
                    label: 'Historical Outdoor (Actual)',
                    data: [...data.historical_data.actual_outdoor_temp, 
                           ...new Array(forecastLabels.length).fill(null)],
                    borderColor: '#8B4513',
                    backgroundColor: 'rgba(139, 69, 19, 0.2)',
                    tension: 0.4,
                    pointRadius: 4,
                    borderWidth: 3,
                    borderDash: [0], // Solid line for actual data
                    fill: false
                });
            }

            // 2. Historical Cached Forecasted Outdoor (from weather cache)
            if (data.historical_weather && data.historical_weather.length > 0) {
                // Pad historical weather data to match timeline
                const historicalWeatherData = data.historical_weather.map(hw => hw.temperature);
                // Pad to match historical timeline length
                while (historicalWeatherData.length < historicalLabels.length) {
                    historicalWeatherData.push(null);
                }
                
                datasets.push({
                    label: 'Historical Cached Forecast',
                    data: [...historicalWeatherData, 
                           ...new Array(forecastLabels.length).fill(null)],
                    borderColor: '#607D8B',
                    backgroundColor: 'rgba(96, 125, 139, 0.2)',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2,
                    borderDash: [3, 3], // Dashed line for cached forecast
                    fill: false
                });
            }

            // 3. Historical Actual Indoor Temperature (from sensors)
            if (data.historical_data.actual_indoor_temp) {
                datasets.push({
                    label: 'Historical Indoor (Actual)',
                    data: [...data.historical_data.actual_indoor_temp, 
                           ...new Array(forecastLabels.length).fill(null)],
                    borderColor: '#0D47A1',
                    backgroundColor: 'rgba(13, 71, 161, 0.2)',
                    tension: 0.4,
                    pointRadius: 5,
                    borderWidth: 4,
                    borderDash: [0], // Solid line for actual data
                    fill: false
                });
            }

            // === FORECAST SECTION ===

            // 4. Forecasted Outdoor Temperature
            if (data.forecast_data.forecasted_outdoor_temp) {
                datasets.push({
                    label: 'Forecasted Outdoor',
                    data: [...new Array(historicalLabels.length).fill(null),
                           ...data.forecast_data.forecasted_outdoor_temp],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 3,
                    borderDash: [5, 5], // Dashed line for forecasted data
                    fill: false
                });
            }

            // 5. Projected Indoor with HVAC Control
            if (data.forecast_data.projected_indoor_with_hvac) {
                datasets.push({
                    label: 'Forecasted Indoor (Smart HVAC)',
                    data: [...new Array(historicalLabels.length).fill(null),
                           ...data.forecast_data.projected_indoor_with_hvac],
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    pointRadius: 4,
                    borderWidth: 4,
                    borderDash: [8, 4], // Dashed line for projections
                    fill: false
                });
            }

            // 6. Projected Indoor without HVAC (No Control)
            if (data.forecast_data.projected_indoor_no_hvac) {
                datasets.push({
                    label: 'Forecasted Indoor (No Control)',
                    data: [...new Array(historicalLabels.length).fill(null),
                           ...data.forecast_data.projected_indoor_no_hvac],
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2,
                    borderDash: [15, 5], // Long dashed line for no control scenario
                    fill: false
                });
            }

        } else {
            // Fallback to legacy format if new structure not available
            console.log('📊 Using legacy data structure for chart');
            
            // Handle historical weather data in legacy mode
            if (data.historical_weather && data.historical_weather.length > 0) {
                console.log(`Adding ${data.historical_weather.length} historical weather points to legacy chart`);
                
                // Create historical labels and data
                const historicalLabels = data.historical_weather.map(hw => {
                    const date = new Date(hw.timestamp);
                    return date.toLocaleTimeString([], { hour: '2-digit', minute:'2-digit', hour12: true });
                });

                // Create combined labels (historical + forecast)
                const forecastLabels = data.timestamps.map(ts => {
                    const date = new Date(ts);
                    return date.toLocaleTimeString([], { hour: '2-digit', minute:'2-digit', hour12: true });
                });
                allLabels = [...historicalLabels, ...forecastLabels];

                // Historical outdoor temperature dataset
                datasets.push({
                    label: 'Historical Outdoor (Cached)',
                    data: [...data.historical_weather.map(hw => hw.temperature), 
                           ...new Array(forecastLabels.length).fill(null)],
                    borderColor: '#8B4513',
                    backgroundColor: 'rgba(139, 69, 19, 0.2)',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2,
                    borderDash: [3, 3],
                    fill: false
                });

                // Forecast outdoor data with null values for historical period
                datasets.push({
                    label: 'Forecasted Outdoor',
                    data: [...new Array(historicalLabels.length).fill(null),
                           ...data.outdoor_forecast],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 3,
                    borderDash: [5, 5],
                    fill: false
                });

                // Indoor forecasts also need null padding for historical period
                datasets.push({
                    label: 'Projected Indoor (Smart HVAC)',
                    data: [...new Array(historicalLabels.length).fill(null),
                           ...data.controlled_trajectory.map(p => p.indoor_temp)],
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    pointRadius: 4,
                    borderWidth: 4,
                    borderDash: [8, 4],
                    fill: false
                });

            } else {
                // No historical data - standard legacy chart
                allLabels = data.timestamps.map(ts => {
                    const date = new Date(ts);
                    return date.toLocaleTimeString([], { hour: '2-digit', minute:'2-digit', hour12: true });
                });

                datasets.push({
                    label: 'Projected Indoor (Smart HVAC)',
                    data: data.controlled_trajectory.map(p => p.indoor_temp),
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 3,
                    fill: false
                });
                
                // Add other legacy datasets if available
                if (data.outdoor_forecast) {
                    datasets.push({
                        label: 'Outdoor Forecast',
                        data: data.outdoor_forecast,
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.4,
                        pointRadius: 2,
                        borderWidth: 2,
                        fill: false
                    });
                }
                
                if (data.idle_forecast) {
                    datasets.push({
                        label: 'Projected Indoor (No Control)',
                        data: data.idle_forecast,
                        borderColor: '#FF9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.1)',
                        tension: 0.4,
                        pointRadius: 2,
                        borderWidth: 2,
                        borderDash: [10, 5],
                        fill: false
                    });
                }
            }
        }

        // Create the chart data structure
        const chartData = {
            labels: allLabels,
            datasets: datasets
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Smart HVAC Temperature Forecast & Control Schedule',
                    font: { size: 18, weight: 'bold' }
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        title: function(tooltipItems) {
                            const index = tooltipItems[0].dataIndex;
                            const timestamp = data.timestamps[index];
                            const date = new Date(timestamp);
                            return date.toLocaleDateString() + ' ' + 
                                   date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', hour12: true});
                        },
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            label += context.parsed.y.toFixed(1) + '°F';
                            
                            // Add HVAC state info for controlled trajectory
                            if (context.dataset.label === 'Projected Indoor (Smart HVAC)' && 
                                data.controlled_trajectory[context.dataIndex]) {
                                const hvacState = data.controlled_trajectory[context.dataIndex].hvac_state;
                                if (hvacState && hvacState !== 'off') {
                                    label += ` (HVAC: ${hvacState.toUpperCase()})`;
                                }
                            }
                            
                            return label;
                        },
                        afterBody: function(tooltipItems) {
                            const index = tooltipItems[0].dataIndex;
                            const timestamp = new Date(data.timestamps[index]);
                            
                            // Get comfort band info
                            const comfortMin = parseFloat(document.getElementById('comfortMin')?.textContent || 0);
                            const comfortMax = parseFloat(document.getElementById('comfortMax')?.textContent || 0);
                            
                            if (comfortMin && comfortMax) {
                                return [`Comfort Range: ${comfortMin}°F - ${comfortMax}°F`];
                            }
                            return [];
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time of Day',
                        font: { size: 14, weight: 'bold' }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Temperature (°F)',
                        font: { size: 14, weight: 'bold' }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(0) + '°F';
                        }
                    }
                }
            }
        };

        // Get comfort band values
        const comfortMinEl = document.getElementById('comfortMin');
        const comfortMaxEl = document.getElementById('comfortMax');
        
        let comfortMin = 68; // Default values
        let comfortMax = 74;
        
        if (comfortMinEl && comfortMaxEl) {
            comfortMin = parseFloat(comfortMinEl.textContent.replace('°F', '')) || 68;
            comfortMax = parseFloat(comfortMaxEl.textContent.replace('°F', '')) || 74;
        }

        // Create comprehensive annotations
        const annotations = {
            // Comfort band background
            comfortZone: {
                type: 'box',
                yMin: comfortMin,
                yMax: comfortMax,
                backgroundColor: 'rgba(76, 175, 80, 0.15)',
                borderColor: 'rgba(76, 175, 80, 0.4)',
                borderWidth: 2,
                borderDash: [3, 3],
                label: {
                    content: `Comfort Band (${comfortMin}°F - ${comfortMax}°F)`,
                    enabled: true,
                    position: 'start',
                    backgroundColor: 'rgba(76, 175, 80, 0.8)',
                    color: 'white',
                    font: { size: 11, weight: 'bold' },
                    padding: 6
                }
            },
            // Comfort zone boundaries
            comfortMin: {
                type: 'line',
                yMin: comfortMin,
                yMax: comfortMin,
                borderColor: 'rgba(76, 175, 80, 0.8)',
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                    content: `Min Comfort: ${comfortMin}°F`,
                    enabled: true,
                    position: 'end',
                    backgroundColor: 'rgba(76, 175, 80, 0.9)',
                    color: 'white',
                    font: { size: 10 }
                }
            },
            comfortMax: {
                type: 'line',
                yMin: comfortMax,
                yMax: comfortMax,
                borderColor: 'rgba(76, 175, 80, 0.8)',
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                    content: `Max Comfort: ${comfortMax}°F`,
                    enabled: true,
                    position: 'end',
                    backgroundColor: 'rgba(76, 175, 80, 0.9)',
                    color: 'white',
                    font: { size: 10 }
                }
            }
        };

        // Add "NOW" vertical line to separate historical from forecast data
        if (data.timeline_separator && data.timeline_separator.historical_end_index !== undefined) {
            const nowLineIndex = data.timeline_separator.historical_end_index + 0.5; // Position between last historical and first forecast
            annotations.nowLine = {
                type: 'line',
                xMin: nowLineIndex,
                xMax: nowLineIndex,
                borderColor: '#FF6B6B',
                borderWidth: 4,
                borderDash: [8, 4],
                label: {
                    content: '🕒 NOW - Forecast Begins',
                    enabled: true,
                    position: 'start',
                    backgroundColor: 'rgba(255, 107, 107, 0.9)',
                    color: 'white',
                    font: { size: 12, weight: 'bold' },
                    padding: 8,
                    rotation: 0
                }
            };
        } else if (data.historical_data && data.historical_data.timestamps) {
            // Fallback: use length of historical data as separator
            const nowLineIndex = data.historical_data.timestamps.length - 0.5;
            annotations.nowLine = {
                type: 'line',
                xMin: nowLineIndex,
                xMax: nowLineIndex,
                borderColor: '#FF6B6B',
                borderWidth: 4,
                borderDash: [8, 4],
                label: {
                    content: '🕒 NOW - Forecast Begins',
                    enabled: true,
                    position: 'start',
                    backgroundColor: 'rgba(255, 107, 107, 0.9)',
                    color: 'white',
                    font: { size: 12, weight: 'bold' },
                    padding: 8
                }
            };
        }

        // Add HVAC operation period annotations
        hvacPeriods.forEach((period, index) => {
            annotations[`hvac_${index}`] = {
                type: 'box',
                xMin: period.startIndex - 0.4,
                xMax: period.endIndex + 0.4,
                backgroundColor: period.color,
                borderColor: period.borderColor,
                borderWidth: 2,
                label: {
                    content: `${period.mode.toUpperCase()}: ${period.startTime} - ${period.endTime}`,
                    enabled: true,
                    position: 'center',
                    backgroundColor: period.borderColor,
                    color: 'white',
                    font: { size: 9, weight: 'bold' },
                    padding: 4
                }
            };
        });

        options.plugins.annotation = { annotations };

        if (this.charts.forecast) {
            this.charts.forecast.destroy();
        }

        this.charts.forecast = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: options
        });
    }

    extractHvacPeriods(data) {
        const periods = [];
        let currentPeriod = null;
        
        // === HISTORICAL SECTION: Use actual HVAC mode data ===
        if (data.historical_data && data.historical_data.actual_hvac_mode && data.historical_data.actual_hvac_mode.length > 0) {
            console.log('Processing historical HVAC periods from actual_hvac_mode data');
            
            data.historical_data.actual_hvac_mode.forEach((hvacState, index) => {
                const timestamp = data.historical_data.timestamps[index];
                const timeStr = new Date(timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', hour12: true});
                
                // Only highlight when HVAC was actually running (not just set to a mode)
                if (hvacState === 'cooling' || hvacState === 'heating') {
                    if (!currentPeriod || currentPeriod.mode !== hvacState) {
                        // Start new historical period
                        if (currentPeriod) {
                            periods.push(currentPeriod);
                        }
                        
                        currentPeriod = {
                            mode: hvacState,
                            startIndex: index,
                            startTime: timeStr,
                            endIndex: index,
                            endTime: timeStr,
                            section: 'historical',
                            color: hvacState === 'heating' ? 'rgba(255, 87, 34, 0.2)' : 'rgba(33, 150, 243, 0.2)',
                            borderColor: hvacState === 'heating' ? 'rgba(255, 87, 34, 0.8)' : 'rgba(33, 150, 243, 0.8)'
                        };
                    } else {
                        // Continue current period
                        currentPeriod.endIndex = index;
                        currentPeriod.endTime = timeStr;
                    }
                } else {
                    // End current historical period
                    if (currentPeriod) {
                        periods.push(currentPeriod);
                        currentPeriod = null;
                    }
                }
            });
            
            // Close any ongoing historical period
            if (currentPeriod) {
                periods.push(currentPeriod);
                currentPeriod = null;
            }
        }
        
        // === FORECAST SECTION: Use predicted HVAC operation ===
        if (data.forecast_data && data.forecast_data.projected_hvac_mode && data.forecast_data.projected_hvac_mode.length > 0) {
            console.log('Processing forecast HVAC periods from projected_hvac_mode data');
            
            const historicalLength = data.historical_data ? (data.historical_data.timestamps || []).length : 0;
            
            data.forecast_data.projected_hvac_mode.forEach((hvacState, index) => {
                const timestamp = data.forecast_data.timestamps[index];
                const timeStr = new Date(timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', hour12: true});
                const adjustedIndex = historicalLength + index; // Adjust for combined timeline
                
                if (hvacState !== 'off' && hvacState !== 'idle' && hvacState !== 'unknown') {
                    if (!currentPeriod || currentPeriod.mode !== hvacState) {
                        // Start new forecast period
                        if (currentPeriod) {
                            periods.push(currentPeriod);
                        }
                        
                        currentPeriod = {
                            mode: hvacState,
                            startIndex: adjustedIndex,
                            startTime: timeStr,
                            endIndex: adjustedIndex,
                            endTime: timeStr,
                            section: 'forecast',
                            color: hvacState === 'heat' ? 'rgba(255, 87, 34, 0.2)' : 'rgba(33, 150, 243, 0.2)',
                            borderColor: hvacState === 'heat' ? 'rgba(255, 87, 34, 0.8)' : 'rgba(33, 150, 243, 0.8)'
                        };
                    } else {
                        // Continue current period
                        currentPeriod.endIndex = adjustedIndex;
                        currentPeriod.endTime = timeStr;
                    }
                } else {
                    // End current forecast period
                    if (currentPeriod) {
                        periods.push(currentPeriod);
                        currentPeriod = null;
                    }
                }
            });
        }
        
        // Close any ongoing period
        if (currentPeriod) {
            periods.push(currentPeriod);
        }
        
        console.log(`Extracted ${periods.length} HVAC periods:`, periods);
        return periods;
    }

    updateEnergyChart(data) {
        const ctx = document.getElementById('energyChart');
        if (!ctx) return;

        // Check if we have the required trajectory data
        let controlledTrajectory = data.controlled_trajectory;
        let currentTrajectory = data.current_trajectory;
        
        // Handle new data structure if trajectories not directly available
        if (!controlledTrajectory && data.forecast_data && data.forecast_data.projected_indoor_with_hvac) {
            // Create synthetic trajectory data from new structure
            controlledTrajectory = data.forecast_data.projected_indoor_with_hvac.map((temp, i) => ({
                indoor_temp: temp,
                hvac_state: data.forecast_data.projected_hvac_mode ? data.forecast_data.projected_hvac_mode[i] : 'off'
            }));
        }
        
        if (!currentTrajectory && data.forecast_data && data.forecast_data.projected_indoor_no_hvac) {
            // Create synthetic trajectory data from new structure  
            currentTrajectory = data.forecast_data.projected_indoor_no_hvac.map((temp, i) => ({
                indoor_temp: temp,
                hvac_state: 'off' // No HVAC control
            }));
        }

        // Fallback to empty arrays if still no data
        if (!controlledTrajectory || !Array.isArray(controlledTrajectory)) {
            console.warn('No controlled trajectory data available for energy chart');
            controlledTrajectory = [];
        }
        
        if (!currentTrajectory || !Array.isArray(currentTrajectory)) {
            console.warn('No current trajectory data available for energy chart');
            currentTrajectory = [];
        }

        // Calculate HVAC runtime for each trajectory
        const controlledRuntime = controlledTrajectory.filter(p => p.hvac_state && p.hvac_state !== 'off').length;
        const currentRuntime = currentTrajectory.filter(p => p.hvac_state && p.hvac_state !== 'off').length;
        
        // Calculate runtime reduction safely
        let runtimeReduction = 0;
        if (currentRuntime > 0) {
            runtimeReduction = ((currentRuntime - controlledRuntime) / currentRuntime * 100).toFixed(1);
        } else if (controlledRuntime === 0) {
            runtimeReduction = 0; // Both are zero - no change
        } else {
            runtimeReduction = -100; // Current is 0 but controlled has runtime
        }

        const chartData = {
            labels: ['Current Control', 'Smart Control'],
            datasets: [{
                label: 'HVAC Runtime (minutes)',
                data: [
                    currentRuntime * 5, // 5-minute intervals
                    controlledRuntime * 5
                ],
                backgroundColor: ['#FF9800', '#4CAF50'],
                borderWidth: 0
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Energy Savings: ${runtimeReduction}% Runtime Reduction`,
                    font: { size: 16 }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Runtime (minutes)'
                    }
                }
            }
        };

        if (this.charts.energy) {
            this.charts.energy.destroy();
        }

        this.charts.energy = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: options
        });
    }

    updateComfort(comfort) {
        console.log('updateComfort called with:', comfort);
        
        if (!comfort) {
            console.log('No comfort data received');
            return;
        }

        // Update comfort metrics
        const recommendedModeEl = document.getElementById('recommendedMode');
        if (recommendedModeEl) {
            recommendedModeEl.textContent = comfort.recommended_mode ? comfort.recommended_mode.toUpperCase() : 'OFF';
        }
        
        const timeToUpperEl = document.getElementById('timeToUpper');
        if (timeToUpperEl) {
            timeToUpperEl.textContent = comfort.time_to_upper ? comfort.time_to_upper.toFixed(0) + ' min' : 'N/A';
        }
        
        const timeToLowerEl = document.getElementById('timeToLower');
        if (timeToLowerEl) {
            timeToLowerEl.textContent = comfort.time_to_lower ? comfort.time_to_lower.toFixed(0) + ' min' : 'N/A';
        }
        
        const comfortScoreEl = document.getElementById('comfortScore');
        if (comfortScoreEl) {
            comfortScoreEl.textContent = comfort.comfort_score ? comfort.comfort_score.toFixed(0) + '%' : 'N/A';
        }

        // Update recommendations
        this.updateRecommendations(comfort.recommendations || []);

        // Update efficiency metrics
        if (comfort.efficiency_metrics) {
            const metrics = comfort.efficiency_metrics;
            document.getElementById('efficiencyScore').textContent = 
                metrics.efficiency_score.toFixed(0) + '%';
            
            // Update progress bar
            const progressBar = document.querySelector('.efficiency-progress');
            if (progressBar) {
                progressBar.style.width = metrics.efficiency_score + '%';
            }
        }
    }

    updateRecommendations(recommendations) {
        const container = document.getElementById('recommendations');
        if (!container) return;

        container.innerHTML = '';
        
        if (recommendations.length === 0) {
            container.innerHTML = '<div class="recommendation low-priority"><div class="recommendation-title">System operating optimally</div><div class="recommendation-action">No actions needed at this time</div></div>';
            return;
        }

        recommendations.forEach(rec => {
            const div = document.createElement('div');
            div.className = `recommendation ${rec.priority}-priority`;
            div.innerHTML = `
                <div class="recommendation-title">${rec.message}</div>
                <div class="recommendation-action">${rec.action}</div>
            `;
            container.appendChild(div);
        });
    }

    updateParameters(params) {
        console.log('updateParameters called with:', params);
        
        if (!params) {
            console.log('No parameters data received');
            return;
        }

        // Handle both nested and flat parameter structures
        const model = params.thermal_model || params;
        
        // Update thermal parameters (convert hours to more readable format)
        const timeConstantEl = document.getElementById('timeConstant');
        if (timeConstantEl && model.time_constant !== undefined) {
            const hours = parseFloat(model.time_constant);
            timeConstantEl.textContent = hours.toFixed(1) + ' hours';
        }
        
        const heatingRateEl = document.getElementById('heatingRate');
        if (heatingRateEl && model.heating_rate !== undefined) {
            heatingRateEl.textContent = model.heating_rate.toFixed(2) + ' °F/hr';
        }
        
        const coolingRateEl = document.getElementById('coolingRate');
        if (coolingRateEl && model.cooling_rate !== undefined) {
            coolingRateEl.textContent = model.cooling_rate.toFixed(2) + ' °F/hr';
        }

        // Update model quality
        const quality = params.model_quality || params.quality;
        if (quality) {
            const convergenceEl = document.getElementById('modelConvergence');
            if (convergenceEl) {
                convergenceEl.textContent = quality.parameter_convergence ? 'Converged' : 'Learning';
            }
            
            const errorEl = document.getElementById('modelError');
            if (errorEl && quality.mae !== null && quality.mae !== undefined) {
                errorEl.textContent = quality.mae.toFixed(3) + ' °F/hr';
            }
        }

        // Update ML model info
        if (params.ml_correction && params.ml_correction.is_trained) {
            document.getElementById('mlStatus').textContent = 'Trained';
            if (params.ml_correction.latest_performance) {
                document.getElementById('mlAccuracy').textContent = 
                    (params.ml_correction.latest_performance.r2 * 100).toFixed(1) + '%';
            }
        }
    }

    updateStatistics(stats) {
        if (!stats) return;

        // Update database statistics
        if (stats.database) {
            document.getElementById('measurementCount').textContent = 
                stats.database.measurements_count.toLocaleString();
            document.getElementById('forecastCount').textContent = 
                stats.database.forecasts_count.toLocaleString();
            
            if (stats.database.data_range && stats.database.data_range.start) {
                const startDate = new Date(stats.database.data_range.start);
                const days = Math.floor((Date.now() - startDate) / (1000 * 60 * 60 * 24));
                document.getElementById('dataAge').textContent = days + ' days';
            }
        }
    }

    async triggerUpdate() {
        const btn = document.getElementById('updateBtn');
        const originalText = btn.innerHTML;
        
        try {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Updating...';
            
            const response = await fetch('/api/trigger/update');
            const result = await response.json();
            
            if (response.ok) {
                this.showNotification('Update triggered successfully', 'success');
                // Reload data after a delay
                setTimeout(() => this.loadAllData(), 3000);
            } else {
                throw new Error(result.error || 'Update failed');
            }
        } catch (error) {
            this.showNotification('Failed to trigger update: ' + error.message, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === tabName + 'Tab');
        });
    }

    formatRelativeTime(date) {
        const seconds = Math.floor((Date.now() - date) / 1000);
        
        if (seconds < 60) return 'just now';
        if (seconds < 3600) return Math.floor(seconds / 60) + ' minutes ago';
        if (seconds < 86400) return Math.floor(seconds / 3600) + ' hours ago';
        return Math.floor(seconds / 86400) + ' days ago';
    }

    showLoading(show) {
        const elements = document.querySelectorAll('.metric-value');
        elements.forEach(el => {
            if (show) {
                // Store original content before showing spinner
                if (!el.hasAttribute('data-original-content')) {
                    el.setAttribute('data-original-content', el.innerHTML);
                }
                el.innerHTML = '<span class="spinner"></span>';
                el.classList.add('loading');
            } else {
                // Remove loading state
                el.classList.remove('loading');
                // Don't restore original content - let the update methods handle it
            }
        });
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showDataError(section, message) {
        // Find elements that should show error state instead of spinners
        const selectorMap = {
            'status': '.status-section .metric-value, .current-data .metric-value',
            'parameters': '.model-section .metric-value',
            'statistics': '.stats-section .metric-value'
        };
        
        const selector = selectorMap[section];
        if (selector) {
            document.querySelectorAll(selector).forEach(el => {
                el.classList.remove('loading');
                el.innerHTML = '<span class="error-state">N/A</span>';
            });
        }
        
        // Show notification
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    async resetModel() {
        if (!confirm('Are you sure you want to reset the model and clear all historical data? This cannot be undone.')) {
            return;
        }

        try {
            this.showNotification('Resetting model and clearing data...', 'info');
            
            const response = await fetch('/api/model/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showNotification('Model reset successfully! Starting fresh data collection...', 'success');
                // Refresh the dashboard after a short delay
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                throw new Error(result.error || 'Reset failed');
            }
        } catch (error) {
            this.showNotification('Failed to reset model: ' + error.message, 'error');
        }
    }

    exportData() {
        // Placeholder for data export functionality
        this.showNotification('Export functionality coming soon...', 'info');
    }

    async showMLTrainingDialog() {
        try {
            // Fetch current ML training info
            const response = await fetch('/api/ml/training-info');
            const info = await response.json();
            
            if (!response.ok) {
                throw new Error(info.error || 'Failed to fetch training info');
            }
            
            // Populate modal with training information
            document.getElementById('modalModelType').textContent = info.model_type || 'Random Forest';
            document.getElementById('modalDataPoints').textContent = info.data_points || '0';
            document.getElementById('modalTrainingPeriod').textContent = info.training_period || '30 days';
            document.getElementById('modalEstimatedDuration').textContent = info.estimated_duration || '5-15 seconds';
            
            // Show the modal
            document.getElementById('mlTrainingModal').style.display = 'block';
        } catch (error) {
            this.showNotification('Failed to load training info: ' + error.message, 'error');
        }
    }

    hideMLTrainingDialog() {
        document.getElementById('mlTrainingModal').style.display = 'none';
    }

    async confirmMLTraining() {
        const confirmBtn = document.getElementById('confirmTrainBtn');
        const btnText = confirmBtn.querySelector('.btn-text');
        const spinner = confirmBtn.querySelector('.spinner');
        
        try {
            // Show loading state
            confirmBtn.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'inline';
            
            this.showNotification('Starting ML model training...', 'info');
            
            const response = await fetch('/api/ml/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showNotification('ML model trained successfully! ' + 
                    (result.metrics ? `R² Score: ${result.metrics.r2?.toFixed(3)}` : ''), 'success');
                
                // Hide modal and refresh data
                this.hideMLTrainingDialog();
                setTimeout(() => {
                    this.loadAllData();
                }, 1000);
            } else {
                throw new Error(result.error || 'Training failed');
            }
        } catch (error) {
            this.showNotification('Failed to train ML model: ' + error.message, 'error');
        } finally {
            // Reset button state
            confirmBtn.disabled = false;
            btnText.style.display = 'inline';
            spinner.style.display = 'none';
        }
    }

    startAutoRefresh() {
        this.updateTimer = setInterval(() => {
            this.loadAllData();
        }, this.updateInterval);
    }

    stopAutoRefresh() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new HomeForecastDashboard();
});

// Add notification styles
const style = document.createElement('style');
style.textContent = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px 20px;
    border-radius: 4px;
    background: #333;
    color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transform: translateX(400px);
    transition: transform 0.3s ease;
    z-index: 1000;
}
.notification.show {
    transform: translateX(0);
}
.notification.success {
    background: #4CAF50;
}
.notification.error {
    background: #F44336;
}
.notification.warning {
    background: #FF9800;
}
`;
document.head.appendChild(style);