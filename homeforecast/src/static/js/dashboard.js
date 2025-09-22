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
    }

    async loadAllData() {
        try {
            // Show loading indicators
            this.showLoading(true);

            // Fetch required data (status, parameters, statistics)
            const statusPromise = this.fetchStatus();
            const parametersPromise = this.fetchModelParameters();
            const statisticsPromise = this.fetchStatistics();

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
            
            // Update each section individually with error handling
            try { this.updateStatus(status); } catch (e) { console.error('Error updating status:', e); }
            try { this.updateParameters(parameters); } catch (e) { console.error('Error updating parameters:', e); }
            try { this.updateStatistics(statistics); } catch (e) { console.error('Error updating statistics:', e); }
            
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
        if (addonVersionEl && status.version) {
            addonVersionEl.textContent = status.version;
        }

        // Update timezone information
        const systemTimezoneEl = document.getElementById('systemTimezone');
        if (systemTimezoneEl && status.timezone) {
            systemTimezoneEl.textContent = status.timezone;
        }

        // Update current local time
        const currentLocalTimeEl = document.getElementById('currentLocalTime');
        if (currentLocalTimeEl) {
            if (status.current_time) {
                currentLocalTimeEl.textContent = status.current_time;
            } else {
                currentLocalTimeEl.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            }
        }

        // Set other system info (these could come from backend later)
        const pythonVersionEl = document.getElementById('pythonVersion');
        if (pythonVersionEl) {
            pythonVersionEl.textContent = '3.12'; // Could be made dynamic
        }

        const logLevelEl = document.getElementById('logLevel');
        if (logLevelEl) {
            logLevelEl.textContent = 'INFO'; // Could be made dynamic
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

        console.log('Updating forecast chart with data:', data);

        // Process HVAC operation periods for annotations
        const hvacPeriods = this.extractHvacPeriods(data);
        console.log('Extracted HVAC periods:', hvacPeriods);

        const chartData = {
            labels: data.timestamps.map(ts => {
                const date = new Date(ts);
                // If we have timezone info, could adjust here, but browser handles local time conversion
                return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            }),
            datasets: [
                {
                    label: 'Projected Indoor (Smart HVAC)',
                    data: data.controlled_trajectory.map(p => p.indoor_temp),
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 3,
                    fill: false
                },
                {
                    label: 'Forecasted Outdoor',
                    data: data.outdoor_forecast,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Indoor (No HVAC Control)',
                    data: data.idle_trajectory.map(p => p.indoor_temp),
                    borderColor: '#F44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    borderDash: [8, 4],
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Indoor (Current Trajectory)',
                    data: data.current_trajectory ? data.current_trajectory.map(p => p.indoor_temp) : [],
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    borderDash: [4, 2],
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1,
                    fill: false
                }
            ]
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
                            return new Date(timestamp).toLocaleString();
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
        if (!data.controlled_trajectory || data.controlled_trajectory.length === 0) {
            return [];
        }

        const periods = [];
        let currentPeriod = null;
        
        data.controlled_trajectory.forEach((point, index) => {
            const hvacState = point.hvac_state || 'off';
            const timestamp = data.timestamps[index];
            const timeStr = new Date(timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            if (hvacState !== 'off') {
                if (!currentPeriod || currentPeriod.mode !== hvacState) {
                    // Start new period
                    if (currentPeriod) {
                        periods.push(currentPeriod);
                    }
                    
                    currentPeriod = {
                        mode: hvacState,
                        startIndex: index,
                        startTime: timeStr,
                        endIndex: index,
                        endTime: timeStr,
                        color: hvacState === 'heat' ? 'rgba(255, 87, 34, 0.2)' : 'rgba(33, 150, 243, 0.2)',
                        borderColor: hvacState === 'heat' ? 'rgba(255, 87, 34, 0.8)' : 'rgba(33, 150, 243, 0.8)'
                    };
                } else {
                    // Continue current period
                    currentPeriod.endIndex = index;
                    currentPeriod.endTime = timeStr;
                }
            } else {
                // End current period
                if (currentPeriod) {
                    periods.push(currentPeriod);
                    currentPeriod = null;
                }
            }
        });
        
        // Don't forget the last period
        if (currentPeriod) {
            periods.push(currentPeriod);
        }
        
        return periods;
    }

    updateEnergyChart(data) {
        const ctx = document.getElementById('energyChart');
        if (!ctx) return;

        // Calculate HVAC runtime for each trajectory
        const controlledRuntime = data.controlled_trajectory.filter(p => p.hvac_state !== 'off').length;
        const currentRuntime = data.current_trajectory.filter(p => p.hvac_state !== 'off').length;
        
        const runtimeReduction = ((currentRuntime - controlledRuntime) / currentRuntime * 100).toFixed(1);

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