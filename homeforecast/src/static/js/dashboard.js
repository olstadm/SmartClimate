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

            // Fetch all data in parallel
            const [status, forecast, comfort, parameters, statistics] = await Promise.all([
                this.fetchStatus(),
                this.fetchForecast(),
                this.fetchComfortAnalysis(),
                this.fetchModelParameters(),
                this.fetchStatistics()
            ]);

            // Update UI with fetched data
            this.updateStatus(status);
            this.updateForecast(forecast);
            this.updateComfort(comfort);
            this.updateParameters(parameters);
            this.updateStatistics(statistics);

        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load data. Please check the logs.');
        } finally {
            this.showLoading(false);
        }
    }

    async fetchStatus() {
        const response = await fetch('/api/status');
        return response.json();
    }

    async fetchForecast() {
        const response = await fetch('/api/forecast/latest');
        if (response.ok) {
            return response.json();
        }
        return null;
    }

    async fetchComfortAnalysis() {
        const response = await fetch('/api/comfort/analysis');
        if (response.ok) {
            return response.json();
        }
        return null;
    }

    async fetchModelParameters() {
        const response = await fetch('/api/model/parameters');
        return response.json();
    }

    async fetchStatistics() {
        const response = await fetch('/api/statistics');
        return response.json();
    }

    updateStatus(status) {
        // Update status indicator
        const statusEl = document.getElementById('status');
        if (statusEl) {
            statusEl.textContent = status.status || 'Unknown';
            statusEl.className = 'status status-' + (status.status === 'running' ? 'ok' : 'error');
        }

        // Update last update time
        const lastUpdateEl = document.getElementById('lastUpdate');
        if (lastUpdateEl && status.last_update) {
            const date = new Date(status.last_update);
            lastUpdateEl.textContent = this.formatRelativeTime(date);
        }

        // Update configuration display
        if (status.config) {
            document.getElementById('comfortMin').textContent = status.config.comfort_min + '°C';
            document.getElementById('comfortMax').textContent = status.config.comfort_max + '°C';
            document.getElementById('mlEnabled').textContent = status.config.ml_enabled ? 'Enabled' : 'Disabled';
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
                data.initial_conditions.indoor_temp.toFixed(1) + '°C';
        }
    }

    updateForecastChart(data) {
        const ctx = document.getElementById('forecastChart');
        if (!ctx) return;

        const chartData = {
            labels: data.timestamps.map(ts => 
                new Date(ts).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
            ),
            datasets: [
                {
                    label: 'Indoor (Smart Control)',
                    data: data.controlled_trajectory.map(p => p.indoor_temp),
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'Indoor (No HVAC)',
                    data: data.idle_trajectory.map(p => p.indoor_temp),
                    borderColor: '#F44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'Indoor (Current)',
                    data: data.current_trajectory.map(p => p.indoor_temp),
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    borderDash: [2, 2],
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'Outdoor',
                    data: data.outdoor_forecast,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
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
                    text: '12-Hour Temperature Forecast',
                    font: { size: 16 }
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            label += context.parsed.y.toFixed(1) + '°C';
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '°C';
                        }
                    }
                }
            }
        };

        // Add comfort zone annotation
        const comfortMin = parseFloat(document.getElementById('comfortMin').textContent);
        const comfortMax = parseFloat(document.getElementById('comfortMax').textContent);
        
        options.plugins.annotation = {
            annotations: {
                comfortZone: {
                    type: 'box',
                    yMin: comfortMin,
                    yMax: comfortMax,
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderColor: 'rgba(76, 175, 80, 0.3)',
                    borderWidth: 1,
                    label: {
                        content: 'Comfort Zone',
                        enabled: true,
                        position: 'start'
                    }
                }
            }
        };

        if (this.charts.forecast) {
            this.charts.forecast.destroy();
        }

        this.charts.forecast = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: options
        });
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
        if (!comfort) return;

        // Update comfort metrics
        document.getElementById('recommendedMode').textContent = 
            comfort.recommended_mode ? comfort.recommended_mode.toUpperCase() : 'OFF';
        
        document.getElementById('timeToUpper').textContent = 
            comfort.time_to_upper ? comfort.time_to_upper.toFixed(0) + ' min' : 'N/A';
        
        document.getElementById('timeToLower').textContent = 
            comfort.time_to_lower ? comfort.time_to_lower.toFixed(0) + ' min' : 'N/A';
        
        document.getElementById('comfortScore').textContent = 
            comfort.comfort_score ? comfort.comfort_score.toFixed(0) + '%' : 'N/A';

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
        if (!params.thermal_model) return;

        const model = params.thermal_model;
        
        // Update thermal parameters
        document.getElementById('timeConstant').textContent = 
            model.time_constant.toFixed(1) + ' hours';
        document.getElementById('heatingRate').textContent = 
            model.heating_rate.toFixed(2) + ' °C/hr';
        document.getElementById('coolingRate').textContent = 
            model.cooling_rate.toFixed(2) + ' °C/hr';

        // Update model quality
        if (params.model_quality) {
            const quality = params.model_quality;
            document.getElementById('modelConvergence').textContent = 
                quality.parameter_convergence ? 'Converged' : 'Learning';
            
            if (quality.mae !== null) {
                document.getElementById('modelError').textContent = 
                    quality.mae.toFixed(3) + ' °C/hr';
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
        const elements = document.querySelectorAll('.metric-value, .status');
        elements.forEach(el => {
            if (show) {
                el.innerHTML = '<span class="spinner"></span>';
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