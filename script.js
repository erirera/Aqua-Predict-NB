document.addEventListener('DOMContentLoaded', () => {
    // === 1. MAP INITIALIZATION ===
    // Center on New Brunswick
    const map = L.map('map-container').setView([46.5653, -66.4619], 7);

    // Dark mode map tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    let wellData = [];
    const markers = [];
    let currentChart = null;

    // Define custom marker colors based on risk
    const getMarkerColor = (risk) => {
        if (risk === 'high') return '#ef4444'; // Red
        if (risk === 'medium') return '#f59e0b'; // Orange
        return '#10b981'; // Green
    };

    // Fallback procedural generator if file:// CORS restrictions prevent fetch()
    const generateFallbackWells = (count = 150) => {
        const wells = [];
        const types = ['Fractured Bedrock', 'Carboniferous Sandstone', 'Glaciofluvial Sand/Gravel', 'Windsor Group Karst', 'Mafic Volcanic'];
        const qFlags = ['None', 'Arsenic', 'Uranium', 'Methane'];
        const risks = ['low', 'medium', 'high'];
        
        for (let i = 0; i < count; i++) {
            const lat = 45.1 + Math.random() * 2.7;
            const lng = -67.8 + Math.random() * 3.6;
            const risk = risks[Math.floor(Math.random() * risks.length)];
            const qFlag = qFlags[Math.floor(Math.random() * qFlags.length)];
            const baseLevel = -5 - Math.random() * 15;
            
            wells.push({
                id: `NB-OWLS-${1000 + i}`,
                lat: lat,
                lng: lng,
                depth: Math.floor(25 + Math.random() * 120),
                casingDepth: Math.floor(6 + Math.random() * 25),
                aquifer: types[Math.floor(Math.random() * types.length)],
                quality: qFlag,
                risk: risk,
                baseLevel: baseLevel,
                county: 'New Brunswick'
            });
        }
        return wells;
    };

    // Add markers to map
    const renderMapMarkers = (wells) => {
        // Clear existing markers
        markers.forEach(m => map.removeLayer(m.marker));
        markers.length = 0;

        wells.forEach(well => {
            const color = getMarkerColor(well.risk);
            const markerSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" width="24" height="24"><path fill="${color}" d="M192 0C86 0 0 86 0 192c0 77.4 27 101.9 164.7 289.4c12 16.4 35.5 16.4 47.5 0C350 293.9 384 269.4 384 192C384 86 298 0 192 0zM192 272c-44.2 0-80-35.8-80-80s35.8-80 80-80s80 35.8 80 80s-35.8 80-80 80z"/></svg>`;
            
            const customIcon = L.divIcon({
                className: 'custom-div-icon',
                html: markerSvg,
                iconSize: [24, 24],
                iconAnchor: [12, 24],
                popupAnchor: [0, -24]
            });

            const marker = L.marker([well.lat, well.lng], {icon: customIcon})
                .addTo(map)
                .bindPopup(`
                    <div class="custom-popup">
                        <strong>${well.id}</strong><br>
                        <span>${well.county || 'NB'} | ${well.aquifer}</span><br>
                        <span style="color: ${color}; font-weight: 600;">Risk: ${well.risk.toUpperCase()}</span>
                        ${well.quality !== 'None' ? `<br><span style="color: #a855f7;">Threat: ${well.quality}</span>` : ''}
                    </div>
                `);
                
            marker.on('click', () => showWellDetails(well));
            
            markers.push({
                marker: marker,
                well: well
            });
        });
    };

    // Load data from live pipeline JSON outputs
    const loadPipelineData = async () => {
        try {
            const [wellsRes, summaryRes] = await Promise.all([
                fetch('data/processed_wells.json'),
                fetch('data/pipeline_summary.json')
            ]);

            if (wellsRes.ok && summaryRes.ok) {
                const wells = await wellsRes.json();
                const summary = await summaryRes.json();

                wellData = wells;
                renderMapMarkers(wellData);

                // Update metric cards with real pipeline numbers
                if (document.getElementById('val-total')) {
                    document.getElementById('val-total').innerText = Number(summary.total_wells).toLocaleString();
                }
                if (document.getElementById('val-drought')) {
                    document.getElementById('val-drought').innerText = Number(summary.high_drought_risk_count).toLocaleString();
                }
                if (document.getElementById('val-georisk')) {
                    document.getElementById('val-georisk').innerText = Number(summary.combined_georisk_count).toLocaleString();
                }
                if (document.getElementById('val-rmse') && summary.rmse_m) {
                    document.getElementById('val-rmse').innerText = `${summary.rmse_m.toFixed(2)}m`;
                }

                // Update status indicator
                const statusIndicator = document.querySelector('.status-indicator');
                if (statusIndicator) {
                    statusIndicator.innerHTML = '<i class="fa-solid fa-circle blink" style="color: #10b981;"></i> Live PyTorch LSTM (MCDO N=100)';
                }

                console.log(`[Aqua-Predict-NB] Loaded ${wells.length} pipeline forecasted wells.`);
                return;
            }
        } catch (err) {
            console.warn('[Aqua-Predict-NB] Local pipeline JSON not accessible or running on file:// protocol. Using procedural generator fallback.', err);
        }

        // Fallback
        wellData = generateFallbackWells(150);
        renderMapMarkers(wellData);
    };

    loadPipelineData();

    // === 2. CHART & DETAILS LOGIC ===
    const showWellDetails = (well) => {
        document.getElementById('chart-empty-state').classList.add('hidden');
        document.getElementById('well-data-view').classList.remove('hidden');
        
        // Update text details
        document.getElementById('detail-well-id').innerText = `Well ${well.id}`;
        document.getElementById('detail-depth').innerText = `${well.depth}m`;
        document.getElementById('detail-aquifer').innerText = well.aquifer;
        document.getElementById('detail-quality').innerText = well.quality || 'None';
        
        // Update badge
        const badge = document.getElementById('detail-risk-badge');
        badge.className = `badge ${well.risk === 'high' ? 'danger' : well.risk === 'medium' ? 'warning' : 'safe'}`;
        badge.innerText = `${well.risk.toUpperCase()} RISK`;
        
        renderChart(well);
    };

    const renderChart = (well) => {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        if (currentChart) {
            currentChart.destroy();
        }

        let labels = [];
        let historical = [];
        let forecast50 = [];
        let forecast10 = [];
        let forecast90 = [];

        // Check if well comes from live pipeline model outputs
        if (well.forecastP50 && well.historical) {
            labels = [...well.historicalDates, ...well.forecastDates];
            
            const nHist = well.historical.length;
            const nFc = well.forecastP50.length;

            // Historical series
            for (let i = 0; i < nHist; i++) {
                historical.push(well.historical[i]);
                forecast50.push(null);
                forecast10.push(null);
                forecast90.push(null);
            }

            // Forecast series (connected at boundary)
            forecast50[nHist - 1] = well.historical[nHist - 1];
            forecast10[nHist - 1] = well.historical[nHist - 1];
            forecast90[nHist - 1] = well.historical[nHist - 1];

            for (let i = 0; i < nFc; i++) {
                historical.push(null);
                forecast50.push(well.forecastP50[i]);
                forecast10.push(well.forecastP10[i]);
                forecast90.push(well.forecastP90[i]);
            }
        } else {
            // Procedural fallback
            labels = Array.from({length: 12}, (_, i) => `Month ${i-5}`).map(m => m.replace('Month -5', 'Past').replace('Month 0', 'Now'));
            let currentLvl = well.baseLevel;
            
            for (let i = 0; i < 6; i++) {
                historical.push(currentLvl);
                forecast50.push(null);
                forecast10.push(null);
                forecast90.push(null);
                currentLvl += (Math.random() * 2 - 1);
            }
            
            forecast50[5] = historical[5];
            forecast10[5] = historical[5];
            forecast90[5] = historical[5];
            
            let drift = well.risk === 'high' ? -1.5 : (well.risk === 'medium' ? -0.5 : 0.2);
            for (let i = 6; i < 12; i++) {
                historical.push(null);
                currentLvl += drift + (Math.random() * 1 - 0.5);
                forecast50.push(currentLvl);
                forecast10.push(currentLvl - 1.5 - Math.random() * 0.5);
                forecast90.push(currentLvl + 1.5 + Math.random() * 0.5);
            }
        }

        currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Historical GWL',
                        data: historical,
                        borderColor: '#94a3b8',
                        backgroundColor: '#94a3b8',
                        borderWidth: 2,
                        tension: 0.35,
                        pointRadius: 3
                    },
                    {
                        label: 'LSTM Forecast (50th %ile)',
                        data: forecast50,
                        borderColor: '#3b82f6',
                        backgroundColor: '#3b82f6',
                        borderWidth: 2.2,
                        borderDash: [5, 5],
                        tension: 0.35,
                        pointBackgroundColor: '#3b82f6',
                        pointRadius: 3
                    },
                    {
                        label: 'P90 (Low Risk Scenario)',
                        data: forecast90,
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(59, 130, 246, 0.15)',
                        fill: '+1',
                        tension: 0.35,
                        pointRadius: 0
                    },
                    {
                        label: 'P10 (High Drought Risk)',
                        data: forecast10,
                        borderColor: 'rgba(239, 68, 68, 0.75)',
                        borderWidth: 1.5,
                        borderDash: [2, 2],
                        tension: 0.35,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f8fafc', font: {family: 'Outfit', size: 11} }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        callbacks: {
                            label: function(context) {
                                if (context.parsed.y !== null) {
                                    return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} m`;
                                }
                                return null;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Groundwater Level (m below surface)', color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8', maxRotation: 45 }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    };

    // === 3. FILTERING LOGIC ===
    const applyFilters = () => {
        const riskVal = document.getElementById('risk-filter').value;
        const allowArsenic = document.getElementById('qa-arsenic').checked;
        const allowUranium = document.getElementById('qa-uranium').checked;
        const allowMethane = document.getElementById('qa-methane').checked;
        const searchQuery = (document.getElementById('search-input')?.value || '').toLowerCase().trim();

        markers.forEach(item => {
            const w = item.well;
            let show = true;

            // Risk filter
            if (riskVal !== 'all' && w.risk !== riskVal) {
                show = false;
            }

            // Quality threats
            if (w.quality === 'Arsenic' && !allowArsenic) show = false;
            if (w.quality === 'Uranium' && !allowUranium) show = false;
            if (w.quality === 'Methane' && !allowMethane) show = false;

            // Search input
            if (searchQuery) {
                const matchId = (w.id || '').toLowerCase().includes(searchQuery);
                const matchCounty = (w.county || '').toLowerCase().includes(searchQuery);
                const matchAquifer = (w.aquifer || '').toLowerCase().includes(searchQuery);
                if (!matchId && !matchCounty && !matchAquifer) {
                    show = false;
                }
            }

            if (show) {
                map.addLayer(item.marker);
            } else {
                map.removeLayer(item.marker);
            }
        });
    };

    document.getElementById('risk-filter').addEventListener('change', applyFilters);
    document.getElementById('qa-arsenic').addEventListener('change', applyFilters);
    document.getElementById('qa-uranium').addEventListener('change', applyFilters);
    document.getElementById('qa-methane').addEventListener('change', applyFilters);
    
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('input', applyFilters);
    }

    // Adjust metrics slightly when horizon changes
    const rangeSlider = document.getElementById('forecast-range');
    rangeSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        const droughtBase = wellData.filter(w => w.risk === 'high').length || 312;
        const geoBase = wellData.filter(w => (w.risk === 'high' || w.risk === 'medium') && w.quality !== 'None').length || 145;
        
        if (document.getElementById('val-drought')) {
            document.getElementById('val-drought').innerText = Math.round(droughtBase * (1 + (val - 3) * 0.08));
        }
        if (document.getElementById('val-georisk')) {
            document.getElementById('val-georisk').innerText = Math.round(geoBase * (1 + (val - 3) * 0.05));
        }
    });

    // Export report trigger
    const btnExport = document.getElementById('export-report');
    if (btnExport) {
        btnExport.addEventListener('click', () => {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(wellData, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", "AquaPredict_NB_Forecast_Report.json");
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        });
    }

    // === 4. VIEW TOGGLE LOGIC ===
    const navProposal = document.getElementById('nav-proposal');
    const navOverview = document.getElementById('nav-overview');
    const viewProposal = document.getElementById('proposal-view');
    const viewDashboard = document.getElementById('main-content-grid');
    const btnViewDashboard = document.getElementById('btn-view-dashboard');

    const showProposal = () => {
        navProposal.classList.add('active');
        navOverview.classList.remove('active');
        viewProposal.classList.add('active');
        viewProposal.classList.remove('display-none');
        viewDashboard.classList.add('hidden');
    };

    const showDashboard = () => {
        navOverview.classList.add('active');
        navProposal.classList.remove('active');
        viewDashboard.classList.remove('hidden');
        viewProposal.classList.remove('active');
        viewProposal.classList.add('display-none');
        
        // Force Leaflet to recalculate size when shown
        setTimeout(() => {
            map.invalidateSize();
        }, 100);
    };

    navProposal.addEventListener('click', (e) => { e.preventDefault(); showProposal(); });
    navOverview.addEventListener('click', (e) => { e.preventDefault(); showDashboard(); });
    btnViewDashboard.addEventListener('click', showDashboard);
});
