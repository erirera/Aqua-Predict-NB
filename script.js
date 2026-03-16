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

    // === 2. DUMMY DATA GENERATION ===
    // Generate some random wells in NB with varying risks
    const generateWells = (count) => {
        const wells = [];
        const types = ['Fractured Bedrock', 'Sand/Gravel', 'Karst'];
        const qFlags = ['None', 'Arsenic', 'Uranium', 'Methane', 'Arsenic, Uranium'];
        const risks = ['low', 'medium', 'high'];
        
        for (let i = 0; i < count; i++) {
            // Random coordinates within NB rough bounding box
            const lat = 45.0 + Math.random() * 2.8;
            const lng = -68.0 + Math.random() * 4.0;
            
            const risk = risks[Math.floor(Math.random() * risks.length)];
            const qFlag = qFlags[Math.floor(Math.random() * qFlags.length)];
            
            // Historical baseline GWL values
            const baseLevel = -5 - Math.random() * 20; 
            
            wells.push({
                id: `NB-OWLS-${1000 + i}`,
                lat: lat,
                lng: lng,
                depth: Math.floor(20 + Math.random() * 150),
                aquifer: types[Math.floor(Math.random() * types.length)],
                quality: qFlag,
                risk: risk,
                baseLevel: baseLevel
            });
        }
        return wells;
    };

    const wellData = generateWells(150);
    const markers = [];

    // Define custom marker colors based on risk
    const getMarkerColor = (risk) => {
        if (risk === 'high') return '#ef4444'; // Red
        if (risk === 'medium') return '#f59e0b'; // Orange
        return '#10b981'; // Green
    };

    // Add markers to map
    wellData.forEach(well => {
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
                    <span>Risk: ${well.risk.toUpperCase()}</span>
                </div>
            `);
            
        marker.on('click', () => showWellDetails(well));
        
        markers.push({
            marker: marker,
            well: well
        });
    });

    // === 3. CHART & DETAILS LOGIC ===
    let currentChart = null;

    const showWellDetails = (well) => {
        document.getElementById('chart-empty-state').classList.add('hidden');
        document.getElementById('well-data-view').classList.remove('hidden');
        
        // Update text details
        document.getElementById('detail-well-id').innerText = `Well ${well.id}`;
        document.getElementById('detail-depth').innerText = `${well.depth}m`;
        document.getElementById('detail-aquifer').innerText = well.aquifer;
        document.getElementById('detail-quality').innerText = well.quality;
        
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

        // Generate semi-random time series data that looks like LSTM forecast with uncertainty
        const labels = Array.from({length: 12}, (_, i) => `Month ${i-5}`).map(m => m.replace('Month -5', 'Past').replace('Month 0', 'Now'));
        
        const historical = [];
        const forecast50 = [];
        const forecast10 = [];
        const forecast90 = [];
        
        let currentLvl = well.baseLevel;
        
        for(let i=0; i<6; i++) {
            historical.push(currentLvl);
            forecast50.push(null);
            forecast10.push(null);
            forecast90.push(null);
            currentLvl += (Math.random() * 2 - 1); // Random walk
        }
        
        // Connect historical and forecast visually
        forecast50[5] = historical[5];
        forecast10[5] = historical[5];
        forecast90[5] = historical[5];
        
        let drift = well.risk === 'high' ? -1.5 : (well.risk === 'medium' ? -0.5 : 0.2);
        
        for(let i=6; i<12; i++) {
            historical.push(null);
            currentLvl += drift + (Math.random() * 1 - 0.5);
            forecast50.push(currentLvl);
            forecast10.push(currentLvl - 2 - Math.random());
            forecast90.push(currentLvl + 2 + Math.random());
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
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'LSTM Forecast (50th %ile)',
                        data: forecast50,
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.4,
                        pointBackgroundColor: '#3b82f6'
                    },
                    {
                        label: 'P90 (Low Risk Scenario)',
                        data: forecast90,
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: '+1',
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'P10 (High Drought Risk)',
                        data: forecast10,
                        borderColor: 'rgba(239, 68, 68, 0.5)',
                        borderWidth: 1,
                        borderDash: [2, 2],
                        tension: 0.4,
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
                        labels: { color: '#f8fafc', font: {family: 'Outfit'} }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(15, 23, 42, 0.9)'
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
                        ticks: { color: '#94a3b8' }
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

    // === 4. FILTERING LOGIC ===
    const riskFilter = document.getElementById('risk-filter');
    riskFilter.addEventListener('change', (e) => {
        const val = e.target.value;
        markers.forEach(item => {
            if (val === 'all' || item.well.risk === val) {
                map.addLayer(item.marker);
            } else {
                map.removeLayer(item.marker);
            }
        });
    });

    // Adjust metrics slightly when horizon changes just as a visual cue
    const rangeSlider = document.getElementById('forecast-range');
    rangeSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        document.getElementById('val-drought').innerText = 300 + Math.floor(val * 4.5);
        document.getElementById('val-georisk').innerText = 140 + Math.floor(val * 2.1);
    });

    // === 5. VIEW TOGGLE LOGIC ===
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

