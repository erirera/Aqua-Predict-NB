# NB AI Groundwater Risk Dashboard (Proposal 3)

This repository contains an interactive web dashboard for visualizing AI-driven groundwater level forecasting in New Brunswick (NB), based on "AI Research Proposal 3: Groundwater Level Prediction for NB Private Wells".

## 🚀 Features

*   **Interactive Spatial Map:** Displays private wells across New Brunswick using Leaflet.js
*   **Risk Categorization:** Color-coded markers indicating groundwater drought risk (High = Red, Medium = Orange, Low = Green).
*   **AI Forecast Visualization:** Interactive time-series prediction charts built with Chart.js, simulating LSTM forecasts (1-6 months ahead) with 10th and 90th percentile uncertainty bounds.
*   **Water Quality Overlays:** Filter toggles to simulate combined geological risk layers (Arsenic, Uranium, Methane).
*   **Modern Aesthetics:** Premium "Glassmorphism" UI with smooth gradients, responsive layouts, and intuitive controls.

## 🛠️ Technology Stack

*   **HTML5** (Semantic structure)
*   **CSS3** (Variables, Grid/Flexbox layouts, Backdrop-filters)
*   **Vanilla JavaScript (ES6)**
*   **Leaflet.js** (Mapping and Geo-spatial plotting)
*   **Chart.js** (Data visualization & Line charts)
*   **FontAwesome** (Scalable vector icons)

## 📖 How to Run

Because this is a static, client-side web application, you do not need to install any heavy dependencies or set up a backend server. 

1.  Clone or download this project.
2.  Open the `index.html` file directly in any modern web browser (Chrome, Edge, Firefox, Safari).
3.  Ensure you have an active internet connection so the external CDNs (Leaflet, Chart.js, Google Fonts, FontAwesome) can load correctly.

## 📂 Project Structure

*   `index.html` - The main layout, containing the sidebar, mapping container, and chart sections.
*   `styles.css` - Custom styling, application theme variables, and responsive design queries.
*   `script.js` - Logic for initializing Leaflet, generating random well data, managing interactivity (filters, clicks), and rendering Chart.js graphs.

## 📝 Future Implementation (Data Pipeline)
Currently, the map plots randomized mock data to demonstrate the dashboard's capabilities. A full implementation would involve:
1.  Fetching API data compiled from the NB OWLS (Online Well Log System).
2.  Aligning with ECCC Meteorological driver data.
3.  Deploying a true PyTorch LSTM inference engine to replace the random-walk script logic.
