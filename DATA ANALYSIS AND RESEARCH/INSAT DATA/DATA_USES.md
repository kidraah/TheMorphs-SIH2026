# Application of INSAT in Nowcasting Severe Weather

## 1. Real-Time Cloudburst Detection
Because cloudbursts develop in under an hour, INSAT's 15-30 minute refresh rate is the primary trigger for the Early Warning System. 
- **Tracking CTT Drop Rate:** Algorithms track the spatial cluster of a cloud and measure its CTT over time. A drop of >10 Kelvin in 30 minutes triggers an immediate deep-convection alert.

## 2. QPE (Quantitative Precipitation Estimation)
Ground rain gauges in the Himalayas are sparse and often destroyed during flash floods. The **HEM (Hydro-Estimator Method)** uses INSAT TIR channels to estimate rain rates (mm/hr) pixel by pixel, providing the "ground truth" for how much water is actually falling over a remote valley.

## 3. AI Model Validation
In building the SIH26077 ML Pipeline, INSAT HEM data serves as the `Y` (target variable) that the neural network is trained to predict, while IMDAA data serves as the `X` (input features).
