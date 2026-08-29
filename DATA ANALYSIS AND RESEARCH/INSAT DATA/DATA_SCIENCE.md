# The Science of INSAT: Reading the Infrared Sky

## 1. Brightness Temperature & Overshooting Tops
Thermal Infrared (TIR-1, 10.8 µm) measures the heat emitted by cloud tops. 
- In the troposphere, temperature decreases with altitude. Therefore, a colder cloud top means a taller cloud.
- During a severe Himalayan thunderstorm, violent updrafts push the cloud top all the way into the stratosphere (Tropopause Punch). INSAT registers these **Overshooting Tops** as extremely cold pixels (< 210 Kelvin or -63°C).

## 2. Convective Initiation
The Water Vapor (WV, 6.8 µm) channel is highly sensitive to mid-level moisture. Before a cloud even forms, INSAT can detect localized pooling of water vapor. The difference between WV and TIR brightness temperatures is a mathematical indicator of growing instability before the cloudburst begins.

## 3. The Hydro-Estimator Method (HEM) Physics
HEM doesn't just guess rain based on cold clouds. It applies physics:
- **Core Assumption:** Colder clouds precipitate more heavily than warmer clouds.
- **Orographic Correction:** HEM integrates topographical data. When winds force moist air up a mountain (orographic lift), HEM artificially boosts the precipitation estimate on the windward side, solving a major flaw in older satellite algorithms.
