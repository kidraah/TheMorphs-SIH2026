# Detailed Phenomena and Science of INSAT

## 1. Overshooting Tops & The Tropopause Punch
The troposphere (where weather happens) gets colder as you go up. It hits a boundary called the Tropopause (~15km high), above which the stratosphere gets warmer. 
In a normal thunderstorm, the updraft hits the tropopause, loses buoyancy, and flattens out into an "anvil" shape (around -50°C). 
In a **cloudburst**, the updraft is so violent that its momentum punches *through* the tropopause into the stratosphere. INSAT detects this as a tiny cluster of extremely cold pixels (<-65°C) surrounded by warmer anvil pixels. This is the **Overshooting Top**, the absolute clearest signature of deadly, catastrophic rainfall.

## 2. Convective Initiation (Water Vapor - Infrared Difference)
Before a cloud forms, moist air rises. The WV channel detects this moisture high up. If the Brightness Temperature of the WV channel becomes *warmer* than the TIR channel, it mathematically proves that an active, explosive updraft is occurring. This allows INSAT algorithms to predict a severe storm 30-45 minutes before radar even sees the first raindrop.

## 3. The Hydro-Estimator Method (HEM) Physics
HEM estimates rain from space using thermodynamic rules:
- **Rule 1:** Colder clouds are taller. Taller clouds hold more water and rain harder.
- **Rule 2 (Parallax):** Satellites view clouds from an angle. A 15km high cloud appears shifted on the map compared to where rain hits the ground. HEM uses trigonometry to correct this shift.
- **Rule 3 (Orographic Lift):** HEM ingests elevation data. If the wind is blowing up a mountain, HEM mathematically amplifies the rain estimate, knowing the mountain is squeezing extra water out of the cloud.
