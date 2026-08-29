# Importance and Necessity of IMDAA

## 1. Why Do We Need It?
We need IMDAA because observations alone are incomplete. A rain gauge only tells you it rained at one exact point. A satellite only shows the cloud top. If you want to know the wind shear 5 kilometers above a mountain valley where no sensor exists, you *must* use a reanalysis dataset.

## 2. Why Not Use Global Datasets? (The Himalayan Problem)
Global datasets like ERA5 (ECMWF) have a resolution of ~31 km. 
In the Himalayas, a mountain peak and a deep valley can exist within 5 km of each other. A 31 km grid square completely smooths out the mountain. The global model thinks the Himalayas are flat, rolling hills. 
IMDAA, with its 12 km resolution, captures the steep topography (orography) much better. It understands that wind hitting a steep valley wall will be forced upwards, which is the exact trigger for localized cloudbursts.

## 3. Importance in AI
AI models cannot be trained on raw, messy observations with gaps. They require clean, continuous, normalized grid tensors. IMDAA provides the perfect `X` (input feature) tensor for training neural networks.
