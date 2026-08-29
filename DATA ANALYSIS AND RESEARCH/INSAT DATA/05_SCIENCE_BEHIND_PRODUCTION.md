# The Science Behind Producing INSAT Data

## 1. Remote Sensing and Planck's Law
The fundamental science is Radiometry. Every object above absolute zero emits electromagnetic radiation. **Planck’s Law** states that the spectrum and intensity of this radiation depend completely on the object's temperature.
The INSAT sensors are designed to "look" through specific atmospheric windows—wavelengths where atmospheric gases do not block the light.

## 2. The Scanning Mechanism
INSAT does not take a single "picture" like a camera. It has a spinning/stepping mirror. 
- It sweeps horizontally across the earth (East to West), recording one thin line of pixels.
- The mirror tilts down a fraction of a degree (North to South).
- It sweeps back (West to East).
It takes ~27 minutes to scan the full disk of the Earth line by line.

## 3. Brightness Temperature Derivation
When the sensor detects a specific photon count (radiance) in the Thermal Infrared channel, computers apply the **Inverse Planck Function**. Because we know the exact wavelength the sensor looks at, we can mathematically reverse-engineer the exact physical temperature of the object (the cloud top) that emitted that photon.
