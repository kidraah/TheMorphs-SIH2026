# What Does INSAT Show?

INSAT does not measure "rain" or "wind" directly. It measures **radiance** (light and heat energy). 

## 1. The Imager Channels
- **Visible (VIS):** Shows exactly what a human eye would see. Only works during the day. Shows cloud thickness and extent.
- **Thermal Infrared (TIR 1 & 2):** Shows the temperature of the objects it looks at. Because the atmosphere gets colder the higher you go, TIR effectively shows the *altitude* of cloud tops. Cold = High. Warm = Low. Works day and night.
- **Water Vapor (WV):** Sensitive specifically to the radiation emitted by water vapor molecules in the mid-to-upper atmosphere. Shows the flow of moisture even when no clouds are present.
- **Short/Mid-Wave Infrared (SWIR/MIR):** Used for detecting fog, snow cover, and forest fires.

## 2. Derived Level-2 Products
By running math on the raw radiances, INSAT shows:
- **Cloud Top Temperature (CTT):** Exact temperature of the highest cloud pixel.
- **Quantitative Precipitation Estimation (QPE):** An estimate of rainfall rate (mm/hr).
- **Atmospheric Motion Vectors (AMV):** By tracking clouds between two frames (e.g., at 10:00 and 10:15), it calculates wind speed and direction.
