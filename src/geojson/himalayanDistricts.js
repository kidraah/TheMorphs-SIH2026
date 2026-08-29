/**
 * Approximate GeoJSON district polygons for Uttarakhand & Himachal Pradesh.
 *
 * The `properties.districtKey` must match the `district` field
 * in districtRiskData.js — that is how choropleth colors are driven by data.
 *
 * Coordinates are approximate but geographically realistic for
 * 1:500,000 scale visualization.
 */

export const HIMALAYAN_DISTRICTS_GEOJSON = {
  type: 'FeatureCollection',
  features: [

    // ══════════════════ UTTARAKHAND ══════════════════

    // 1. Dehradun – west, lower foothills
    {
      type: 'Feature',
      properties: { districtKey: 'Dehradun', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [77.58, 30.65], [77.80, 30.72], [78.00, 30.78], [78.18, 30.70],
          [78.22, 30.38], [78.10, 30.10], [77.90, 30.08], [77.68, 30.15],
          [77.58, 30.40], [77.58, 30.65]
        ]]
      }
    },

    // 2. Haridwar – south-west plains
    {
      type: 'Feature',
      properties: { districtKey: 'Haridwar', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [77.70, 30.08], [78.10, 30.08], [78.25, 29.80], [78.10, 29.60],
          [77.80, 29.58], [77.65, 29.75], [77.70, 30.08]
        ]]
      }
    },

    // 3. Udham Singh Nagar – south-east Terai plains
    {
      type: 'Feature',
      properties: { districtKey: 'Udham Singh Nagar', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [78.65, 29.58], [79.35, 29.52], [79.45, 29.08], [78.60, 29.12],
          [78.40, 29.30], [78.55, 29.58], [78.65, 29.58]
        ]]
      }
    },

    // 4. Nainital – lower Kumaon hills
    {
      type: 'Feature',
      properties: { districtKey: 'Nainital', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [78.55, 29.60], [79.30, 29.55], [79.35, 29.08], [78.60, 29.12],
          [78.40, 29.32], [78.55, 29.60]
        ]]
      }
    },

    // 5. Champawat – lower east Kumaon
    {
      type: 'Feature',
      properties: { districtKey: 'Champawat', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [79.60, 29.55], [80.18, 29.48], [80.22, 29.08], [79.45, 29.10],
          [79.35, 29.30], [79.60, 29.55]
        ]]
      }
    },

    // 6. Pauri Garhwal – central Garhwal mid-hills
    {
      type: 'Feature',
      properties: { districtKey: 'Pauri Garhwal', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [78.22, 30.38], [78.72, 30.38], [79.10, 30.22], [79.15, 29.72],
          [78.55, 29.60], [78.10, 29.60], [78.10, 30.10], [78.22, 30.38]
        ]]
      }
    },

    // 7. Almora – central Kumaon hills
    {
      type: 'Feature',
      properties: { districtKey: 'Almora', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [79.15, 29.75], [79.75, 29.70], [79.82, 29.45], [79.35, 29.30],
          [79.10, 29.40], [79.05, 29.65], [79.15, 29.75]
        ]]
      }
    },

    // 8. Tehri Garhwal – upper Garhwal valleys
    {
      type: 'Feature',
      properties: { districtKey: 'Tehri Garhwal', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [78.18, 30.70], [78.72, 30.70], [78.80, 30.40], [78.72, 30.38],
          [78.22, 30.38], [78.10, 30.55], [78.18, 30.70]
        ]]
      }
    },

    // 9. Uttarkashi – northern Garhwal
    {
      type: 'Feature',
      properties: { districtKey: 'Uttarkashi', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [77.88, 31.35], [78.45, 31.42], [79.08, 31.18], [79.18, 30.75],
          [78.80, 30.70], [78.18, 30.70], [77.90, 30.90], [77.88, 31.35]
        ]]
      }
    },

    // 10. Rudraprayag – Kedarnath/Mandakini zone
    {
      type: 'Feature',
      properties: { districtKey: 'Rudraprayag', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [78.72, 30.70], [79.18, 30.75], [79.22, 30.30], [78.88, 30.22],
          [78.72, 30.40], [78.72, 30.70]
        ]]
      }
    },

    // 11. Chamoli – Badrinath / Nanda Devi zone (highest risk)
    {
      type: 'Feature',
      properties: { districtKey: 'Chamoli', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [79.18, 30.75], [79.42, 31.00], [79.70, 30.92], [80.05, 30.68],
          [80.02, 30.30], [79.62, 30.22], [79.22, 30.22], [79.18, 30.50],
          [79.18, 30.75]
        ]]
      }
    },

    // 12. Bageshwar – Saryu basin
    {
      type: 'Feature',
      properties: { districtKey: 'Bageshwar', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [79.62, 30.22], [80.05, 30.22], [80.08, 29.78], [79.78, 29.70],
          [79.50, 29.75], [79.45, 30.05], [79.62, 30.22]
        ]]
      }
    },

    // 13. Pithoragarh – Nepal/Tibet border high Himalaya
    {
      type: 'Feature',
      properties: { districtKey: 'Pithoragarh', state: 'Uttarakhand' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [80.05, 30.60], [80.52, 30.32], [80.55, 29.72], [80.20, 29.52],
          [80.00, 29.48], [79.82, 29.70], [80.05, 29.80], [80.02, 30.30],
          [80.05, 30.60]
        ]]
      }
    },

    // ══════════════════ HIMACHAL PRADESH ══════════════════

    // 14. Una – south-west foothills
    {
      type: 'Feature',
      properties: { districtKey: 'Una', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [75.88, 31.82], [76.35, 31.88], [76.40, 31.45], [76.00, 31.38],
          [75.82, 31.55], [75.88, 31.82]
        ]]
      }
    },

    // 15. Hamirpur – south-west
    {
      type: 'Feature',
      properties: { districtKey: 'Hamirpur', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [76.35, 31.88], [76.75, 31.92], [76.80, 31.55], [76.42, 31.48],
          [76.35, 31.88]
        ]]
      }
    },

    // 16. Bilaspur – Gobind Sagar reservoir area
    {
      type: 'Feature',
      properties: { districtKey: 'Bilaspur', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [76.42, 31.48], [76.80, 31.55], [76.88, 31.28], [76.72, 31.10],
          [76.45, 31.15], [76.42, 31.48]
        ]]
      }
    },

    // 17. Kangra – Dhauladhar foothills (Palampur, Dharamshala)
    {
      type: 'Feature',
      properties: { districtKey: 'Kangra', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [75.82, 32.55], [76.38, 32.68], [76.82, 32.42], [76.92, 31.92],
          [76.75, 31.92], [76.40, 31.88], [76.00, 31.90], [75.82, 32.10],
          [75.82, 32.55]
        ]]
      }
    },

    // 18. Mandi – Uhl river, Beas valley
    {
      type: 'Feature',
      properties: { districtKey: 'Mandi', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [76.80, 31.55], [76.92, 31.92], [77.12, 32.05], [77.35, 31.88],
          [77.42, 31.55], [77.28, 31.20], [76.88, 31.28], [76.80, 31.55]
        ]]
      }
    },

    // 19. Kullu – Parvati valley, Manali, Great Himalayan National Park
    {
      type: 'Feature',
      properties: { districtKey: 'Kullu', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [77.12, 32.05], [77.35, 32.40], [77.60, 32.30], [77.72, 31.92],
          [77.42, 31.55], [77.12, 31.68], [77.05, 31.88], [77.12, 32.05]
        ]]
      }
    },

    // 20. Shimla – capital district, Giri valley
    {
      type: 'Feature',
      properties: { districtKey: 'Shimla', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [76.72, 31.10], [76.88, 31.28], [77.28, 31.20], [77.42, 31.02],
          [77.48, 30.78], [77.20, 30.65], [77.00, 30.72], [76.80, 30.88],
          [76.72, 31.10]
        ]]
      }
    },

    // 21. Solan – Kasauli area, Pinjore
    {
      type: 'Feature',
      properties: { districtKey: 'Solan', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [76.45, 31.15], [76.72, 31.10], [76.80, 30.88], [76.70, 30.65],
          [76.42, 30.62], [76.30, 30.88], [76.45, 31.15]
        ]]
      }
    },

    // 22. Sirmaur – Giri Ganga, border with Uttarakhand
    {
      type: 'Feature',
      properties: { districtKey: 'Sirmaur', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [76.70, 30.65], [76.80, 30.88], [77.00, 30.72], [77.20, 30.65],
          [77.30, 30.38], [77.05, 30.28], [76.72, 30.30], [76.70, 30.65]
        ]]
      }
    },

    // 23. Chamba – Ravi river high Himalayas
    {
      type: 'Feature',
      properties: { districtKey: 'Chamba', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [75.82, 32.55], [76.38, 32.68], [76.82, 32.42], [77.10, 32.72],
          [77.40, 32.88], [77.60, 32.65], [77.40, 32.30], [77.35, 32.40],
          [77.12, 32.05], [76.92, 31.92], [76.82, 32.00], [76.42, 32.38],
          [75.82, 32.55]
        ]]
      }
    },

    // 24. Kinnaur – Sutlej gorge, China border
    {
      type: 'Feature',
      properties: { districtKey: 'Kinnaur', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [77.48, 30.78], [77.72, 31.05], [78.08, 31.32], [78.45, 31.42],
          [78.45, 31.10], [78.18, 30.82], [78.00, 30.62], [77.68, 30.48],
          [77.48, 30.78]
        ]]
      }
    },

    // 25. Lahaul & Spiti – cold desert, Trans-Himalayan plateau
    {
      type: 'Feature',
      properties: { districtKey: 'Lahaul & Spiti', state: 'Himachal Pradesh' },
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [77.35, 32.40], [77.60, 32.65], [77.88, 32.72], [78.40, 32.55],
          [78.60, 32.20], [78.45, 31.80], [78.08, 31.32], [77.72, 31.05],
          [77.42, 31.55], [77.35, 31.88], [77.35, 32.40]
        ]]
      }
    },
  ]
};
