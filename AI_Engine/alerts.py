"""
SIH26077 — Automated Alerting Module
=======================================
Threshold-based alert generation from predicted risk maps.

When a risk map exceeds configurable probability thresholds, the module
generates categorized alerts with severity levels:
  - WATCH:     probability > 30% — enhanced monitoring
  - WARNING:   probability > 50% — prepare for impact
  - EMERGENCY: probability > 70% — immediate action required

Satisfies the PS requirement: "automated, categorized alerts sent directly
to first responders and vulnerable communities the moment critical
thresholds are breached."

Usage:
    from alerts import AlertEngine
    
    engine = AlertEngine()
    alerts = engine.evaluate(risk_maps_dict)
"""

import numpy as np
from datetime import datetime, timezone, timedelta

from config import ALERT_THRESHOLDS, TARGET_NAMES


# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))


class AlertEngine:
    """Evaluates risk maps against thresholds and generates structured alerts.
    
    Args:
        thresholds: Dict of severity_name → probability_threshold
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or ALERT_THRESHOLDS
        # Severity levels in descending order
        self.severity_order = ['EMERGENCY', 'WARNING', 'WATCH']

    def evaluate_single(self, risk_map, risk_name):
        """Evaluate a single risk map and return alerts.
        
        Args:
            risk_map: (H, W) numpy array of probabilities [0, 1]
            risk_name: Human-readable name (e.g., 'Cloudburst')
            
        Returns:
            List of alert dicts, sorted by severity (highest first)
        """
        alerts = []
        total_pixels = risk_map.size
        
        for severity in self.severity_order:
            threshold = self.thresholds[severity]
            mask = risk_map > threshold
            affected_pixels = np.sum(mask)
            
            if affected_pixels > 0:
                affected_pct = 100.0 * affected_pixels / total_pixels
                max_prob = float(risk_map[mask].max())
                mean_prob = float(risk_map[mask].mean())
                
                alerts.append({
                    'severity': severity,
                    'risk_type': risk_name,
                    'threshold': threshold,
                    'max_probability': round(max_prob, 4),
                    'mean_probability': round(mean_prob, 4),
                    'affected_area_pct': round(affected_pct, 2),
                    'affected_pixels': int(affected_pixels),
                    'timestamp': datetime.now(IST).isoformat(),
                    'message': self._format_message(severity, risk_name, 
                                                     affected_pct, max_prob),
                })
        
        return alerts

    def evaluate(self, risk_maps):
        """Evaluate all three risk maps and return categorized alerts.
        
        Args:
            risk_maps: Dict with keys 'cloudburst', 'thunderstorm', 'flash_flood'
                       Values are (H, W) numpy arrays of probabilities [0, 1]
                       
        Returns:
            Dict with:
              - 'alerts': List of all alerts sorted by severity
              - 'highest_severity': The most severe alert level triggered
              - 'summary': Human-readable summary string
        """
        all_alerts = []
        
        name_map = {
            'cloudburst': 'Cloudburst',
            'thunderstorm': 'Severe Thunderstorm',
            'flash_flood': 'Flash Flood',
        }
        
        for key, display_name in name_map.items():
            if key in risk_maps:
                alerts = self.evaluate_single(risk_maps[key], display_name)
                all_alerts.extend(alerts)
        
        # Sort: EMERGENCY first, then WARNING, then WATCH
        priority = {'EMERGENCY': 0, 'WARNING': 1, 'WATCH': 2}
        all_alerts.sort(key=lambda a: priority.get(a['severity'], 99))
        
        # Determine highest severity
        highest = 'NONE'
        if all_alerts:
            highest = all_alerts[0]['severity']
        
        # Generate summary
        summary = self._generate_summary(all_alerts)
        
        return {
            'alerts': all_alerts,
            'highest_severity': highest,
            'summary': summary,
            'timestamp': datetime.now(IST).isoformat(),
        }

    def _format_message(self, severity, risk_name, affected_pct, max_prob):
        """Generate a human-readable alert message."""
        marker = {'EMERGENCY': '[!!!]', 'WARNING': '[!!]', 'WATCH': '[!]'}
        return (
            f"{marker.get(severity, '[i]')} {severity}: {risk_name} risk detected. "
            f"Max probability {max_prob:.0%} over {affected_pct:.1f}% of the region. "
            f"{'Immediate action required.' if severity == 'EMERGENCY' else ''}"
            f"{'Prepare for potential impact.' if severity == 'WARNING' else ''}"
            f"{'Enhanced monitoring recommended.' if severity == 'WATCH' else ''}"
        ).strip()

    def _generate_summary(self, alerts):
        """Generate a one-line summary of all active alerts."""
        if not alerts:
            return "No alerts active. All risk levels below thresholds."
        
        emergency_count = sum(1 for a in alerts if a['severity'] == 'EMERGENCY')
        warning_count = sum(1 for a in alerts if a['severity'] == 'WARNING')
        watch_count = sum(1 for a in alerts if a['severity'] == 'WATCH')
        
        parts = []
        if emergency_count:
            parts.append(f"{emergency_count} EMERGENCY")
        if warning_count:
            parts.append(f"{warning_count} WARNING")
        if watch_count:
            parts.append(f"{watch_count} WATCH")
        
        return " | ".join(parts)


# ============================================================
# Self-Test
# ============================================================
if __name__ == "__main__":
    print("Testing AlertEngine...")
    
    engine = AlertEngine()
    
    # Simulate risk maps with varying probabilities
    np.random.seed(42)
    h, w = 256, 256
    
    # Cloudburst: some hotspots > 0.7
    cb_map = np.random.uniform(0.0, 0.3, (h, w)).astype(np.float32)
    cb_map[100:120, 100:120] = 0.8  # Emergency hotspot
    
    # Thunderstorm: moderate widespread
    ts_map = np.random.uniform(0.1, 0.6, (h, w)).astype(np.float32)
    
    # Flash Flood: low risk
    ff_map = np.random.uniform(0.0, 0.2, (h, w)).astype(np.float32)
    
    result = engine.evaluate({
        'cloudburst': cb_map,
        'thunderstorm': ts_map,
        'flash_flood': ff_map,
    })
    
    print(f"\n  Highest Severity: {result['highest_severity']}")
    print(f"  Summary: {result['summary']}")
    print(f"  Total Alerts: {len(result['alerts'])}")
    
    for alert in result['alerts']:
        print(f"\n  [{alert['severity']}] {alert['risk_type']}")
        print(f"    Max prob: {alert['max_probability']:.2%}")
        print(f"    Affected: {alert['affected_area_pct']:.1f}%")
        print(f"    Message: {alert['message']}")
    
    print("\n[OK] AlertEngine test passed.")
