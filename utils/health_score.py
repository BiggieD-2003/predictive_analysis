import numpy as np

def calculate_health_score(row):
    """
    Calculate a health score (0-100) using:
    temperature, torque, tool wear, downtime
    """
    max_temp = 310.0
    max_torque = 60.0
    max_tool_wear = 250.0
    planned = row.get('planned_production_time', 480.0)
    max_downtime = planned * 0.2

    air_temp = row.get('air_temperature', 300.0)
    torque = row.get('torque', 40.0)
    tool_wear = row.get('tool_wear', 0.0)
    downtime = row.get('downtime', 0.0)

    temp_penalty = min(max(0.0, (air_temp - 295.0) / (max_temp - 295.0)), 1.0)
    torque_penalty = min(max(0.0, torque / max_torque), 1.0)
    wear_penalty = min(max(0.0, tool_wear / max_tool_wear), 1.0)
    downtime_penalty = min(max(0.0, downtime / max_downtime), 1.0) if max_downtime > 0 else 0.0

    total_penalty = (0.30 * temp_penalty) + (0.20 * torque_penalty) + (0.30 * wear_penalty) + (0.20 * downtime_penalty)
    health_score = max(0.0, min(100.0, 100.0 - (total_penalty * 100.0)))
    return round(health_score, 2)

def get_health_category(score):
    """Determine category based on health score."""
    if score >= 80:
        return 'Healthy'
    elif score >= 60:
        return 'Warning'
    return 'Critical'

def get_maintenance_recommendation(probability):
    """Map prediction probability to recommendation."""
    if probability >= 0.70:
        return "High Priority: Immediate inspection/shutdown required within 24h"
    elif probability >= 0.40:
        return "Medium Priority: Maintenance inspection recommended within 3 shifts"
    return "Low Priority: Nominal operating condition (routine cycle)"

def diagnose_failure_cause(row):
    """
    Physics-informed root cause diagnostic engine that inspects
    thermodynamic and mechanical bounds to explain why machine risk is elevated.
    """
    causes = []

    air_temp = float(row.get('air_temperature', 300.0))
    proc_temp = float(row.get('process_temperature', 310.0))
    temp_diff = proc_temp - air_temp

    speed = float(row.get('rotational_speed', 1500.0))
    torque = float(row.get('torque', 40.0))
    wear = float(row.get('tool_wear', 0.0))

    power = torque * (speed * 2 * np.pi / 60)
    overstrain = wear * torque

    # Check thermal dissipation (HDF)
    if temp_diff < 8.6 and speed < 1380:
        causes.append({
            'mode': 'Heat Dissipation Failure (HDF)',
            'severity': 'Critical',
            'detail': f'Thermal delta ({temp_diff:.1f} K < 8.6 K) at low speed ({speed:.0f} RPM < 1380) causes severe overheating.'
        })

    # Check power limits (PWF)
    if power < 3500 or power > 9000:
        causes.append({
            'mode': 'Power Overload/Anomaly (PWF)',
            'severity': 'Critical',
            'detail': f'Mechanical power ({power:.0f} W) outside safe operational zone [3500 W - 9000 W].'
        })

    # Check overstrain (OSF)
    if overstrain > 11000:
        causes.append({
            'mode': 'Mechanical Overstrain (OSF)',
            'severity': 'Critical',
            'detail': f'Overstrain product ({overstrain:.0f} min·Nm) exceeds critical structural safety limit (11,000 min·Nm).'
        })

    # Check tool wear life (TWF)
    if wear >= 200:
        causes.append({
            'mode': 'Critical Tool Wear (TWF)',
            'severity': 'High',
            'detail': f'Tool wear ({wear:.0f} min) has reached or exceeded maximum rated lifetime (200 min).'
        })

    # High torque check
    if torque > 55:
        causes.append({
            'mode': 'High Spindle Torque',
            'severity': 'Medium',
            'detail': f'Torque ({torque:.1f} Nm) is abnormally elevated above standard 55 Nm limit.'
        })

    return causes
