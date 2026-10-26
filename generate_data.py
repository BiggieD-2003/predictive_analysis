import pandas as pd
import numpy as np
import os

def generate_mock_data(num_records=1000):
    np.random.seed(42)
    
    machine_ids = [f"M_{i:04d}" for i in range(1, num_records + 1)]
    machine_types = np.random.choice(['Type_A', 'Type_B', 'Type_C'], size=num_records)
    
    # Simulate normal ranges
    air_temp = np.random.normal(300, 2, num_records)
    process_temp = air_temp + np.random.normal(10, 1, num_records)
    rotational_speed = np.random.normal(1500, 50, num_records)
    torque = np.random.normal(40, 5, num_records)
    tool_wear = np.random.randint(0, 250, num_records)
    
    operating_hours = np.random.randint(100, 10000, num_records)
    planned_production_time = np.random.randint(400, 500, num_records)
    actual_running_time = planned_production_time - np.random.randint(0, 50, num_records)
    
    jobs_completed = actual_running_time * np.random.uniform(0.8, 1.2, num_records)
    
    # Simulate failures based on some logic (high temperatures, high torque, high wear)
    failure_prob = (
        (tool_wear > 200) * 0.4 +
        (torque > 50) * 0.3 +
        (air_temp > 304) * 0.2 +
        (actual_running_time < planned_production_time * 0.8) * 0.3
    )
    failure = (np.random.random(num_records) < failure_prob).astype(int)
    
    df = pd.DataFrame({
        'machine_id': machine_ids,
        'machine_type': machine_types,
        'air_temperature': air_temp,
        'process_temperature': process_temp,
        'rotational_speed': rotational_speed,
        'torque': torque,
        'tool_wear': tool_wear,
        'operating_hours': operating_hours,
        'planned_production_time': planned_production_time,
        'actual_running_time': actual_running_time,
        'jobs_completed': jobs_completed.astype(int),
        'failure': failure
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/machine_data.csv', index=False)
    print("Mock data generated at 'data/machine_data.csv'")

if __name__ == "__main__":
    generate_mock_data()
