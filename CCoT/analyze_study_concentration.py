"""
Analyze student study concentration using multimodal sensor data with Chain of Thought reasoning
Processes sensor data (heart rate, noise level, steps) and provides personalized feedback
"""
import json
import os
from dotenv import load_dotenv
from snowflake.snowpark import Session
from datetime import datetime, timezone


def calculate_sensor_averages(sensor_data):
    """
    Calculate average values for each sensor type
    
    Args:
        sensor_data: List of sensor readings
    
    Returns:
        Dictionary with average values for each sensor type
    """
    heart_rates = []
    noise_levels = []
    steps = []
    
    for reading in sensor_data:
        if reading['sensor_type'] == 'heart_rate':
            heart_rates.append(reading['value'])
        elif reading['sensor_type'] == 'noise_level':
            noise_levels.append(reading['value'])
        elif reading['sensor_type'] == 'number_of_steps_past_minute':
            steps.append(reading['value'])
    
    return {
        'avg_heart_rate': sum(heart_rates) / len(heart_rates) if heart_rates else 0,
        'avg_noise_level': sum(noise_levels) / len(noise_levels) if noise_levels else 0,
        'avg_steps': sum(steps) / len(steps) if steps else 0,
        'num_readings': len(sensor_data) // 3  # Total time points
    }


def analyze_concentration(sensor_data, model="claude-3-5-sonnet", session=None):
    """
    Use LLM with Chain of Thought to analyze student concentration levels
    
    Args:
        sensor_data: List of sensor readings
        model: Model name to use
        session: Snowflake session
    
    Returns:
        LLM's concentration analysis with reasoning and feedback
    """
    # Calculate averages
    stats = calculate_sensor_averages(sensor_data)
    
    # Format sensor data for context
    sensor_summary = ""
    current_time = None
    for reading in sensor_data:
        if reading['timestamp'] != current_time:
            if current_time is not None:
                sensor_summary += "\n"
            current_time = reading['timestamp']
            time_str = datetime.fromisoformat(reading['timestamp'].replace('Z', '+00:00')).strftime('%H:%M')
            sensor_summary += f"\n{time_str}:"
        sensor_summary += f" {reading['sensor_type']}={reading['value']}{reading['unit']}"
    
    # Build concise analysis prompt
    prompt = f"""You are a study concentration analyst. Analyze sensor data and provide a brief assessment.

**SENSOR DATA:**
- Duration: {stats['num_readings']} minutes
- Average Heart Rate: {stats['avg_heart_rate']:.1f} bpm
- Average Noise Level: {stats['avg_noise_level']:.1f} dB
- Average Steps/min: {stats['avg_steps']:.1f} steps

**THRESHOLDS:**
Heart Rate: 60-80 (optimal), 80-90 (engaged), >90 (stressed), <60 (drowsy)
Noise: 30-45 dB (ideal), 45-60 dB (moderate), >60 dB (disruptive)
Movement: 0-20 (focused), 20-40 (breaks), 40-60 (restless), >60 (active)

**TASK:** Provide a concise assessment. Return JSON ONLY with this structure:
{{
  "concentration_level": "GOOD|MODERATE|POOR",
  "summary": "1-2 sentence overall assessment",
  "key_findings": ["finding 1", "finding 2"],
  "recommendations": ["action 1", "action 2"]
}}"""
    
    # Escape single quotes for SQL
    escaped_prompt = prompt.replace("'", "''")
    
    # Use Snowflake Cortex Complete
    query = f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        '{model}',
        '{escaped_prompt}'
    ) AS response
    """
    
    try:
        result = session.sql(query).collect()
        if result:
            return result[0]['RESPONSE']
        return "[ERROR] No response from model"
    except Exception as e:
        return f"[ERROR] {str(e)}"


def process_concentration_analysis(data_path, model="claude-3-5-sonnet", output_path="concentration_analysis_results.json"):
    """
    Process sensor data and analyze study concentration
    
    Args:
        data_path: Path to sensor data JSON file
        model: Model name to use
        output_path: Output file path for results
    """
    # Load environment and connect to Snowflake
    load_dotenv()
    
    connection_params = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_USER_PASSWORD"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE")
    }
    
    print("=" * 100)
    print("Study Concentration Analysis - Multimodal Sensor Data Analysis")
    print("=" * 100)
    print("\nConnecting to Snowflake...")
    session = Session.builder.configs(connection_params).create()
    print("✓ Connected to Snowflake\n")
    
    # Load sensor data
    print(f"Loading sensor data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        sensor_data = json.load(f)
    print(f"✓ Loaded {len(sensor_data)} sensor readings\n")
    
    # Calculate statistics
    stats = calculate_sensor_averages(sensor_data)
    
    print("📊 Sensor Data Summary:")
    print(f"   Duration: {stats['num_readings']} minutes")
    print(f"   Avg Heart Rate: {stats['avg_heart_rate']:.1f} bpm")
    print(f"   Avg Noise Level: {stats['avg_noise_level']:.1f} dB")
    print(f"   Avg Steps/min: {stats['avg_steps']:.1f} steps")
    print()
    
    print(f"🤖 Analyzing concentration with model: {model}")
    print("=" * 100)
    print()
    
    try:
        # Analyze concentration
        analysis = analyze_concentration(
            sensor_data=sensor_data,
            model=model,
            session=session
        )
        
        print("📋 CONCENTRATION ANALYSIS RESULTS:")
        print("=" * 100)
        print(analysis)
        print("=" * 100)
        print()
        print()
        
        # Save results
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "model": model,
            "sensor_statistics": stats,
            "analysis": analysis,
            "raw_sensor_data": sensor_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Analysis complete!")
        print(f"✓ Results saved to: {output_path}")
        
    finally:
        session.close()
        print("\n✓ Snowflake session closed")
        print("=" * 100)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze study concentration using multimodal sensor data')
    parser.add_argument('--data', type=str, default='fyp_test.json',
                        help='Path to sensor data JSON file')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet',
                        help='Model name to use (claude-3-5-sonnet, llama4-maverick, etc.)')
    parser.add_argument('--output', type=str, default='CCoT/output/concentration_analysis_results.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    process_concentration_analysis(
        data_path=args.data,
        model=args.model,
        output_path=args.output
    )
