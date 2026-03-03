"""
Analyze student study concentration using multimodal sensor data with Chain of Thought reasoning
Processes sensor data (heart rate, noise level, steps) and provides personalized feedback
"""
import json
import os
from dotenv import load_dotenv
from snowflake.snowpark import Session
from datetime import datetime


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
    
    # Build Chain of Thought prompt
    prompt = f"""You are an expert study concentration analyst. Analyze student sensor data to assess concentration levels and provide helpful feedback.

**SENSOR DATA OVERVIEW:**
- Duration: {stats['num_readings']} minutes of study session
- Average Heart Rate: {stats['avg_heart_rate']:.1f} bpm
- Average Noise Level: {stats['avg_noise_level']:.1f} dB
- Average Steps per Minute: {stats['avg_steps']:.1f} steps

**DETAILED SENSOR READINGS:**{sensor_summary}

**NORMAL CONCENTRATION THRESHOLDS:**
1. Heart Rate:
   - Optimal focused state: 60-80 bpm (calm but alert)
   - Slightly elevated: 80-90 bpm (engaged, possibly stressed)
   - High: >90 bpm (stress, anxiety, or physical activity interfering)
   - Low: <60 bpm (possibly drowsy or very relaxed)

2. Noise Level:
   - Optimal quiet: 30-45 dB (library-like, minimal distractions)
   - Moderate: 45-60 dB (some background noise, manageable)
   - Disruptive: >60 dB (conversation level, likely distracting)

3. Movement (Steps per Minute):
   - Focused studying: 0-20 steps (minimal movement, seated)
   - Occasional breaks: 20-40 steps (healthy micro-breaks)
   - Frequent movement: 40-60 steps (restlessness, difficulty focusing)
   - High activity: >60 steps (not studying, walking around)

**TASK: Use Compositional Chain-of-Thought reasoning to analyze concentration level**

Follow this compositional reasoning process to systematically analyze each modality before synthesizing insights:

**PHASE 1: INDIVIDUAL MODALITY ANALYSIS**
Analyze each sensor type independently to understand its individual contribution:

1. **STEP 1 - Heart Rate Decomposition:**
   - Compare average heart rate against optimal thresholds
   - Identify any significant deviations (high/low) from normal ranges
   - Note temporal patterns: Are there spikes, drops, or sustained elevations?
   - Preliminary interpretation: What does this modality alone suggest about stress/alertness?

2. **STEP 2 - Noise Environment Decomposition:**
   - Compare average noise level against optimal thresholds
   - Assess environmental quality: library-quiet vs. disruptive
   - Note temporal patterns: Are there sudden noise spikes or sustained high levels?
   - Preliminary interpretation: What does this modality alone suggest about environmental distractions?

3. **STEP 3 - Movement Pattern Decomposition:**
   - Compare average steps against optimal thresholds
   - Assess movement type: focused stillness vs. restlessness vs. active breaks
   - Note temporal patterns: Consistent movement or sporadic bursts?
   - Preliminary interpretation: What does this modality alone suggest about physical engagement?

**PHASE 2: CROSS-MODAL COMPOSITION**
Synthesize insights across modalities to identify complex patterns:

4. **STEP 4 - Identify Cross-Modal Correlations:**
   - Temporal alignment: Do noise spikes coincide with heart rate increases?
   - Behavioral patterns: Does movement correlate with environmental or physiological changes?
   - Causal relationships: Does one modality appear to trigger changes in others?
   - Example: "High noise → increased heart rate → restless movement" suggests environmental stress

5. **STEP 5 - Compositional Concentration Assessment:**
   - Synthesize findings from Steps 1-4 into a holistic view
   - Determine overall concentration level: GOOD, MODERATE, or POOR
   - Identify the primary factor(s) affecting concentration
   - Explain how multiple modalities interact to support or undermine focus

**PHASE 3: ACTIONABLE SYNTHESIS**

6. **STEP 6 - Generate Personalized Recommendations:**
   - Based on compositional analysis, identify root causes (not just symptoms)
   - Provide specific recommendations targeting identified issues
   - Acknowledge positive patterns to reinforce good habits
   - Prioritize 2-3 most impactful interventions

**FORMAT YOUR RESPONSE AS:**

**Compositional Chain-of-Thought Analysis:**

*Phase 1 - Individual Modality Analysis:*
- Heart Rate: [Your analysis from Step 1]
- Noise Environment: [Your analysis from Step 2]
- Movement Pattern: [Your analysis from Step 3]

*Phase 2 - Cross-Modal Composition:*
- Correlations: [Your analysis from Step 4]
- Holistic Assessment: [Your synthesis from Step 5]

**Concentration Level:** [GOOD/MODERATE/POOR]

**Primary Contributing Factors:**
- [Factor 1 and how modalities interact]
- [Factor 2 and how modalities interact]
- [Factor 3 and how modalities interact]

**Personalized Feedback:**
[Helpful, encouraging feedback addressing the compositional insights with 2-3 specific actionable recommendations that target root causes identified through cross-modal analysis]

**IMPORTANT: RESPONSE FORMAT**
After completing your analysis, you MUST respond with ONLY a valid JSON object in this exact format:
{{
  "score": <integer from 1-10>,
  "reasoning": "<BRIEF and CONCISE summary of your analysis>"
}}

Where:
- score: 1-10 integer representing concentration level (1=very poor, 10=excellent)
- reasoning: A CONCISE 3-5 sentence summary covering:
  1) Key sensor observations (heart rate, noise, movement patterns)
  2) Main correlations between sensors
  3) Overall concentration assessment with primary factors
  4) Top 2-3 actionable recommendations
  
Example format: "Heart rate averaged X bpm with spike at TIME. Noise peaked at Y dB correlating with movement increase, indicating environmental disruption. Concentration is LEVEL due to FACTORS. Recommendations: 1) ACTION, 2) ACTION."

Keep it brief and actionable - do NOT include the full phase-by-phase analysis in the reasoning field.

Do not include any text outside the JSON object."""
    
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
            response_text = result[0]['RESPONSE']
            # Parse JSON response - handle escaped strings
            try:
                # First, try direct JSON parse
                response_json = json.loads(response_text)
                return {
                    'score': response_json.get('score'),
                    'reasoning': response_json.get('reasoning')
                }
            except json.JSONDecodeError:
                # The response might be an escaped JSON string - try to unescape it
                try:
                    # Remove outer quotes if present and decode escape sequences
                    if response_text.startswith('"') and response_text.endswith('"'):
                        response_text = response_text[1:-1]
                    # Decode common escape sequences
                    unescaped = response_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                    response_json = json.loads(unescaped)
                    return {
                        'score': response_json.get('score'),
                        'reasoning': response_json.get('reasoning')
                    }
                except (json.JSONDecodeError, ValueError, AttributeError):
                    # Final fallback - response is not valid JSON
                    return {
                        'score': None,
                        'reasoning': response_text,
                        'error': 'Response was not valid JSON'
                    }
        return {'score': None, 'reasoning': "[ERROR] No response from model"}
    except Exception as e:
        return {'score': None, 'reasoning': f"[ERROR] {str(e)}"}


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
        result = analyze_concentration(
            sensor_data=sensor_data,
            model=model,
            session=session
        )
        
        print("📋 CONCENTRATION ANALYSIS RESULTS:")
        print("=" * 100)
        
        # Display score
        if result.get('score') is not None:
            print(f"\n🎯 CONCENTRATION SCORE: {result['score']}/10")
            print()
        
        # Display reasoning
        print("💭 REASONING:")
        print("-" * 100)
        print(result.get('reasoning', 'No reasoning provided'))
        print("=" * 100)
        print()
        
        # Save results
        output_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": model,
            "sensor_statistics": stats,
            "concentration_score": result.get('score'),
            "reasoning": result.get('reasoning'),
            "raw_sensor_data": sensor_data
        }
        
        if 'error' in result:
            output_data['error'] = result['error']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
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
    parser.add_argument('--output', type=str, default='concentration_analysis_results.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    process_concentration_analysis(
        data_path=args.data,
        model=args.model,
        output_path=args.output
    )
