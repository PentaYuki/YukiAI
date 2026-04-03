# hormone_system.py - Enhanced Version
import numpy as np
import time
import json
from typing import Dict, List, Optional
from datetime import datetime
import math

class HormoneSystem:
    def __init__(self, personality_template: str = "highly_sensitive", config_path: Optional[str] = None):
        self.nt_names = ['NE', 'D', 'S', 'CORT', 'OXT', 'GABA', 'Glu', 'VAS']
        
        # Load configuration
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Set personality template
        self.set_personality_template(personality_template)
        
        # Initialize current values at rest
        self.current_values = {nt: self.rest[nt] for nt in self.nt_names}
        
        # Active stimuli tracking with enhanced stimulus modulation
        self.active_stimuli = {}
        self.stimulus_modifiers = {nt: 1.0 for nt in self.nt_names}  # For Serotonin-NE interaction
        
        # History for plasticity and adaptation
        self.history = []
        self.stress_accumulation = 0.0
        self.reward_accumulation = 0.0
        self.sleep_loss_accumulation = 0.0
        
        # Circadian rhythm tracking
        self.last_update_time = time.time()
        self.circadian_adjustments = {nt: 0.0 for nt in self.nt_names}
        
        print(f"🔄 Enhanced Hormone system initialized with {personality_template} personality")

    def _get_default_config(self) -> Dict:
        return {
            "templates": {
                "highly_sensitive": {
                    "rest": {"NE": 0.55, "D": 0.40, "S": 0.30, "Glu": 0.60, "GABA": 0.45, "CORT": 0.18, "OXT": 0.50, "VAS": 0.40},
                    "tau": {"NE": 15, "D": 30, "S": 120, "Glu": 20, "GABA": 90, "CORT": 60, "OXT": 45, "VAS": 60},
                    "sigma": 0.02
                },
                "resilient": {
                    "rest": {"NE": 0.40, "D": 0.70, "S": 0.65, "Glu": 0.45, "GABA": 0.60, "CORT": 0.25, "OXT": 0.60, "VAS": 0.50},
                    "tau": {"NE": 20, "D": 30, "S": 120, "Glu": 25, "GABA": 90, "CORT": 60, "OXT": 45, "VAS": 60},
                    "sigma": 0.015
                },
                "introvert": {
                    "rest": {"NE": 0.35, "D": 0.35, "S": 0.60, "Glu": 0.75, "GABA": 0.65, "CORT": 0.20, "OXT": 0.45, "VAS": 0.40},
                    "tau": {"NE": 25, "D": 45, "S": 120, "Glu": 25, "GABA": 90, "CORT": 60, "OXT": 50, "VAS": 60},
                    "sigma": 0.01
                },
                "attached": {
                    "rest": {"NE": 0.45, "D": 0.50, "S": 0.60, "Glu": 0.50, "GABA": 0.55, "CORT": 0.20, "OXT": 0.75, "VAS": 0.65},
                    "tau": {"NE": 25, "D": 35, "S": 120, "Glu": 30, "GABA": 80, "CORT": 70, "OXT": 60, "VAS": 60},
                    "sigma": 0.01
                },
                "vulnerable": {
                    "rest": {"NE": 0.65, "D": 0.30, "S": 0.20, "Glu": 0.70, "GABA": 0.35, "CORT": 0.35, "OXT": 0.40, "VAS": 0.45},
                    "tau": {"NE": 20, "D": 40, "S": 150, "Glu": 25, "GABA": 100, "CORT": 90, "OXT": 55, "VAS": 65},
                    "sigma": 0.03
                }
            },
            "stim_coeffs": {
                "ping": {"NE": 0.12, "D": 0.05, "S": -0.01, "CORT": 0.08, "OXT": 0.0, "GABA": 0.0, "Glu": 0.02, "VAS": 0.0},
                "reward": {"NE": 0.10, "D": 0.25, "S": 0.02, "CORT": -0.03, "OXT": 0.08, "GABA": 0.00, "Glu": 0.0, "VAS": 0.05},
                "social_msg": {"NE": 0.04, "D": 0.08, "S": 0.02, "CORT": -0.02, "OXT": 0.12, "GABA": 0.00, "Glu": 0.0, "VAS": 0.08},
                "focus_flow": {"NE": -0.05, "D": 0.30, "S": 0.03, "CORT": -0.05, "OXT": 0.02, "GABA": 0.06, "Glu": -0.03, "VAS": 0.02},
                "multitask": {"NE": 0.20, "D": -0.05, "S": -0.03, "CORT": 0.15, "OXT": -0.02, "GABA": -0.04, "Glu": 0.10, "VAS": -0.02},
                "blue_light": {"NE": 0.15, "D": -0.10, "S": -0.12, "CORT": 0.20, "OXT": -0.05, "GABA": -0.10, "Glu": 0.08, "VAS": -0.03},
                "caregiver_presence": {"NE": -0.08, "D": 0.12, "S": 0.05, "CORT": -0.20, "OXT": 0.30, "GABA": 0.10, "Glu": -0.05, "VAS": 0.15},
                "failure": {"NE": 0.18, "D": -0.15, "S": -0.05, "CORT": 0.25, "OXT": -0.08, "GABA": -0.06, "Glu": 0.12, "VAS": -0.05}
            },
            "modifiers": {
                "oxytocin_on_cortisol_alpha": 0.35,
                "serotonin_on_ne_beta": 0.6,
                "gaba_on_dopamine_gamma": 0.25,
                "serotonin_stimulus_modulation": True  # New: Enable Serotonin-based stimulus modulation
            },
            "circadian": {
                "cortisol_peak_hour": 8.0,    # 8 AM
                "cortisol_trough_hour": 20.0,  # 8 PM  
                "serotonin_peak_hour": 14.0,   # 2 PM
                "serotonin_trough_hour": 2.0,  # 2 AM
                "amplitude": 0.15              # Maximum adjustment from rest
            }
        }

    def set_personality_template(self, template_name: str):
        """Set personality template and initialize parameters"""
        if template_name not in self.config["templates"]:
            raise ValueError(f"Unknown personality template: {template_name}")
        
        template = self.config["templates"][template_name]
        self.rest = template["rest"].copy()  # Make a copy to avoid modifying original
        self.base_rest = template["rest"].copy()  # Store base rest values for circadian adjustments
        self.tau = template["tau"]
        self.sigma = template["sigma"]
        self.personality = template_name

    def _update_circadian_rhythm(self, current_hour: float):
        """Update rest values based on circadian rhythm"""
        circ = self.config["circadian"]
        amplitude = circ["amplitude"]
        
        # Cortisol: high in morning, low in evening
        cortisol_phase = self._calculate_circadian_phase(current_hour, circ["cortisol_peak_hour"], circ["cortisol_trough_hour"])
        self.rest['CORT'] = self.base_rest['CORT'] + amplitude * cortisol_phase
        self.circadian_adjustments['CORT'] = amplitude * cortisol_phase
        
        # Serotonin: peaks in afternoon, trough at night
        serotonin_phase = self._calculate_circadian_phase(current_hour, circ["serotonin_peak_hour"], circ["serotonin_trough_hour"])
        self.rest['S'] = self.base_rest['S'] + amplitude * serotonin_phase
        self.circadian_adjustments['S'] = amplitude * serotonin_phase

    def _calculate_circadian_phase(self, current_hour: float, peak_hour: float, trough_hour: float) -> float:
        """Calculate circadian phase adjustment using cosine function"""
        # Normalize to 24-hour cycle
        hour_diff = (current_hour - peak_hour) % 24
        if hour_diff > 12:
            hour_diff = 24 - hour_diff
        
        # Cosine wave: 1 at peak, -1 at trough
        phase = math.cos((hour_diff / 12) * math.pi)
        return phase

    def _update_serotonin_ne_modulation(self):
        """Enhanced Serotonin-NE interaction: Serotonin modulates NE stimulus response"""
        if not self.config["modifiers"].get("serotonin_stimulus_modulation", True):
            return
            
        S = self.current_values['S']
        beta = self.config["modifiers"]["serotonin_on_ne_beta"]
        
        # Serotonin modulates how strongly NE responds to stressful stimuli
        # High serotonin = less reactive to stress; Low serotonin = more reactive
        reactivity_factor = 1.0 - beta * (S - 0.5)
        
        # Apply modulation to NE stimulus coefficients for stressful events
        stressful_stimuli = ['ping', 'multitask', 'failure', 'blue_light']
        for stim_type in stressful_stimuli:
            if stim_type in self.active_stimuli:
                # Reduce NE accumulation from stressful stimuli based on serotonin
                self.stimulus_modifiers['NE'] = max(0.3, reactivity_factor)

    def add_stimulus(self, stimulus_type: str, intensity: float = 1.0, duration: float = 1.0):
        """Add a stimulus with given intensity and duration (minutes)"""
        self.active_stimuli[stimulus_type] = {
            'intensity': intensity,
            'duration': duration,
            'start_time': time.time()
        }
        
        # Apply Serotonin-NE modulation to new stimulus
        self._update_serotonin_ne_modulation()
        
        # Track for plasticity
        if stimulus_type in ['multitask', 'failure', 'blue_light']:
            self.stress_accumulation += intensity
        elif stimulus_type in ['reward', 'social_msg', 'caregiver_presence']:
            self.reward_accumulation += intensity
        elif stimulus_type == 'blue_light':
            self.sleep_loss_accumulation += intensity

    def update(self, dt: float = 1.0):
        """Update hormone levels for time step dt (minutes)"""
        # Update circadian rhythm based on current time
        current_time = datetime.now()
        current_hour = current_time.hour + current_time.minute / 60.0
        self._update_circadian_rhythm(current_hour)
        
        # Update Serotonin-NE modulation
        self._update_serotonin_ne_modulation()
        
        # Apply stimuli impulses with modulation
        for stimulus_type, stimulus_info in list(self.active_stimuli.items()):
            intensity = stimulus_info['intensity']
            elapsed = (time.time() - stimulus_info['start_time']) / 60  # Convert to minutes
            
            if stimulus_type in self.config["stim_coeffs"]:
                coeffs = self.config["stim_coeffs"][stimulus_type]
                for nt, k in coeffs.items():
                    if nt in self.current_values:
                        # Apply stimulus modulation (for NE in stressful situations)
                        modulated_k = k * self.stimulus_modifiers.get(nt, 1.0)
                        impulse = modulated_k * intensity * (dt / stimulus_info['duration'])
                        self.current_values[nt] += impulse
            
            # Remove expired stimulus
            if elapsed >= stimulus_info['duration']:
                del self.active_stimuli[stimulus_type]

        # Decay toward rest (which now includes circadian adjustments)
        for nt in self.nt_names:
            if nt in self.tau and nt in self.rest:
                decay_rate = (self.current_values[nt] - self.rest[nt]) / self.tau[nt]
                self.current_values[nt] -= decay_rate * dt

        # Apply inter-substance modifiers
        self._apply_modifiers()

        # Add noise and clamp
        for nt in self.nt_names:
            noise = np.random.normal(0, self.sigma)
            self.current_values[nt] = np.clip(self.current_values[nt] + noise, 0.0, 1.0)

        # Apply long-term plasticity
        self._apply_plasticity(dt)

        # Reset stimulus modifiers for next update
        self.stimulus_modifiers = {nt: 1.0 for nt in self.nt_names}

        # Save to history
        self.history.append({**self.current_values, 'timestamp': time.time()})
        self.last_update_time = time.time()

    def _apply_modifiers(self):
        """Apply interactions between different neurotransmitters"""
        mod = self.config["modifiers"]
        
        # Oxytocin reduces cortisol response
        if 'CORT' in self.current_values and 'OXT' in self.current_values:
            self.current_values['CORT'] *= (1 - mod["oxytocin_on_cortisol_alpha"] * self.current_values['OXT'])
        
        # GABA inhibits dopamine release under stress (enhanced)
        if 'D' in self.current_values and 'GABA' in self.current_values and 'CORT' in self.current_values:
            if self.current_values['CORT'] > 0.6:  # High stress condition
                stress_level = (self.current_values['CORT'] - 0.6) / 0.4  # Normalize to 0-1
                inhibition = mod["gaba_on_dopamine_gamma"] * self.current_values['GABA'] * stress_level
                self.current_values['D'] *= (1 - inhibition)

    def _apply_plasticity(self, dt: float):
        """Apply long-term adaptations based on accumulated experiences"""
        # Stress adaptation (over 24-72 hour windows)
        stress_daily_increase = self.stress_accumulation * 0.0001 * dt
        if stress_daily_increase > 0:
            self.rest['CORT'] = min(0.8, self.rest['CORT'] + stress_daily_increase)
            self.rest['S'] = max(0.1, self.rest['S'] - stress_daily_increase * 0.5)
            # Thêm clamp cho các giá trị khác
            self.rest['GABA'] = max(0.1, min(0.9, self.rest['GABA'] - stress_daily_increase * 0.3))
        
        # Reward learning - reinforce successful behaviors
        if self.current_values['D'] > 0.7:  # Dopamine spike
            reward_learning_rate = 0.01 * dt
            # Increase baseline dopamine for successful behaviors
            self.base_rest['D'] = min(0.9, self.base_rest['D'] + reward_learning_rate * 0.1)
        
        # Sleep loss adaptation
        if self.sleep_loss_accumulation > 0:
            sleep_effect = self.sleep_loss_accumulation * 0.00005 * dt
            self.base_rest['CORT'] = min(0.8, self.base_rest['CORT'] + sleep_effect)
            self.base_rest['S'] = max(0.1, self.base_rest['S'] - sleep_effect)
            self.base_rest['NE'] = min(0.9, self.base_rest['NE'] + sleep_effect * 0.5)
        
        # Caregiver healing effect
        if self.reward_accumulation > 0:
            healing_rate = 0.00002 * dt
            self.base_rest['OXT'] = min(0.9, self.base_rest['OXT'] + healing_rate)
            self.base_rest['CORT'] = max(0.1, self.base_rest['CORT'] - healing_rate)
            self.base_rest['VAS'] = min(0.9, self.base_rest['VAS'] + healing_rate * 0.5)

    def get_current_state(self) -> Dict[str, float]:
        """Get current hormone state"""
        return self.current_values.copy()

    def get_motivation_level(self) -> float:
        """Enhanced motivation calculation using all 8 hormones"""
        hormones = self.current_values
        
        # Base formula from original system
        base_motivation = (0.5 * hormones.get('D', 0.5) + 
                        0.2 * (1.0 - hormones.get('CORT', 0.5)) + 
                        0.2 * hormones.get('OXT', 0.5) + 
                        0.1 * hormones.get('S', 0.5) - 
                        0.1 * hormones.get('NE', 0.5))
        
        # Enhanced with all 8 hormones
        enhanced_motivation = base_motivation + (
            0.1 * hormones.get('GABA', 0.5) - 
            0.05 * hormones.get('Glu', 0.5) + 
            0.05 * hormones.get('VAS', 0.5)
        )
        
        return np.clip(enhanced_motivation, 0.0, 1.0)

    def get_emotional_state(self) -> str:
        """Enhanced emotional state detection with more detailed categories"""
        D = self.current_values.get('D', 0.5)
        CORT = self.current_values.get('CORT', 0.5)
        OXT = self.current_values.get('OXT', 0.5)
        S = self.current_values.get('S', 0.5)
        NE = self.current_values.get('NE', 0.5)
        GABA = self.current_values.get('GABA', 0.5)
        
        # Enhanced emotional state detection
        if CORT > 0.7 and S < 0.4:
            return "anxious_stressed"
        elif CORT > 0.6 and NE > 0.7:
            return "frustrated_agitated"
        elif D > 0.7 and GABA > 0.6 and CORT < 0.4:
            return "focused_flow"
        elif OXT > 0.7 and CORT < 0.3:
            return "secure_attached"
        elif D > 0.7 and CORT < 0.3:
            return "excited_curious"
        elif OXT > 0.7 and S > 0.6:
            return "content_affectionate"
        elif CORT > 0.6 and D > 0.6:
            return "agitated_alert"
        elif D < 0.3 and S < 0.4:
            return "lethargic_depressed"
        elif GABA > 0.7 and CORT < 0.3:
            return "calm_serene"
        else:
            return "neutral_calm"

    def get_circadian_info(self) -> Dict:
        """Get information about current circadian rhythm adjustments"""
        current_time = datetime.now()
        current_hour = current_time.hour + current_time.minute / 60.0
        
        return {
            'current_hour': current_hour,
            'circadian_adjustments': self.circadian_adjustments.copy(),
            'base_rest_values': self.base_rest.copy(),
            'adjusted_rest_values': self.rest.copy()
        }

    def get_detailed_report(self) -> Dict:
        """Get comprehensive system report"""
        return {
            'personality': self.personality,
            'emotional_state': self.get_emotional_state(),
            'motivation_level': self.get_motivation_level(),
            'current_hormones': self.get_current_state(),
            'circadian_info': self.get_circadian_info(),
            'active_stimuli_count': len(self.active_stimuli),
            'stress_accumulation': self.stress_accumulation,
            'reward_accumulation': self.reward_accumulation,
            'stimulus_modifiers': self.stimulus_modifiers.copy()
        }

    def save_state(self, filepath: str):
        """Save current hormone system state to file"""
        state = {
            'current_values': self.current_values,
            'base_rest_values': self.base_rest,
            'personality': self.personality,
            'stress_accumulation': self.stress_accumulation,
            'reward_accumulation': self.reward_accumulation,
            'sleep_loss_accumulation': self.sleep_loss_accumulation,
            'history_length': len(self.history),
            'timestamp': time.time()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load hormone system state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.current_values = state['current_values']
        self.base_rest = state['base_rest_values']
        self.rest = self.base_rest.copy()  # Will be adjusted by circadian rhythm
        self.personality = state['personality']
        self.stress_accumulation = state.get('stress_accumulation', 0.0)
        self.reward_accumulation = state.get('reward_accumulation', 0.0)
        self.sleep_loss_accumulation = state.get('sleep_loss_accumulation', 0.0)

# Demo and testing functions
def test_enhanced_system():
    """Test the enhanced hormone system with new features"""
    print("🧪 Testing Enhanced Hormone System...")
    
    # Test different personality templates
    templates = ["highly_sensitive", "resilient", "introvert", "attached", "vulnerable"]
    
    for template in templates:
        print(f"\n--- Testing {template} personality ---")
        hs = HormoneSystem(template)
        
        # Add some stimuli
        hs.add_stimulus('ping', intensity=0.8, duration=1.0)
        hs.add_stimulus('focus_flow', intensity=0.9, duration=5.0)
        
        # Update system
        for i in range(3):
            hs.update(dt=2.0)
            report = hs.get_detailed_report()
            print(f"  Step {i+1}: {report['emotional_state']} | Motivation: {report['motivation_level']:.2f}")
        
        # Show circadian info
        circ_info = hs.get_circadian_info()
        print(f"  Circadian CORT adjustment: {circ_info['circadian_adjustments']['CORT']:.3f}")

if __name__ == "__main__":
    test_enhanced_system()