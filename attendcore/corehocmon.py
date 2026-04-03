# attendcore/corehocmon.py


    # Trọng số cho từng sensor theo tháng tuổi (0-24 tháng)
SENSOR_WEIGHTS = {
        'vision': [
            0.00, 0.05, 0.10, 0.15, 0.20, 0.25,  # 0-5 tháng
            0.30, 0.35, 0.40, 0.45, 0.50, 0.55,  # 6-11 tháng
            0.60, 0.65, 0.70, 0.75, 0.80, 0.85,  # 12-17 tháng
            0.90, 0.95, 1.00, 1.00, 1.00, 1.00   # 18-24 tháng
        ],
        'hearing': [
            0.80, 0.85, 0.85, 0.80, 0.75, 0.70,  # 0-5 tháng
            0.65, 0.60, 0.55, 0.50, 0.45, 0.40,  # 6-11 tháng
            0.35, 0.30, 0.25, 0.20, 0.18, 0.15,  # 12-17 tháng
            0.12, 0.10, 0.08, 0.07, 0.06, 0.05   # 18-24 tháng
        ],
        'smell': [
            0.90, 0.85, 0.80, 0.75, 0.70, 0.65,  # 0-5 tháng
            0.60, 0.55, 0.50, 0.45, 0.40, 0.35,  # 6-11 tháng
            0.30, 0.25, 0.20, 0.18, 0.15, 0.12,  # 12-17 tháng
            0.10, 0.08, 0.06, 0.05, 0.04, 0.03   # 18-24 tháng
        ],
        'taste': [
            0.70, 0.75, 0.80, 0.85, 0.90, 0.95,  # 0-5 tháng
            1.00, 0.95, 0.90, 0.85, 0.80, 0.75,  # 6-11 tháng
            0.70, 0.65, 0.60, 0.55, 0.50, 0.45,  # 12-17 tháng
            0.40, 0.35, 0.30, 0.25, 0.20, 0.15   # 18-24 tháng
        ],
        'touch': [
            0.85, 0.90, 0.95, 1.00, 0.95, 0.90,  # 0-5 tháng
            0.85, 0.80, 0.75, 0.70, 0.65, 0.60,  # 6-11 tháng
            0.55, 0.50, 0.45, 0.40, 0.35, 0.30,  # 12-17 tháng
            0.25, 0.20, 0.18, 0.15, 0.12, 0.10   # 18-24 tháng
        ],
        'temperature': [
            0.40, 0.45, 0.50, 0.55, 0.60, 0.65,  # 0-5 tháng
            0.70, 0.75, 0.80, 0.85, 0.90, 0.95,  # 6-11 tháng
            1.00, 0.95, 0.90, 0.85, 0.80, 0.75,  # 12-17 tháng
            0.70, 0.65, 0.60, 0.55, 0.50, 0.45   # 18-24 tháng
        ]
    }
    
    # Ngưỡng an toàn theo tháng tuổi
SAFETY_THRESHOLDS = {
        'hearing': [
            {'min_freq': 20, 'max_freq': 500, 'max_db': 60},    # 0 tháng
            {'min_freq': 50, 'max_freq': 4000, 'max_db': 65},   # 1-3 tháng
            {'min_freq': 100, 'max_freq': 8000, 'max_db': 70},  # 4-6 tháng
            {'min_freq': 100, 'max_freq': 12000, 'max_db': 75}, # 7-12 tháng
            {'min_freq': 20, 'max_freq': 15000, 'max_db': 80},  # 13-24 tháng
        ],
        'vision': [
            {'min_clarity': 0.0, 'min_contrast': 0.0, 'max_flicker': 30},    # 0-1 tháng
            {'min_clarity': 0.2, 'min_contrast': 0.3, 'max_flicker': 50},    # 2-3 tháng
            {'min_clarity': 0.4, 'min_contrast': 0.5, 'max_flicker': 75},    # 4-6 tháng
            {'min_clarity': 0.6, 'min_contrast': 0.6, 'max_flicker': 100},   # 7-12 tháng
            {'min_clarity': 0.8, 'min_contrast': 0.7, 'max_flicker': 120},   # 13-24 tháng
        ],
        'smell': [
            {'max_concentration': 0.1},    # 0-3 tháng
            {'max_concentration': 0.3},    # 4-6 tháng
            {'max_concentration': 0.5},    # 7-12 tháng
            {'max_concentration': 1.0},    # 13-24 tháng
        ],
        'taste': [
            {'max_bitter': 10.0},   # 0-3 tháng
            {'max_bitter': 15.0},   # 4-6 tháng
            {'max_bitter': 20.0},   # 7-12 tháng
            {'max_bitter': 25.0},   # 13-24 tháng
        ],
        'temp': [
            {'min': 36.5, 'max': 37.5},   # 0-3 tháng
            {'min': 36.3, 'max': 37.5},   # 4-6 tháng
            {'min': 36.0, 'max': 37.8},   # 7-12 tháng
            {'min': 35.8, 'max': 38.0},   # 13-24 tháng
        ]
    }
    
    # Hợp chất ảnh hưởng tích cực/tiêu cực
SMELL_COMPOUNDS = {
        'positive': ['nonanal', 'limonene', 'linalool', 'vanillin', 'coumarin'],
        'negative': ['hexanoic_acid', 'skatole', 'isovaleric_acid', 'putrescine']
    }
    
TASTE_COMPOUNDS = {
    'positive': ['lactose', 'sucrose', 'glucose', 'fructose'],
    'negative': ['caffeine', 'quinine', 'denatonium'],
    'sour': ['citric_acid', 'acetic_acid', 'malic_acid']  # Thêm dòng này
}
    
    # Vùng nhạy cảm xúc giác
SENSITIVE_AREAS = ['face', 'hand', 'foot', 'cheek', 'stomach']
    

ATTENTION_HORMONE_MAP = {
    # Bổ sung hormone mapping cho trạng thái chú ý tổng hợp

    # Vision-based
    'bright_color_interest': {'dopamine': +0.8},
    'motion_tracking': {'dopamine': +0.6, 'adrenaline': +0.4},
    'overstimulated_by_flicker': {'cortisol': +0.9, 'adrenaline': +0.7},
    'face_focused': {'oxytocin': +0.85},
    'novel_object_interest': {'dopamine': +0.6},
    'pattern_preference': {'serotonin': +0.5},

    # Hearing-based
    'seeking_mother_voice': {'oxytocin': +0.9, 'dopamine': +0.7},
    'soothing_sound_focus': {'dopamine': +0.6, 'cortisol': -0.5},
    'startled_by_loud_noise': {'cortisol': +0.95, 'adrenaline': +0.8},
    'rhythmic_sound_engagement': {'serotonin': +0.75},
    'music_engagement': {'dopamine': +0.7, 'serotonin': +0.6},

    # Smell-based
    'seeking_maternal_scent': {'oxytocin': +0.8, 'dopamine': +0.6},
    'familiar_food_attraction': {'dopamine': +0.7, 'endorphins': +0.5},
    'odor_aversion': {'cortisol': +0.85},

    # Taste-based
    'sweet_taste_seeking': {'dopamine': +0.8, 'endorphins': +0.6},
    'bitter_taste_rejection': {'cortisol': +0.9},
    'taste_exploration': {'dopamine': +0.7},
    'sour_taste_response': {'cortisol': +0.6},

    # Touch-based
    'soft_contact_engagement': {'oxytocin': +0.75, 'serotonin': +0.65},
    'deep_pressure_seeking': {'serotonin': +0.8, 'dopamine': +0.5},
    'tactile_discomfort': {'cortisol': +0.9, 'adrenaline': +0.7},
    'tickle_reaction': {'endorphins': +0.5, 'dopamine': +0.4},

    # Temperature-based
    'warmth_seeking': {'oxytocin': +0.7, 'cortisol': -0.6},
    'heat_discomfort': {'cortisol': +0.85},
    'cold_stress': {'adrenaline': +0.8, 'cortisol': +0.75},
    'rapid_temp_change': {'adrenaline': +0.6},

    # Medical conditions
    'sleeping': {'melatonin': +0.9, 'cortisol': -0.8},
    'basic_needs_required': {'cortisol': +0.95},
    'separation_anxiety': {'cortisol': +0.9, 'adrenaline': +0.7},
    'comforted_by_caregiver': {'oxytocin': +1.0},
    # 'face_focused' and 'seeking_mother_voice' are already defined above

    # BỔ SUNG COGNITIVE CONFLICT
    'cognitive_conflict': {
        'GABA': +0.3,
        'glutamate': +0.4
    },
}

ATTENTION_STATUS_HORMONES = {
    "high_attend": {
        'dopamine_boost': +0.5,
        'serotonin_boost': +0.4,
        'cortisol_reduction': -0.6,
        'oxytocin_boost': +0.3,
        'cognitive_conflict_impact': +0.2
    },
    "moderate_attend": {
        'dopamine_boost': +0.3,
        'serotonin_boost': +0.2,
        'cortisol_reduction': -0.4
    },
    "neutral": {
        'serotonin_boost': +0.1,
        'cortisol_reduction': -0.2
    },
    "unattend": {
        'dopamine_penalty': -0.4,
        'cortisol_boost': +0.5
    },
    "strong_unattend": {
        'dopamine_penalty': -0.7,
        'cortisol_boost': +0.9,
        'adrenaline_boost': +0.6
    }
}