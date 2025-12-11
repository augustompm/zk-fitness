"""
constants.py

Project constants based on medical literature.

References:
    [1] WHO 2020 - Physical Activity Guidelines
    [12] ACSM Guidelines - Heart rate zones
"""

TANAKA_INTERCEPT = 208
TANAKA_SLOPE = 0.7

MODERATE_ZONE_LOW_PCT = 0.50
MODERATE_ZONE_HIGH_PCT = 0.70
VIGOROUS_ZONE_LOW_PCT = 0.70
VIGOROUS_ZONE_HIGH_PCT = 0.85

WEEKLY_GOAL_MODERATE_MINUTES = 150
WEEKLY_GOAL_VIGOROUS_MINUTES = 75
VIGOROUS_TO_MODERATE_RATIO = 2

MET_SEDENTARY_MAX = 1.5
MET_LIGHT_MAX = 3.0
MET_MODERATE_MAX = 6.0

INTENSITY_SEDENTARY = 0
INTENSITY_LIGHT = 1
INTENSITY_MODERATE = 2
INTENSITY_VIGOROUS = 3

INTENSITY_LABELS = {
    INTENSITY_SEDENTARY: "sedentary",
    INTENSITY_LIGHT: "light",
    INTENSITY_MODERATE: "moderate",
    INTENSITY_VIGOROUS: "vigorous"
}

MHEALTH_SAMPLING_RATE_HZ = 50

MHEALTH_ACTIVITY_TO_INTENSITY = {
    0: None,
    1: INTENSITY_SEDENTARY,
    2: INTENSITY_SEDENTARY,
    3: INTENSITY_SEDENTARY,
    4: INTENSITY_MODERATE,
    5: INTENSITY_VIGOROUS,
    6: INTENSITY_LIGHT,
    7: INTENSITY_LIGHT,
    8: INTENSITY_LIGHT,
    9: INTENSITY_MODERATE,
    10: INTENSITY_VIGOROUS,
    11: INTENSITY_VIGOROUS,
    12: INTENSITY_VIGOROUS
}

MHEALTH_SENSOR_COLUMNS = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ecg_lead1", "ecg_lead2",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "arm_acc_x", "arm_acc_y", "arm_acc_z",
    "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
    "arm_mag_x", "arm_mag_y", "arm_mag_z",
    "label"
]

DEFAULT_WINDOW_SIZE_SECONDS = 2.0
DEFAULT_WINDOW_SAMPLES = int(DEFAULT_WINDOW_SIZE_SECONDS * MHEALTH_SAMPLING_RATE_HZ)


def calculate_fcmax_tanaka(age):
    return TANAKA_INTERCEPT - TANAKA_SLOPE * age


def calculate_hr_zones(age):
    fcmax = calculate_fcmax_tanaka(age)
    return {
        "fcmax": fcmax,
        "moderate_low": fcmax * MODERATE_ZONE_LOW_PCT,
        "moderate_high": fcmax * MODERATE_ZONE_HIGH_PCT,
        "vigorous_low": fcmax * VIGOROUS_ZONE_LOW_PCT,
        "vigorous_high": fcmax * VIGOROUS_ZONE_HIGH_PCT
    }


def calculate_equivalent_minutes(moderate_minutes, vigorous_minutes):
    return moderate_minutes + (vigorous_minutes * VIGOROUS_TO_MODERATE_RATIO)


def check_weekly_goal(moderate_minutes, vigorous_minutes):
    equivalent = calculate_equivalent_minutes(moderate_minutes, vigorous_minutes)
    return equivalent >= WEEKLY_GOAL_MODERATE_MINUTES
