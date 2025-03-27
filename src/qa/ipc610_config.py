"""
IPC-610 Defect Classes and Acceptance Criteria Configuration
Based on IPC-A-610 Revision G - Acceptability of Electronic Assemblies
"""

IPC610_DEFECT_CLASSES = {
    "defect_classes": [
        {
            "name": "SolderSlopeDefect",
            "description": "Solder paste slope defects",
            "class3_limits": {
                "max_slope_angle": 45  # degrees
            }
        },
        {
            "name": "SolderBridgingDefect",
            "description": "Solder bridging defects",
            "class3_limits": {
                "max_bridge_width": 0.5  # mm
            }
        }
    ]
} 