"""
IPC-610 Defect Classes and Acceptance Criteria Configuration
Based on IPC-A-610 Revision G - Acceptability of Electronic Assemblies
"""

IPC610_DEFECT_CLASSES = {
    "defect_classes": [
        # Solder Joint Defects
        {
            "name": "SolderBridging",
            "description": "Unintended connection between conductors",
            "class3_limits": {
                "max_distance": 0.1  # mm
            }
        },
        {
            "name": "InsufficientSolder",
            "description": "Incomplete or insufficient solder fill",
            "class3_limits": {
                "min_fill_percentage": 75  # %
            }
        },
        {
            "name": "ExcessSolder",
            "description": "Excessive solder amount",
            "class3_limits": {
                "max_height_ratio": 1.5  # ratio to nominal
            }
        },
        {
            "name": "Voiding",
            "description": "Voids or cavities in solder joint",
            "class3_limits": {
                "max_area_percentage": 25  # %
            }
        },
        
        # Component Placement Defects
        {
            "name": "Tombstoning",
            "description": "Component lifted on one end",
            "class3_limits": {
                "max_angle": 5  # degrees
            }
        },
        {
            "name": "ComponentShift",
            "description": "Lateral displacement of component",
            "class3_limits": {
                "max_offset": 0.5  # mm
            }
        },
        {
            "name": "ComponentRotation",
            "description": "Angular misalignment",
            "class3_limits": {
                "max_rotation": 3  # degrees
            }
        },
        
        # Surface Defects
        {
            "name": "Contamination",
            "description": "Foreign material or residue",
            "class3_limits": {
                "max_area": 0.25  # mm²
            }
        },
        {
            "name": "SolderBall",
            "description": "Isolated sphere of solder",
            "class3_limits": {
                "max_diameter": 0.2  # mm
            }
        },
        
        # PCB and Pad Defects
        {
            "name": "PadLift",
            "description": "Separation of pad from board",
            "class3_limits": {
                "max_lift_percentage": 0  # % (not allowed for class 3)
            }
        },
        {
            "name": "CopperExposure",
            "description": "Exposed copper on PCB",
            "class3_limits": {
                "max_area": 0.1  # mm²
            }
        },
        
        # Lead Defects
        {
            "name": "LeadBend",
            "description": "Improper lead forming",
            "class3_limits": {
                "max_bend_angle": 30  # degrees
            }
        },
        {
            "name": "LeadProtrusion",
            "description": "Lead extending beyond joint",
            "class3_limits": {
                "max_protrusion": 0.5  # mm
            }
        }
    ]
} 