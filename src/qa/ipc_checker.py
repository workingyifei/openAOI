from .metrics import DefectMetrics
import numpy as np
import os

class IPC610Validator:
    def __init__(self, settings=None):
        self.settings = settings
        self.metrics = DefectMetrics()
        self.thresholds = self._load_thresholds()
        
    def _load_thresholds(self):
        """Load IPC-610 class 3 thresholds"""
        # Load directly from the config file
        config_path = os.path.join(os.path.dirname(__file__), 'ipc610_config.py')
        namespace = {}
        with open(config_path, 'r') as f:
            exec(f.read(), namespace)
        ipc_classes = namespace.get('IPC610_DEFECT_CLASSES', {})
            
        return {defect['name']: defect['class3_limits'] 
                for defect in ipc_classes['defect_classes']}
    
    def validate(self, defects):
        """Validate defects against IPC-610 standards"""
        validated_defects = []
        
        for defect in defects:
            violations = self._check_violations(defect)
            defect['violations'] = violations
            defect['ipc_status'] = 'FAIL' if violations else 'PASS'
            validated_defects.append(defect)
                
        return validated_defects
    
    def _check_violations(self, defect):
        """Check IPC-610 violations for a specific defect"""
        violations = []
        defect_type = defect['class']
        thresholds = self.thresholds.get(defect_type, {})
        
        if defect_type == 'Bad_qiaojiao':
            metrics = self.metrics.compute_bridge_metrics(
                defect,
                0.1  # Default pixel to mm ratio
            )
            
            if metrics['bridge_width'] > thresholds.get('max_bridge_width', 0.5):
                violations.append({
                    'type': 'excessive_bridging',
                    'value': metrics['bridge_width'],
                    'threshold': thresholds.get('max_bridge_width', 0.5),
                    'unit': 'mm'
                })
                
        elif defect_type == 'Bad_podu':
            metrics = self.metrics.compute_slope_metrics(
                defect,
                defect.get('mask')
            )
            
            if metrics.get('slope_angle', 0) > thresholds.get('max_slope_angle', 45):
                violations.append({
                    'type': 'excessive_slope',
                    'value': metrics['slope_angle'],
                    'threshold': thresholds.get('max_slope_angle', 45),
                    'unit': 'degrees'
                })
                    
        elif defect_type == 'SolderBridgingDefect':
            # Additional logic for SolderBridgingDefect
            pass
        elif defect_type == 'SolderSlopeDefect':
            # Additional logic for SolderSlopeDefect
            pass
                
        return violations 