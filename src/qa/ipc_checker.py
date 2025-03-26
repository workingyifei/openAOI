from .metrics import DefectMetrics
import numpy as np

class IPC610Validator:
    def __init__(self, settings):
        self.settings = settings
        self.metrics = DefectMetrics()
        self.thresholds = self._load_thresholds()
        
    def _load_thresholds(self):
        """Load IPC-610 class 3 thresholds"""
        ipc_classes = self.settings.load_ipc_classes()
        return {defect['name']: defect['class3_limits'] 
                for defect in ipc_classes['defect_classes']}
    
    def validate(self, defects, point_cloud=None):
        """Validate defects against IPC-610 standards"""
        validated_defects = []
        
        for defect in defects:
            violations = self._check_violations(
                defect, 
                point_cloud
            )
            
            if violations:
                defect['violations'] = violations
                defect['ipc_status'] = 'FAIL'
                validated_defects.append(defect)
            else:
                defect['ipc_status'] = 'PASS'
                
        return validated_defects
    
    def _check_violations(self, defect, point_cloud):
        """Check IPC-610 violations for a specific defect"""
        violations = []
        defect_type = defect['class']
        thresholds = self.thresholds.get(defect_type, {})
        
        if defect_type == 'SolderBridging':
            metrics = self.metrics.compute_solder_bridge_metrics(
                defect,
                self.settings.PIXEL_TO_MM_RATIO
            )
            
            if metrics['max_distance'] > thresholds['max_distance']:
                violations.append({
                    'type': 'excessive_bridging',
                    'value': metrics['max_distance'],
                    'threshold': thresholds['max_distance'],
                    'unit': 'mm'
                })
                
        elif defect_type == 'Voiding':
            metrics = self.metrics.compute_void_metrics(
                defect,
                defect.get('mask')
            )
            
            if metrics.get('void_percentage', 0) > thresholds['max_area_percentage']:
                violations.append({
                    'type': 'excessive_voiding',
                    'value': metrics['void_percentage'],
                    'threshold': thresholds['max_area_percentage'],
                    'unit': 'percentage'
                })
                
        elif defect_type == 'Tombstoning':
            if point_cloud is not None:
                metrics = self.metrics.compute_tombstone_metrics(
                    defect,
                    point_cloud
                )
                
                if metrics['angle_degrees'] > thresholds['max_angle']:
                    violations.append({
                        'type': 'excessive_angle',
                        'value': metrics['angle_degrees'],
                        'threshold': thresholds['max_angle'],
                        'unit': 'degrees'
                    })
                    
        return violations 