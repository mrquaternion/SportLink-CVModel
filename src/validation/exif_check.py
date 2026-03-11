from enum import Enum
from haversine import haversine, Unit

exif_data_fields = ['cam_lat', 'cam_lon', 'date_taken']
infrastructure_data_fields = ['infra_id', 'sport', 'lat', 'lon']

def check(confidence, label, exif, infrastructure):
    if not all(field in exif for field in exif_data_fields):
        raise CheckException(CheckError.MISSING_EXIF_DATA)
    
    if not all(field in infrastructure for field in infrastructure_data_fields):
        raise CheckException(CheckError.MISSING_INFRASTRUCTURE_DATA)
    
    if further_than_100m(exif, infrastructure):
        raise CheckException(CheckError.TOO_FAR_FROM_INFRASTRUCTURE)
    
    if confidence < 0.8:
        pass
    
    if label != infrastructure['sport']:
        raise CheckException(
            CheckError.INVALID_PREDICTION,
            prediction=label,
            target=infrastructure['sport'],
            confidence=confidence,
        )
        

def further_than_100m(exif, infrastructure):
    tup1 = (float(exif['cam_lat']), float(exif['cam_lon']))
    tup2 = (float(infrastructure['lat']), float(infrastructure['lon']))

    dist = haversine(tup1, tup2, unit=Unit.METERS)

    return dist > 100.0


class CheckError(Enum):
    MISSING_EXIF_DATA = 'missing_exif_data'
    MISSING_INFRASTRUCTURE_DATA = 'missing_infrastructure_data'
    TOO_FAR_FROM_INFRASTRUCTURE = 'too_far_from_infrastructure'
    LOW_CONFIDENCE = 'low_confidence'
    INVALID_PREDICTION = 'invalid_prediction'


class CheckException(Exception):
    def __init__(self, error: CheckError, prediction=None, target=None, confidence=None):
        self.error = error
        self.prediction = prediction
        self.target = target
        self.confidence = confidence
        super().__init__(error.value)

    def __str__(self):
        base = self.error.value
        if self.prediction is not None or self.target is not None or self.confidence is not None:
            return (
                f"{base} "
                f"(prediction={self.prediction}, target={self.target}, confidence={self.confidence})"
            )
        return base