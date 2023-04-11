from rest_framework import serializers
from classifier.classificationType import ClassificationType, ClassificationTypeUtils


def typeValidator(value: str) -> bool:
    fromString = ClassificationTypeUtils.fromString(value)

    if fromString == ClassificationType.NONE:
        return False

    return True


class ConfigSerializer(serializers.Serializer):
    modelPath = serializers.CharField(required=True)
    dataPath = serializers.CharField(required=False)
    imageWidth = serializers.IntegerField(
        required=False, default=62, min_value=1)
    imageHeight = serializers.IntegerField(
        required=False, default=62, min_value=1)
    testRatio = serializers.FloatField(
        required=False, default=0.5, min_value=0.0, max_value=1.0)
    saveModel = serializers.BooleanField(required=True)
    loadModel = serializers.BooleanField(required=True)
    dataLoaderWorkers = serializers.IntegerField(
        required=False, default=2, min_value=1, max_value=4)
    learningRate = serializers.FloatField(
        required=True, min_value=0.0, max_value=1.0)
    epochs = serializers.IntegerField(required=True, min_value=1)
    momentum = serializers.FloatField(
        required=True, min_value=0.0, max_value=1.0)
    batchSize = serializers.IntegerField(
        required=True, min_value=1)
    type = serializers.CharField(
        required=True, validators=[typeValidator])
