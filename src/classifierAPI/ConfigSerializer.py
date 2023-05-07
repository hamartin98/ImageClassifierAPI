from rest_framework import serializers

from classifier.classificationType import ClassificationType, ClassificationTypeUtils


def typeValidator(value: str) -> bool:
    '''Validate classification type'''
    fromString = ClassificationTypeUtils.fromString(value)

    if fromString == ClassificationType.NONE:
        return False

    return True


class ConfigSerializer(serializers.Serializer):
    '''Model validator class for classifer config'''
    modelPath = serializers.CharField(required=True)
    dataPath = serializers.CharField(required=False)
    imageWidth = serializers.IntegerField(
        required=False, default=62, min_value=1)
    imageHeight = serializers.IntegerField(
        required=False, default=62, min_value=1)
    trainRatio = serializers.FloatField(
        required=False, default=0.7, min_value=0.0, max_value=1.0)
    testRatio = serializers.FloatField(
        required=False, default=0.2, min_value=0.0, max_value=1.0)
    valRatio = serializers.FloatField(
        required=False, default=0.1, min_value=0.0, max_value=1.0)
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
    mean = serializers.ListField(
        required=False, child=serializers.FloatField(min_value=0.0, max_value=1.0), min_length=3, max_length=3)
    std = serializers.ListField(
        required=False, child=serializers.FloatField(min_value=0.0, max_value=1.0), min_length=3, max_length=3)
    augmentDataSet = serializers.BooleanField(required=False)
    balanceDataSet = serializers.BooleanField(required=False)
    useResNet = serializers.BooleanField(required=False)

    def validate(self, data):
        '''Add multi field custom validation'''
        trainRatio = data['trainRatio']
        testRatio = data['testRatio']
        valRatio = data['valRatio']

        sum = 0.0

        if trainRatio is not None:
            sum += trainRatio

        if testRatio is not None:
            sum += testRatio

        if valRatio is not None:
            sum += valRatio

        if round(sum) < 0.995:
            raise serializers.ValidationError(
                'The sum of train, test and validation ratios must be 1')

        return data
