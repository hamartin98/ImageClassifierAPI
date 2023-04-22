from rest_framework.response import Response
from rest_framework.decorators import api_view

from classifier.activeTrainingInfo import ActiveTrainingInfo
from classifier.classificationMap import BaseClassification, ClassificationMap
from classifier.classificationType import ClassificationTypeUtils
from classifier.config.classifierConfig import ClassifierConfig
from classifier.multiModelClassifier import MultiModelClassifier
from classifier.teacher import Teacher
from classifier.utils.imageUtils import calculateMeanAndStdForImages
from .ConfigSerializer import ConfigSerializer
from .responseThenContinue import ResponseThenContinue


@api_view(['GET'])
def status(request):
    '''Get application status'''

    response = {'status': 'OK'}

    return Response(response)


@api_view(['POST'])
def classifyImage(request):
    '''Split and classify each part of the given image'''

    try:
        file = request.data['image']
        print(request.data)
        print(file)

        rows = 1
        if 'rows' in request.data:
            rows = int(request.data['rows'])

        cols = 1
        if 'cols' in request.data:
            cols = int(request.data['cols'])

        imageClassifier = MultiModelClassifier()
        result = imageClassifier.classifyWithMultiModels(file, rows, cols)

        response = {
            'message': 'Classification succesful',
            'rows': rows,
            'cols': cols,
            'result': result
        }

        return Response(response)
    except KeyError as exception:
        response = Response({'error': f'{exception}'})
        response.status_code = 400

        return response
    except Exception as exception:
        print(exception)
        response = Response({'error': f'Error happened: {exception}'})
        response.status_code = 400

        return response


@api_view(['POST'])
def singleClassTeach(request):
    '''Teach a single classification'''

    try:
        if not ActiveTrainingInfo.canStartNew():
            response = Response(
                {'message': f'Another Training is in progress, cannot start new one until its fininshed'}
            )
            response.status_code = 201

            return response

        configValidator = ConfigSerializer(data=request.data)
        if not configValidator.is_valid():
            response = Response(
                {'error': f'Validation errors happened: {configValidator.errors}'})
            response.status_code = 400
            return response

        classificationType = ClassificationTypeUtils.fromString(
            request.data['type'])
        classification: BaseClassification = ClassificationMap.getClassificationByType(ClassificationMap(),
                                                                                       classificationType)

        config = ClassifierConfig(None)
        config.setFromJson(request.data)
        classification.configureAndSetupNetwork(config)

        teacher = Teacher(classification, config)

        message = {
            'message': 'Training started, for more information call the trainingStatus endpoint'
        }

        return ResponseThenContinue(message, teacher.train)

    except Exception as exception:
        print(exception)
        response = Response({'error': f'Error happened: {exception}'})
        response.status_code = 400

        return response


@api_view(['GET'])
def getTrainingStatus(request):
    '''Get the status of the current training'''

    try:
        ActiveTrainingInfo.print()
        result = ActiveTrainingInfo.toJson()
        response = Response({'trainingStatus': result})

        return response
    except Exception as exception:
        print(exception)
        response = Response({'error': f'Error happened: {exception}'})
        response.status_code = 400

        return response


@api_view(['POST'])
def getDataSetMean(request):
    '''Get mean value of the given dataset'''

    try:
        if 'dataPath' not in request.data:
            raise Exception('dataPath attribute is required')
            
        dataPath = request.data['dataPath']
        
        meanValues = calculateMeanAndStdForImages(dataPath)

        return Response(meanValues)

    except Exception as exception:
        print(exception)
        response = Response({'error': f'Error happened: {exception}'})
        response.status_code = 400

        return response
