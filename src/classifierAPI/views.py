from rest_framework.response import Response
from rest_framework.decorators import api_view
from classifier.multiModelClassifier import MultiModelClassifier
from classifier.teacher import Teacher
from classifier.classificationMap import BaseClassification, ClassificationMap
from classifier.classificationType import ClassificationTypeUtils
from .serializers import ConfigSerializer
from classifier.classifierConfig import ClassifierConfig
from .responseThenContinue import ResponseThenContinue


@api_view(['GET'])
def status(request):
    response = {'status': 'OK'}
    return Response(response)


@api_view(['POST'])
def classifyImage(request):
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
    except KeyError:
        response = Response({'error': 'Request has no resource file atteched'})
        response.status_code = 400
        return response
    except Exception as e:
        print(e)
        response = Response({'error': f'Error happened: {e}'})
        response.status_code = 400
        return response


@api_view(['POST'])
def singleClassTeach(request):
    try:
        print(request.data)

        configValidator = ConfigSerializer(data=request.data)
        if not configValidator.is_valid():
            response = Response(
                {'error': f'Validation errors happened: {configValidator.errors}'})
            response.status_code = 400
            return response

        classificationType = ClassificationTypeUtils.fromString(request.data['type'])
        classification: BaseClassification = ClassificationMap.getClassificationByType(ClassificationMap(),
            classificationType)
        
        config = ClassifierConfig(None)
        config.setFromJson(request.data)
        classification.configureAndSetupNetwork(config)
        
        teacher = Teacher(classification, config)
        # TODO: determine by type what to do
        
        message = {
            'message': 'Teaching started, for more information call the teachingStatus endpoint'
        }
        
        return ResponseThenContinue(message, teacher.trainAndTest)
    
    except Exception as e:
        print(e)
        response = Response({'error': f'Error happened: {e}'})
        response.status_code = 400
        return response

# TODO: Create endpoint to get active training status info
