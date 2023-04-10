from rest_framework.response import Response
from rest_framework.decorators import api_view
from classifier.multiModelClassifier import MultiModelClassifier


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
