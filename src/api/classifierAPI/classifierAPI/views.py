from rest_framework.response import Response
from rest_framework.decorators import api_view

@api_view(['GET'])
def status(request):
    response = {'status': 'OK'}
    return Response(response)

@api_view(['POST'])
def classifyImage(request):
    try:
        file = request.data['image']
        # TODO: Do something with the image
        return Response({'message': 'Congratulations you sent an image'})
    except KeyError:
        return Response({'error': 'Request has no resource file atteched'})