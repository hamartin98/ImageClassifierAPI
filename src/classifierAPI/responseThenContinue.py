from rest_framework.response import Response


class ResponseThenContinue(Response):
    '''Special response to return to client, then call the given callback'''

    def __init__(self, data, thenCallback, **kwargs):
        '''Override init'''
        super().__init__(data, **kwargs)
        self.thenCallback = thenCallback

    def close(self):
        '''Override close to call the callback after close'''
        super().close()
        self.thenCallback()
