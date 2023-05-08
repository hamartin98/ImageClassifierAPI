# ImageClassifierAPI
## Summary
- PyTorch based satellite image classifier API
- It uses PyTorch for deep learning
- It uses Django to create the API to communicate with the program
- You can use it with CUDA support to use GPU for computing (if available)

## Build docker image
- From main dir run
- With CUDA capabilities:
```
docker build -f docker/Dockerfile . -t image-classifier:latest
```

```
docker build -f docker/Dockerfile . -t image-classifier:latest --no-cache
```

- Without CUDA capabilities:
```
docker build -f docker/Dockerfile.no_cuda . -t image-classifier-nocuda:latest
```

## Run docker image
- From main dir run:

```
docker run --name image-classifier --gpus all -p 8000:8000 -v $pwd/data:/data -t image-classifier:latest
```


- Or without CUDA capabilities:
```
docker run --name image-classifier-nocuda -p 8000:8000 -v $pwd/data:/data -t image-classifier-nocuda:latest
```

## Test connection from outside
- Run this command from terminal:
```
curl 127.0.0.1:8000/api/status
```

## Use endpoints
- In the file ImageClassifier.postman_collection you can find the endpoints with sample request
- To use this, install Postman and import this collection
- For the classification endpoint a sample response can be found in the classification_sample_response.json file

### Classification endpoint
- The endpoint expects an image
You can add the optional 'rows' and 'cols' parameters, to specificy the image splitting dimensions, when tese are used, the image is splitted according to these parameters before classification
If these parameters are not provided, the whole image is used
- The endpoint returns a json object consisting of an array with the dimension of the given number of rows and columns, with each item consisting of the 3 class labels