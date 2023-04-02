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
docker run --name image-classifier \
           -p 8000:8000 \
           -v $pwd/data:/data \
           -t image-classifier:latest
```

```
docker run --name image-classifier -p 8000:8000 -v $pwd/data:/data -t image-classifier:latest
```


- Or without CUDA capabilities:
```
docker run --name image-classifier-nocuda \
           -p 8000:8000 \
           -v $pwd/data:/data \
           -t image-classifier-nocuda:latest
```

## Test connection from outside
- Run this command from terminal:
```
curl 127.0.0.1:8000/api/status
```