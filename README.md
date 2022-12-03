# Case Study 
 

## Build docker image 

create docker image to use all steps in machine 
learning life cycyle

andela-case is the tag of the docker image 

```
docker bulid . -t andela-case
```

## Docker Storage Mount
- zipped data 
- artifacts
- data
- models
- experiment 
```
docker run -it -v $(pwd)/storage:/app/storage andela-case
```

## Docker Port Publish
- application 
```
docker run -it -p 8501:8501 andela-case 
```


## Run Order 

### 1- Unzip data to storage 

```
docker run -it -v $(pwd)/storage:/app/storage andela-case python unzip_data.py
```

### 2- Model Trainning 

```
docker run -it -v $(pwd)/storage:/app/storage andela-case python train.py
```

### 3- Serve Model  

```
docker run -it -p 8501:8501 -v $(pwd)/storage:/app/storage andela-case streamlit run application_app.py
```

[aplication](http://localhost:8501)
