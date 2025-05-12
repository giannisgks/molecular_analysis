# 🧪 Machine Learning tool
This is an application/tool, served via a web interface(Streamlit) where you can use ML algorithms on your data 
## 🚀 Getting Started 

### 1. Build the Docker Image 
```docker build -t molecular-analysis:1.0 . ``` 

### 2. Verify the Image Was Created 
```docker images``` 

### 3. Run the Image/Create Container
```docker run --rm -p 8501:8501 molecular-analysis:1.0```