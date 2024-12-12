# Aidviser
An AI-powered medical chatbot designed to assist users with symptom analysis, disease information, and healthcare guidance using advanced machine learning.

This repository contains Medical Chat AI, an interactive AI-based medical assistant that leverages machine learning and natural language processing to provide users with accurate and helpful healthcare insights. It includes features like symptom analysis, disease detection, and chatbot functionality, making it an essential tool for healthcare support.

The application integrates advanced AI models with a user-friendly interface, enabling seamless communication and guidance for both healthcare professionals and patients.

# How to run this project

### 1. Create a virtual enviornment 
```bash
conda create -n aidviser python=3.8 -y
````

### 2. Activate the virtual environment
```bash
conda activate aidviser
```

### 3. Install all the Requirements 
```bash
pip install -r requirements.txt
```

### 4. Create .env file in the root directory
```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 5. Download the quantize model

```link
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```
Download the model named
```model
llama-2-7b-chat.ggmlv3.q4_0.bin
```
Place this downloaded model in a folder called model in the root directoy.

### 6. Run the program

Make sure you have selected the right interpreter which goes by the name of the aidviser.

```bash
python store_index.py
```

```bash
python app.py
```

# Open the application in the browser once the server is running
```link
http://localhost:8081/
```

# Technologies Used

- Python
- HTML
- Css
- JavaScript




