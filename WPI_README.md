
# Updated Installation
```
conda create -n ground python=3.9 dill matplotlib plotly scipy scikit-learn
conda activate ground
pip install utm
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchdata -c pytorch  # CPU
conda install -c conda-forge pyproj
conda install -c conda-forge spot
```

From here you follow the same file download instructions as the original readme.

# Running
To run ground using gpt-5 set the OPENAI_API_KEY environmental variable to your api key.
If you want to run it with local models you first need to install [Ollama](https://ollama.com/download).
From here you need to install mxbai-embed-large:latest, your choice of vision capable llm, and your llm of choice.
The vision capable llm and llm of choice can be the same. By default, they are gemma:4b and deepseek-r1:12b
If you wish to have different models specify them with the OLLAMA_VISION_MODEL and OLLAMA_TEXT_MODEL environmental
variables respectfully. If you are using an ollama instance not on your local machine specify the ip in OLLAMA_HOST
environment variable. Now with all of that set up you are good to run.