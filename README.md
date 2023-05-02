# 🍱🔗 BentoChain - LangChain Deployment on BentoML 

Reference: Medium [post](https://medium.com/@ahmedbesbes/deploy-a-voice-based-chatbot-with-bentoml-langchain-and-gradio-7f25af3e45df)

----


BentoChain is a 🦜️🔗 [LangChain](https://github.com/hwchase17/langchain) deployment example using 🍱 [BentoML](https://github.com/bentoml/BentoML) inspired by [langchain-gradio-template](https://github.com/hwchase17/langchain-gradio-template). This example demonstrates how to create a voice chatbot using the OpenAI API, Transformers speech models, Gradio, and BentoML. The chatbot takes input from a microphone, which is then converted into text using a speech recognition model. 

The chatbot responds to the user's input with text, which can be played back to the user using a text-to-speech model.

## Demo

https://user-images.githubusercontent.com/6267065/235378103-54dd7c5b-16d1-4be7-b44a-fedde094c516.mp4

## Why deploy LangChain applications with BentoML?

🐳 Containerizes LangChain applications as standard OCI images.

🎱 Generates OpenAPI and gRPC endpoints automatically.

☁️ Deploys models as microservices running on the most optimal hardware and scaling independently.

## Instructions

Install Python dependencies.

```sh
poetry install
poetry shell
```

Create SSL certificate and key (this helps establish an HTTPS connexion that is needed to allow using the microphone on modern browers)

```sh
mkdir ssl
cd ssl
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes
```

Download and save speech recognition and text-to-speech models.

```sh
python train.py
```

Start the application locally.

```sh
bentoml serve service:svc --reload --ssl-certfile ssl/cert.pem --ssl-keyfile ssl/key.pem
```

Visit http://0.0.0.0:3000 for an OpenAPI Swagger page and http://0.0.0.0:3000/chatbot for a Gradio UI for the chatbot.


Build application into a distributable Bento artifact.

```sh
bentoml build
```

Containerize the application as an OCI image. This step requires Docker running.

```sh
bentoml containerize voicegpt:ahbt5xwxqsivkcvj
```

Run in Docker container.

```sh
docker run -it --rm -p 3333:3000 voicegpt:ahbt5xwxqsivkcvj serve --production
```

Push to yatai

```sh
bentoml push voicegpt:ahbt5xwxqsivkcvj
```
