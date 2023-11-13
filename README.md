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

Download and save speech recognition and text-to-speech models.

```sh
python train.py
```

Start the application locally.

```sh
bentoml serve
```

Visit http://0.0.0.0:3000 for an OpenAPI Swagger page and http://0.0.0.0:3000/chatbot for a Gradio UI for the chatbot. Note that the microphone input functionality may not be functional on browsers like Google Chrome because the endpoint is not HTTPS. However, the microphone input will become functional
after deploying to BentoCloud.


Build application into a distributable Bento artifact.

```sh
bentoml build

Building BentoML service "voicegpt:vmjw2vucbodwkcvj" from build context "/Users/ssheng/github/BentoChain".
Packing model "speecht5_tts_processor:7pjfnkucbgjzycvj"
Packing model "speecht5_tts_vocoder:7suthpucbgjzycvj"
Packing model "whisper_processor:7s6wbnecbgjzycvj"
Packing model "whisper_model:7td75iucbgjzycvj"
Packing model "speecht5_tts_model:7pkfc3ecbgjzycvj"

██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

Successfully built Bento(tag="voicegpt:vmjw2vucbodwkcvj").

Possible next steps:

 * Containerize your Bento with `bentoml containerize`:
    $ bentoml containerize voicegpt:vmjw2vucbodwkcvj

 * Push to BentoCloud with `bentoml push`:
    $ bentoml push voicegpt:vmjw2vucbodwkcvj
```

# Production Deployment

BentoML provides a number of [deployment options](https://docs.bentoml.com/en/latest/concepts/deploy.html).
The easiest way to set up a production-ready endpoint of your text embedding service is via BentoCloud,
the serverless cloud platform built for BentoML, by the BentoML team.

Next steps:

1. Sign up for a BentoCloud account [here](https://www.bentoml.com/).
2. Get an API Token, see instructions [here](https://docs.bentoml.com/en/latest/bentocloud/getting-started/ship.html#acquiring-an-api-token).
3. Push your Bento to BentoCloud:
   
   ```sh
   bentoml push voicegpt:vmjw2vucbodwkcvj
   ```

4. Deploy via Web UI, see [Deploying on BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/getting-started/ship.html#deploying-your-bento)


And and push to BentoCloud.



