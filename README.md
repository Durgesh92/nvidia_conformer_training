# Setup Instructions

#### Install NGC
```bash
wget -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip && unzip -o ngccli_linux.zip && chmod u+x ngc

cd ngc-cli/

echo "export PATH=\"\$PATH:$(pwd)\"" >> ~/.bash_profile && source ~/.bash_profile
```

#### Setup access to nvidia containers
- Sign up on [https://catalog.ngc.nvidia.com/](https://catalog.ngc.nvidia.com/) and create an API key
- Run following command and follow the instructions, it will ask to enter API key
```bash
ngc config set
```
- Run docker login and set user name as `$oauthtoken` and API key
```bash
docker login nvcr.io
Username: $oauthtoken
Password: API KEY
```

#### Install CUDA
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run

sudo bash cuda_11.7.1_515.65.01_linux.run
```

#### Install CUDA Docker
```bash 
sudo docker pull nvidia/cuda:11.7.0-base-ubuntu20.04
```

#### Install nvidia-docker2
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker
```
- Check if everything is installed properly 
```bash
docker run --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi
```

#### Download Riva Quickstart and Install nvidia containers
```bash
ngc registry resource download-version "nvidia/riva/riva_quickstart:2.7.0"
```
```bash
cd riva_quickstart_v2.7.0

sudo bash riva_init.sh
```

#### Download Model
- Download English conformer model
```bash
ngc registry model download-version "nvidia/tao/speechtotext_en_us_conformer:deployable_v4.0"
```
- Download Hindi conformer model
```bash
ngc registry model download-version "nvidia/tao/speechtotext_hi_in_conformer:deployable_2.0"
```

#### Install KenLM
```bash
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz

cd kenlm/

mkdir -p build && cd build && cmake .. && make -j 4 && sudo make install
```

# LM Adaptation
- Build ARPA LM 
```bash
lmplz -o 4 < corpus.txt > lm.arpa
```
- Build vocab 
```bash
cat corpus.txt | tr " " "\n" | sort | uniq > vocab.txt
```

- move `lm.arpa` and `vocab.txt` in downloaded model folder
```bash
mv lm.arpa /home/ubuntu22/speechtotext_en_us_conformer_vdeployable_v4.0/

mv vocab.txt /home/ubuntu22/speechtotext_en_us_conformer_vdeployable_v4.0/
```

- mount models into riva servicemaker container. (make sure you replace `/home/ubuntu22/speechtotext_en_us_conformer_vdeployable_v4.0` with correct path on your system)
```bash
sudo docker run --gpus all -it --rm -v /home/ubuntu22/speechtotext_en_us_conformer_vdeployable_v4.0:/servicemaker-dev -v /home/ubuntu22/speechtotext_en_us_conformer_vdeployable_v4.0/deploy:/data --entrypoint="/bin/bash" nvcr.io/nvidia/riva/riva-speech:2.7.0-servicemaker
```
- Build model
```bash
riva-build speech_recognition \
   /servicemaker-dev/Conformer-CTC-L-en-US-ASR-set-4p0.rmir:tlt_encode \
   /servicemaker-dev/Conformer-CTC-L-en-US-ASR-set-4p0.riva:tlt_encode \
   --name=conformer-en-US-asr-streaming \
   --featurizer.use_utterance_norm_params=False \
   --featurizer.precalc_norm_time_steps=0 \
   --featurizer.precalc_norm_params=False \
   --ms_per_timestep=40 \
   --endpointing.start_history=200 \
   --nn.fp16_needs_obey_precision_pass \
   --endpointing.residue_blanks_at_start=-2 \
   --chunk_size=0.16 \
   --left_padding_size=1.92 \
   --right_padding_size=1.92 \
   --decoder_type=flashlight \
   --flashlight_decoder.asr_model_delay=-1 \
   --decoding_language_model_arpa=/servicemaker-dev/lm.arpa \
   --decoding_vocab=/servicemaker-dev/vocab.txt \
   --flashlight_decoder.lm_weight=0.8 \
   --flashlight_decoder.word_insertion_score=1.0 \
   --flashlight_decoder.beam_size=32 \
   --flashlight_decoder.beam_threshold=20. \
   --flashlight_decoder.num_tokenization=1 \
   --language_code=en-US
```

```bash
riva-deploy /servicemaker-dev/Conformer-CTC-L-en-US-ASR-set-4p0.rmir:tlt_encode /data/models
```
- Deploy model (make sure you replace `/home/ubuntu22/speechtotext_en_us_conformer_vdeployable_v4.0` with correct path on your system). This will start serving model on `50053` port
```bash
sudo docker run -d --init --rm  --gpus '"'device=0'"' -p 50053:50053 -v /home/ubuntu22/speechtotext_en_us_conformer_vdeployable_v4.0/deploy:/data --ulimit memlock=-1 --ulimit stack=67108864 --name riva-speech -p 8000 -p 8001 -p 8002 nvcr.io/nvidia/riva/riva-speech:2.7.0 start-riva --riva-uri=0.0.0.0:50053 --asr_service=true --tts_service=false --nlp_service=false
```

- Deploy file decoding API. This will start serving model on `7841` port
```bash
pip install riva_api-2.4.0-py3-none-any.whl
pip install flask grpc 

mkdir -p static/

python api.py
```
- To test file decoging mode
```bash
curl -F "file=@test.wav" "0.0.0.0:7841/transcribe"
```

# Training conformer

#### Data Preparation
- Assuming `/home/ubuntu22/Documents/asr_dataset/mix_IPA/` as your training directory create `data`, `specs` and `results` folder
```bash
mkdir -p /home/ubuntu22/Documents/asr_dataset/mix_IPA/data
mkdir -p /home/ubuntu22/Documents/asr_dataset/mix_IPA/specs
mkdir -p /home/ubuntu22/Documents/asr_dataset/mix_IPA/results
```
- Put all audio files in data/wav/ folder and create a training menifest file as follows. Prepare `train_manifest.json` and `test_manifest.json` and put it under `/home/ubuntu22/Documents/asr_dataset/mix_IPA/data` folder. Below is format of these menifest files.
```bash
{"audio_filepath": "/data/wav/vyabber_05_04_2019#Male#QOXjAibjkc.wav", "duration": 2.478125, "text": "GUD MORNING ME A2D1IT1Y BA2T1 KAR RAHA2 HU RELIGER HELTH INSHORANS SE"}
{"audio_filepath": "/data/wav/vyabber_05_04_2019#Male#HnrZypKiBJ.wav", "duration": 1.823, "text": "KIT1ANE FEMILI MEMBAR KE LIE POLASI PLA2N KAR RAHE A2P"}
{"audio_filepath": "/data/wav/vyabber_05_04_2019#Male#YE3w9EuIxl.wav", "duration": 2.896375, "text": "YA2 USAKA2 KUCHH OPAREI2SHAN HOGA2 NA2 CHA2R SA2L BA2D1 VO OPAREI2SHAN KAVAR HOGA2 NA2 BAS"}
{"audio_filepath": "/data/wav/vyabber_05_04_2019#Male#mJc7G29nvC.wav", "duration": 2.140375, "text": "SAR CHA2R SA2L KE BA2D1 OPAREI2SHAN KAVAR HOGA2"}
```
- Export environment variables
```bash
export HOST_DATA_DIR=/home/ubuntu22/Documents/asr_dataset/mix_IPA/data
export HOST_SPECS_DIR=/home/ubuntu22/Documents/asr_dataset/mix_IPA/specs
export HOST_RESULTS_DIR=/home/ubuntu22/Documents/asr_dataset/mix_IPA/results
```
- Create training config. (run this code as a python script)
```python
import json
import os
mounts_file = os.path.expanduser("~/.tao_mounts.json")
tlt_configs = {
   "Mounts":[
       {
           "source": os.environ["HOST_DATA_DIR"],
           "destination": "/data"
       },
       {
           "source": os.environ["HOST_SPECS_DIR"],
           "destination": "/specs"
       },
       {
           "source": os.environ["HOST_RESULTS_DIR"],
           "destination": "/results"
       },
       {
           "source": os.path.expanduser("~/.cache"),
           "destination": "/root/.cache"
       }
   ],
   "DockerOptions": {
        "shm_size": "16G",
        "ulimits": {
            "memlock": -1,
            "stack": 67108864
         }
   }
}
# Writing the mounts file.
with open(mounts_file, "w") as mfile:
    json.dump(tlt_configs, mfile, indent=4)
```
- copy `/home/ubuntu22/.tao_mounts.json` to `/root/.tlt_mounts.json`
```bash
sudo su
sudo cp /home/ubuntu22/.tao_mounts.json /root/.tlt_mounts.json
```
- Download training specs
```bash
sudo tao speech_to_text_conformer download_specs -o /specs -r /results
```
- open `/home/ubuntu22/Documents/asr_dataset/mix_IPA/specs` and set sample rate to `8000` at all places next to `sample_rate` tag. (this is optional step if your data has sampling rate of 8000)

- Start training. (Make sure you run this in background, use `byobu` for ease)
```bash
sudo tao speech_to_text_conformer train \
     -e /specs/train_conformer_bpe_medium.yaml \
     -g 1 \
     -k tlt_encode \
     -r /results/conformer/train \
     training_ds.manifest_filepath=/data/train_manifest.json \
     validation_ds.manifest_filepath=/data/test_manifest.json \
     trainer.max_epochs=7 \
     training_ds.num_workers=4 \
     validation_ds.num_workers=4 \
     model.tokenizer.dir=/data/indiantts/tokenizer_spe_unigram_v28
```
