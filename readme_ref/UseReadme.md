make sure environment is activated
make sure to export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export HF-TOKEN env

start server with example:
python server.py --hf-token "Reference env variable" --port 6543 --device-id xpu

Using Server:
Send request to transcribe an audio file
curl -X POST -F "file=@Intro session.m4a" http://192.168.68.186:6543/transcribe/ 
