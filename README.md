docker build -t lipidbot .
docker run -d \
  -v C:/Users/yqzn9/Downloads/hf_home/:/hf_home \
  -p 7120:7120 \
  lipidbot

1. ASk for config file
2. we need to run code to loal sentence transformer. 
