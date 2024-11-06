
if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=1SDtfiGnfubqHmF3yUlAx9xVwt7krHuPk -O data.zip
fi

if [ ! -d data ]; then
    unzip data.zip
fi

if [ ! -d adapter_checkpoint ]; then
    gdown https://drive.google.com/uc?id=1XrA6szuQn_G2FoYhivu3knwX3j6VAMRa -O adapter_checkpoint.zip
fi

if [ ! -d adapter_checkpoint ]; then
    unzip adapter_checkpoint.zip
fi