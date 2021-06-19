## Creating Docker container
```bash
docker build -t demandmodeltrain:latest .
docker run \
  --gpus=all \
  --name liu_2019 \
  -p 7777:7777 \
  -p 6006:6006 \
  --volume ${PWD}:/data \
  -it demandmodeltrain:latest \
  bash
```