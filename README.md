## Creating Docker container
```bash
docker build -t demandmodeltrain:latest .
docker run \
  --gpus=all \
  --name dm_train \
  -p 7777:7777 \
  -p 6006:6006 \
  --volume ${PWD}:/data \
  -it demandmodeltrain:latest \
  bash
```