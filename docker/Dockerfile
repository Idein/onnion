# idein/onnion-arm32-env
FROM arm32v7/debian:11

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip python3-numpy python3-pytest \
 && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

CMD ["bash"]
