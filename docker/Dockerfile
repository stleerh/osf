FROM debian:trixie-slim

COPY oc /usr/local/bin/oc
RUN ln -s /usr/local/bin/oc /usr/local/bin/kubectl

# Keep container running or use entrypoint to execute command passed at runtime
ENTRYPOINT ["/bin/bash", "-c"]
