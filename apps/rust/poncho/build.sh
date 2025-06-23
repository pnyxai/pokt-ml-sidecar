DEFAULT_IMAGE_NAME="pokt_ml_poncho"

# go to root directory
cd ../../..

# Build sidecar
docker build . -f apps/rust/poncho/Dockerfile --progress=plain --tag $DEFAULT_IMAGE_NAME:dev
# Broadcast image name and tag
echo "$DEFAULT_IMAGE_NAME"