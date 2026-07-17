#!/usr/bin/env bash
#
# Deploy the Fabric test network and the data-sharing chaincode.
#
# Prerequisites:
#   - Hyperledger Fabric v2.2 samples, binaries and Docker images installed
#     (see https://hyperledger-fabric.readthedocs.io/en/release-2.2/install.html)
#   - The FABRIC_SAMPLES environment variable pointing at the fabric-samples
#     directory that contains test-network.
#
# This script brings up a two-org network with a CouchDB state database,
# creates a channel, and deploys the chaincode in this repository.
set -euo pipefail

: "${FABRIC_SAMPLES:?Set FABRIC_SAMPLES to your fabric-samples directory}"

CHANNEL_NAME="mychannel"
CC_NAME="data_sharing_cc"
CC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../chaincode" && pwd)"
CC_LANG="golang"

cd "${FABRIC_SAMPLES}/test-network"

# Tear down any previous network, then start with CouchDB.
./network.sh down
./network.sh up createChannel -c "${CHANNEL_NAME}" -s couchdb

# Package and deploy the chaincode.
./network.sh deployCC \
  -ccn "${CC_NAME}" \
  -ccp "${CC_PATH}" \
  -ccl "${CC_LANG}" \
  -c "${CHANNEL_NAME}"

echo "Network up and chaincode '${CC_NAME}' deployed on channel '${CHANNEL_NAME}'."
