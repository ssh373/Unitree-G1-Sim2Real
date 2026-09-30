#!/usr/bin/env bash
# Fetch a prebuilt ONNX Runtime into thirdparty/, matching this machine's architecture.
#
#   ./scripts/get_onnxruntime.sh [version]      # default 1.22.0
#
# CMake picks up whatever this leaves in thirdparty/onnxruntime-linux-*. Version 1.14 is the
# minimum: the policies are exported at opset 18, and the loader uses GetInputNameAllocated,
# which replaced GetInputName in 1.14.
set -euo pipefail

VERSION="${1:-1.22.0}"
THIRDPARTY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/thirdparty"

case "$(uname -m)" in
  x86_64)  ARCH=x64 ;;
  aarch64) ARCH=aarch64 ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

NAME="onnxruntime-linux-${ARCH}-${VERSION}"
DEST="${THIRDPARTY}/${NAME}"

if [ -f "${DEST}/include/onnxruntime_cxx_api.h" ]; then
  echo "Already present: ${DEST}"
  exit 0
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${NAME}.tgz"
echo "Fetching ${URL}"
mkdir -p "${THIRDPARTY}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
curl -fSL "${URL}" -o "${TMP}/ort.tgz"
tar xzf "${TMP}/ort.tgz" -C "${THIRDPARTY}"

if [ ! -f "${DEST}/include/onnxruntime_cxx_api.h" ]; then
  echo "Unpacked, but ${DEST}/include/onnxruntime_cxx_api.h is missing -- unexpected archive layout" >&2
  exit 1
fi
echo "ONNX Runtime ${VERSION} (${ARCH}) ready at ${DEST}"
