#pragma once

#include <cstdint>

namespace remix {

  struct AczCameraProbe {
    uint64_t object = 0;
    uint64_t inner = 0;
    uint64_t camera = 0;
    uint32_t samples = 0;
    float block[32] = {};
    uint64_t transformArgs[4] = {};
    // Draw emitter bookkeeping: how many times its entry point ran, and the draw
    // context pointer it was handed. A zero count means the entry point was never
    // reached, which separates a missing hook from a hook that reads a null argument.
    uint64_t emitterSamples = 0;
    uint64_t emitterContext = 0;
  };

  // Installs the ACZ camera probe on the renderer entry point that runs every
  // frame, including flight.
  bool installAczCameraHook();

  // Copies the latest probe snapshot. Returns false before the first sample.
  bool readAczCameraProbe(AczCameraProbe& out);

  // Number of vertex batch calls seen so far. More than one call per frame means
  // several cameras are fed through the same entry point.
  uint64_t readAczTransformSamples();

  // Whether the draw emitter hook was actually patched in. A false value means the
  // entry point signature was not found, which is otherwise indistinguishable from an
  // entry point that simply never runs.
  bool aczEmitterHookInstalled();

  // Copies the most recent vertex batch camera arguments, newest first, and
  // returns how many are populated.
  uint32_t readAczTransformCameras(uint64_t out[4]);

}
