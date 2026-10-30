#pragma once

#include <cstdint>
#include <d3d12.h>

namespace remix {

  bool isGraphicsCommandListIid(REFIID iid);
  bool installCommandListTrace(ID3D12GraphicsCommandList* commandList);
  void installCommandListInterfaceTraces(IUnknown* commandList);

  // Root constant-buffer view bound at the given slot for the most recent draw.
  D3D12_GPU_VIRTUAL_ADDRESS currentGraphicsRootCbv(uint32_t index);

  // Module-relative return address of the most recent indexed draw.
  uint64_t currentDrawCallerRva();

}
