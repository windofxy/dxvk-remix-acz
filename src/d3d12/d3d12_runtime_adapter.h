#pragma once

#include <d3d12.h>

#include <cstdint>

#include "d3d12_native_backend.h"

namespace remix {

  // Native D3D12 frontend boundary. This adapter does not load or create any
  // D3D9 object and leaves the game's D3D12 command stream untouched.
  void initializeD3D12RuntimeAdapter();
  void attachD3D12Device(ID3D12Device* device);
  void attachD3D12CommandQueue(ID3D12CommandQueue* commandQueue, const D3D12_COMMAND_QUEUE_DESC& description);
  void attachD3D12CommandList(ID3D12GraphicsCommandList* commandList);
  void recordD3D12Pipeline(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* description, ID3D12PipelineState* pipelineState, HRESULT result);
  void recordD3D12Resource(
      const D3D12_HEAP_PROPERTIES* heapProperties,
      const D3D12_RESOURCE_DESC* description,
      ID3D12Resource* resource,
      HRESULT result);
  void recordD3D12DescriptorHeap(
      ID3D12DescriptorHeap* heap,
      const D3D12_DESCRIPTOR_HEAP_DESC& description,
      D3D12_CPU_DESCRIPTOR_HANDLE cpuStart,
      D3D12_GPU_DESCRIPTOR_HANDLE gpuStart,
      UINT increment);
  void recordD3D12View(
      const char* viewType,
      D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
      ID3D12Resource* resource,
      DXGI_FORMAT format,
      const void* viewDescription = nullptr,
      UINT viewDescriptionSize = 0);
  void recordD3D12Sampler(
      D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
      const D3D12_SAMPLER_DESC* description);
  void recordD3D12ConstantBufferView(
      D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
      const D3D12_CONSTANT_BUFFER_VIEW_DESC* description);
  void recordD3D12CopyDescriptors(
      UINT numDestDescriptorRanges,
      const D3D12_CPU_DESCRIPTOR_HANDLE* destDescriptorRangeStarts,
      const UINT* destDescriptorRangeSizes,
      UINT numSrcDescriptorRanges,
      const D3D12_CPU_DESCRIPTOR_HANDLE* srcDescriptorRangeStarts,
      const UINT* srcDescriptorRangeSizes,
      UINT descriptorIncrement);
  void recordD3D12CopyDescriptorsSimple(
      UINT numDescriptors,
      D3D12_CPU_DESCRIPTOR_HANDLE destDescriptorRangeStart,
      D3D12_CPU_DESCRIPTOR_HANDLE srcDescriptorRangeStart,
      UINT descriptorIncrement);
  void recordD3D12ResourceBarrier(UINT count, const D3D12_RESOURCE_BARRIER* barriers);
  void recordD3D12CopyTextureRegion(
      const D3D12_TEXTURE_COPY_LOCATION* destination,
      const D3D12_TEXTURE_COPY_LOCATION* source,
      UINT destinationX,
      UINT destinationY,
      UINT destinationZ,
      const D3D12_BOX* sourceBox);
  void recordD3D12GeometryCaptureDuration(uint64_t durationTicks);
  bool getD3D12TextureSnapshot(
      uint64_t resource,
      UINT subresource,
      D3D12NativeTextureSnapshot& snapshot);
  void recordD3D12Draw(
      ID3D12GraphicsCommandList* commandList,
      const D3D12_VERTEX_BUFFER_VIEW& vertexBuffer,
      const D3D12_INDEX_BUFFER_VIEW& indexBuffer,
      const D3D12_GPU_DESCRIPTOR_HANDLE* graphicsDescriptorTables,
      D3D12_PRIMITIVE_TOPOLOGY topology,
      ID3D12PipelineState* pipelineState,
      UINT indexCount,
      UINT instanceCount,
      UINT startIndexLocation,
      INT baseVertexLocation,
      UINT startInstanceLocation,
      D3D12_GPU_VIRTUAL_ADDRESS objectTransformCbv,
      UINT vertexCount = 0,
      const D3D12_GPU_VIRTUAL_ADDRESS* graphicsRootSrvs = nullptr);
  void recordD3D12Submit(ID3D12CommandQueue* commandQueue, UINT commandListCount, ID3D12CommandList* const* commandLists);

  // Resolves one descriptor of a GPU descriptor table and copies the start of
  // the buffer it references. Diagnostic aid for locating a game's per-frame
  // constant uploads; returns false when the descriptor is unknown or does not
  // reference readable memory.
  bool readBoundBufferFromDescriptorTable(
      uint64_t tableGpu,
      uint32_t descriptorIndex,
      size_t size,
      std::vector<uint8_t>& output,
      const char** kindOut,
      uint64_t* bufferGpuOut);
  void recordD3D12FenceSignal(ID3D12CommandQueue* commandQueue, ID3D12Fence* fence, UINT64 value, HRESULT result);
  void recordD3D12FenceWait(ID3D12CommandQueue* commandQueue, ID3D12Fence* fence, UINT64 value, HRESULT result);
  void recordD3D12PresentBegin(void* swapChain, UINT backBufferIndex, ID3D12Resource* backBuffer, const D3D12_RESOURCE_DESC* backBufferDescription);
  void recordD3D12Present(void* swapChain, UINT syncInterval, UINT flags, HRESULT result);

  // Called by the companion DXGI proxy through GetProcAddress. Keep this C
  // ABI stable so the two independently loaded proxy DLLs do not share C++ ABI.
  extern "C" void __cdecl remixD3D12RuntimePresent(void* swapChain, UINT syncInterval, UINT flags, HRESULT result);
  extern "C" void __cdecl remixD3D12RuntimePresentBegin(void* swapChain, UINT backBufferIndex, ID3D12Resource* backBuffer, const D3D12_RESOURCE_DESC* backBufferDescription);

  // Control surface for a game-side adapter. These are exported from d3d12.dll
  // and are the only supported way to drive the compatibility layer.
  extern "C" void __cdecl remixD3D12SetCamera(const float* viewProjection, const float* worldToView);
  extern "C" void __cdecl remixD3D12SetOutputEnabled(UINT enabled);
  extern "C" void __cdecl remixD3D12SetWorldSpacePolicy(UINT mode, float minExtent);

}
