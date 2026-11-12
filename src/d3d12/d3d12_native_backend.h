#pragma once

#include <d3d12.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace remix {

  struct D3D12NativeInputElement {
    std::string semantic;
    UINT semanticIndex = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT slot = 0;
    UINT offset = 0;
  };

  enum class D3D12NativeDescriptorKind : uint8_t {
    Unknown,
    Cbv,
    Srv,
    Uav,
    Rtv,
    Dsv,
    Sampler,
  };

  struct D3D12NativeBoundDescriptor {
    UINT rootParameterIndex = 0;
    UINT tableOffset = 0;
    D3D12NativeDescriptorKind kind = D3D12NativeDescriptorKind::Unknown;
    uint64_t resource = 0;
    D3D12_RESOURCE_DESC resourceDescription {};
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    D3D12_GPU_VIRTUAL_ADDRESS bufferLocation = 0;
    UINT bufferSize = 0;
    std::array<uint8_t, 64> viewDescription {};
    UINT viewDescriptionSize = 0;
    UINT capturedSubresources = 0;
    uint64_t capturedBytes = 0;
  };

  struct D3D12NativeTextureSnapshot {
    UINT width = 0;
    UINT height = 0;
    UINT rowPitch = 0;
    UINT64 rowSize = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::vector<uint8_t> bytes;
  };

  struct D3D12NativePipelineInfo {
    uint64_t vertexShaderHash = 0;
    uint64_t pixelShaderHash = 0;
    uint64_t drawId = 0;
    uint64_t pipelineState = 0;
    uint64_t vertexBufferLocation = 0;
    uint64_t indexBufferLocation = 0;
    UINT vertexBufferSize = 0;
    UINT vertexBufferStride = 0;
    UINT indexBufferSize = 0;
    DXGI_FORMAT indexBufferFormat = DXGI_FORMAT_UNKNOWN;
    UINT indexCount = 0;
    UINT instanceCount = 0;
    UINT startIndexLocation = 0;
    INT baseVertexLocation = 0;
    UINT startInstanceLocation = 0;
    D3D12_GPU_VIRTUAL_ADDRESS objectTransformCbv = 0;
    uint8_t objectTransformSource = 0;
    std::array<uint64_t, 8> descriptorTables {};
    std::vector<D3D12NativeBoundDescriptor> boundDescriptors;
    std::vector<D3D12NativeInputElement> inputElements;
    bool blendEnabled = false;
    bool alphaTestEnabled = false;
    bool depthWriteEnabled = true;
    bool hasObjectToWorld = false;
    std::array<float, 16> objectToWorld {};
    // Non-indexed draws arrive through DrawInstanced; the backend synthesises
    // sequential indices so the same mesh path can submit them.
    bool implicitIndices = false;
    UINT vertexCount = 0;
  };

  // Creates the DXVK/Vulkan device used by the native D3D12 frontend.
  void initializeD3D12NativeBackend();

  // Captures the game's native D3D12 synchronization objects used to order
  // shared back-buffer access with the independent Vulkan queue.
  void attachD3D12NativeDevice(ID3D12Device* device);
  void attachD3D12NativeCommandQueue(ID3D12CommandQueue* commandQueue);

  // Prepares the native Remix scene immediately before the game presents its
  // current back buffer. The resource is borrowed for the duration of the
  // call and is not retained by the backend.
  void beginD3D12NativeFrame(
      void* swapChain,
      UINT backBufferIndex,
      ID3D12Resource* backBuffer,
      const D3D12_RESOURCE_DESC* backBufferDescription);

  // Advances the native backend frame counter after the game's Present call
  // has returned.
  void recordD3D12NativePresent();

  // Updates the Remix main camera from a row-vector view-projection matrix.
  // An explicit row-vector world-to-view matrix replaces the decomposed view
  // part when the engine keeps its view-projection slot fixed.
  void submitD3D12Camera(
      const float* viewProjection,
      const float* worldToView = nullptr);

  // Updates the Remix main camera from a row-vector world-to-view matrix and a
  // row-vector projection matrix. Both are used verbatim, so no decomposition
  // of a combined view-projection is involved.
  void submitD3D12CameraMatrices(
      const float* worldToView,
      const float* viewToProjection);

  // Camera the scene is actually rendered with, in the same row-vector form the
  // matrices were handed to Remix. Returns false until one has been submitted.
  bool readD3D12CameraMatrices(float worldToView[16], float eye[3]);

  // Enables or disables the Remix output path. Disabled output leaves the
  // native D3D12 frame on screen while capture keeps running.
  // Reports the bounds of the transforms submitted since the process started.
  bool readD3D12InstanceBounds(
      float& centerX, float& centerY, float& centerZ, float& radius);

  // Copies the inverse of the last submitted world-to-view matrix as a
  // row-vector matrix. The ACZ model path stores transforms in view space, so
  // this is what converts them back into world space.
  bool readD3D12CameraViewToWorld(std::array<float, 16>& out);

  void setD3D12OutputEnabled(bool enabled);

  // Selects how draws without a resolved object transform are placed.
  // 0: reject them, 1: place them when the decoded bounds exceed minExtent,
  // 2: treat every untransformed draw as world space.
  void setD3D12WorldSpacePolicy(uint32_t mode, float minExtent = 0.0f);

  // Converts one mapped D3D12 indexed draw into a Remix external mesh instance.
  // The original D3D12 command is still submitted by the game.
  void submitD3D12NativeGeometry(
      const D3D12_VERTEX_BUFFER_VIEW& vertexBuffer,
      const D3D12_INDEX_BUFFER_VIEW& indexBuffer,
      D3D12_PRIMITIVE_TOPOLOGY topology,
      UINT indexCount,
      UINT instanceCount,
      UINT startIndexLocation,
      INT baseVertexLocation,
      UINT startInstanceLocation,
      const D3D12NativePipelineInfo* pipelineInfo);

}
