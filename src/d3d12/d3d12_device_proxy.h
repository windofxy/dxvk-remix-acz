#pragma once

#include <d3d12.h>

#include <atomic>

namespace remix {

  void log(const char* format, ...);
  const char* formatIid(REFIID iid);

  class D3D12DeviceProxy final : public ID3D12Device2 {
  public:
    explicit D3D12DeviceProxy(ID3D12Device* device);

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) final;
    ULONG STDMETHODCALLTYPE AddRef() final;
    ULONG STDMETHODCALLTYPE Release() final;

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* dataSize, void* data) final;
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT dataSize, const void* data) final;
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* data) final;
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) final;

    UINT STDMETHODCALLTYPE GetNodeCount() final;
    HRESULT STDMETHODCALLTYPE CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid, void** commandQueue) final;
    HRESULT STDMETHODCALLTYPE CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** commandAllocator) final;
    HRESULT STDMETHODCALLTYPE CreateGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc, REFIID riid, void** pipelineState) final;
    HRESULT STDMETHODCALLTYPE CreateComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID riid, void** pipelineState) final;
    HRESULT STDMETHODCALLTYPE CreateCommandList(UINT nodeMask, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* commandAllocator, ID3D12PipelineState* initialState, REFIID riid, void** commandList) final;
    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(D3D12_FEATURE feature, void* featureSupportData, UINT featureSupportDataSize) final;
    HRESULT STDMETHODCALLTYPE CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC* descriptorHeapDesc, REFIID riid, void** heap) final;
    UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapType) final;
    HRESULT STDMETHODCALLTYPE CreateRootSignature(UINT nodeMask, const void* blobWithRootSignature, SIZE_T blobLengthInBytes, REFIID riid, void** rootSignature) final;
    void STDMETHODCALLTYPE CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) final;
    void STDMETHODCALLTYPE CreateShaderResourceView(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) final;
    void STDMETHODCALLTYPE CreateUnorderedAccessView(ID3D12Resource* resource, ID3D12Resource* counterResource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) final;
    void STDMETHODCALLTYPE CreateRenderTargetView(ID3D12Resource* resource, const D3D12_RENDER_TARGET_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) final;
    void STDMETHODCALLTYPE CreateDepthStencilView(ID3D12Resource* resource, const D3D12_DEPTH_STENCIL_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) final;
    void STDMETHODCALLTYPE CreateSampler(const D3D12_SAMPLER_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) final;
    void STDMETHODCALLTYPE CopyDescriptors(UINT numDestDescriptorRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* destDescriptorRangeStarts, const UINT* destDescriptorRangeSizes, UINT numSrcDescriptorRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* srcDescriptorRangeStarts, const UINT* srcDescriptorRangeSizes, D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapsType) final;
    void STDMETHODCALLTYPE CopyDescriptorsSimple(UINT numDescriptors, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptorRangeStart, D3D12_CPU_DESCRIPTOR_HANDLE srcDescriptorRangeStart, D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapsType) final;
    D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo(UINT visibleMask, UINT numResourceDescs, const D3D12_RESOURCE_DESC* resourceDescs) final;
    D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE GetCustomHeapProperties(UINT nodeMask, D3D12_HEAP_TYPE heapType) final;
    HRESULT STDMETHODCALLTYPE CreateCommittedResource(const D3D12_HEAP_PROPERTIES* heapProperties, D3D12_HEAP_FLAGS heapFlags, const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initialResourceState, const D3D12_CLEAR_VALUE* optimizedClearValue, REFIID riidResource, void** resource) final;
    HRESULT STDMETHODCALLTYPE CreateHeap(const D3D12_HEAP_DESC* desc, REFIID riid, void** heap) final;
    HRESULT STDMETHODCALLTYPE CreatePlacedResource(ID3D12Heap* heap, UINT64 heapOffset, const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE* optimizedClearValue, REFIID riid, void** resource) final;
    HRESULT STDMETHODCALLTYPE CreateReservedResource(const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE* optimizedClearValue, REFIID riid, void** resource) final;
    HRESULT STDMETHODCALLTYPE CreateSharedHandle(ID3D12DeviceChild* object, const SECURITY_ATTRIBUTES* attributes, DWORD access, LPCWSTR name, HANDLE* handle) final;
    HRESULT STDMETHODCALLTYPE OpenSharedHandle(HANDLE ntHandle, REFIID riid, void** object) final;
    HRESULT STDMETHODCALLTYPE OpenSharedHandleByName(LPCWSTR name, DWORD access, HANDLE* ntHandle) final;
    HRESULT STDMETHODCALLTYPE MakeResident(UINT numObjects, ID3D12Pageable* const* objects) final;
    HRESULT STDMETHODCALLTYPE Evict(UINT numObjects, ID3D12Pageable* const* objects) final;
    HRESULT STDMETHODCALLTYPE CreateFence(UINT64 initialValue, D3D12_FENCE_FLAGS flags, REFIID riid, void** fence) final;
    HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() final;
    void STDMETHODCALLTYPE GetCopyableFootprints(const D3D12_RESOURCE_DESC* resourceDesc, UINT firstSubresource, UINT numSubresources, UINT64 baseOffset, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts, UINT* numRows, UINT64* rowSizeInBytes, UINT64* totalBytes) final;
    HRESULT STDMETHODCALLTYPE CreateQueryHeap(const D3D12_QUERY_HEAP_DESC* desc, REFIID riid, void** heap) final;
    HRESULT STDMETHODCALLTYPE SetStablePowerState(BOOL enable) final;
    HRESULT STDMETHODCALLTYPE CreateCommandSignature(const D3D12_COMMAND_SIGNATURE_DESC* desc, ID3D12RootSignature* rootSignature, REFIID riid, void** commandSignature) final;
    void STDMETHODCALLTYPE GetResourceTiling(ID3D12Resource* tiledResource, UINT* numTilesForEntireResource, D3D12_PACKED_MIP_INFO* packedMipDesc, D3D12_TILE_SHAPE* standardTileShapeForNonPackedMips, UINT* numSubresourceTilings, UINT firstSubresourceTilingToGet, D3D12_SUBRESOURCE_TILING* subresourceTilingsForNonPackedMips) final;
    LUID STDMETHODCALLTYPE GetAdapterLuid() final;

    HRESULT STDMETHODCALLTYPE CreatePipelineLibrary(const void* libraryBlob, SIZE_T blobLength, REFIID riid, void** pipelineLibrary) final;
    HRESULT STDMETHODCALLTYPE SetEventOnMultipleFenceCompletion(ID3D12Fence* const* fences, const UINT64* fenceValues, UINT numFences, D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE event) final;
    HRESULT STDMETHODCALLTYPE SetResidencyPriority(UINT numObjects, ID3D12Pageable* const* objects, const D3D12_RESIDENCY_PRIORITY* priorities) final;
    HRESULT STDMETHODCALLTYPE CreatePipelineState(const D3D12_PIPELINE_STATE_STREAM_DESC* desc, REFIID riid, void** pipelineState) final;

  private:
    std::atomic<ULONG> m_refs = 1;
    ID3D12Device* m_device;
    ID3D12Device2* m_device2;
  };

}
