#pragma once

#include <d3d12.h>

#include <atomic>

namespace remix {

  void log(const char* format, ...);

  class D3D12CommandQueueProxy final : public ID3D12CommandQueue {
  public:
    D3D12CommandQueueProxy(ID3D12CommandQueue* queue, ID3D12Device* device);

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) final;
    ULONG STDMETHODCALLTYPE AddRef() final;
    ULONG STDMETHODCALLTYPE Release() final;
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* dataSize, void* data) final;
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT dataSize, const void* data) final;
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* data) final;
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) final;
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** device) final;

    void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource* resource, UINT numResourceRegions, const D3D12_TILED_RESOURCE_COORDINATE* resourceRegionStartCoordinates, const D3D12_TILE_REGION_SIZE* resourceRegionSizes, ID3D12Heap* heap, UINT numRanges, const D3D12_TILE_RANGE_FLAGS* rangeFlags, const UINT* heapRangeStartOffsets, const UINT* rangeTileCounts, D3D12_TILE_MAPPING_FLAGS flags) final;
    void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource* dstResource, const D3D12_TILED_RESOURCE_COORDINATE* dstRegionStartCoordinate, ID3D12Resource* srcResource, const D3D12_TILED_RESOURCE_COORDINATE* srcRegionStartCoordinate, const D3D12_TILE_REGION_SIZE* regionSize, D3D12_TILE_MAPPING_FLAGS flags) final;
    void STDMETHODCALLTYPE ExecuteCommandLists(UINT numCommandLists, ID3D12CommandList* const* commandLists) final;
    void STDMETHODCALLTYPE SetMarker(UINT metadata, const void* data, UINT size) final;
    void STDMETHODCALLTYPE BeginEvent(UINT metadata, const void* data, UINT size) final;
    void STDMETHODCALLTYPE EndEvent() final;
    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* fence, UINT64 value) final;
    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* fence, UINT64 value) final;
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* frequency) final;
    HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* gpuTimestamp, UINT64* cpuTimestamp) final;
    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() final;

  private:
    std::atomic<ULONG> m_refs = 1;
    ID3D12CommandQueue* m_queue;
    ID3D12Device* m_device;
  };

}
