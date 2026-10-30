#include "d3d12_command_queue_proxy.h"
#include "d3d12_runtime_adapter.h"

namespace remix {

  D3D12CommandQueueProxy::D3D12CommandQueueProxy(ID3D12CommandQueue* queue, ID3D12Device* device)
  : m_queue(queue),
    m_device(device) {
    m_device->AddRef();
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject)
      return E_POINTER;

    *ppvObject = nullptr;
    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(ID3D12Object)
     || riid == __uuidof(ID3D12DeviceChild)
     || riid == __uuidof(ID3D12Pageable)
     || riid == __uuidof(ID3D12CommandQueue)) {
      *ppvObject = static_cast<ID3D12CommandQueue*>(this);
      AddRef();
      return S_OK;
    }

    return m_queue->QueryInterface(riid, ppvObject);
  }

  ULONG STDMETHODCALLTYPE D3D12CommandQueueProxy::AddRef() {
    return ++m_refs;
  }

  ULONG STDMETHODCALLTYPE D3D12CommandQueueProxy::Release() {
    ULONG refs = --m_refs;
    if (!refs) {
      m_device->Release();
      m_queue->Release();
      delete this;
    }
    return refs;
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::GetPrivateData(REFGUID guid, UINT* dataSize, void* data) {
    return m_queue->GetPrivateData(guid, dataSize, data);
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::SetPrivateData(REFGUID guid, UINT dataSize, const void* data) {
    return m_queue->SetPrivateData(guid, dataSize, data);
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::SetPrivateDataInterface(REFGUID guid, const IUnknown* data) {
    return m_queue->SetPrivateDataInterface(guid, data);
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::SetName(LPCWSTR name) {
    return m_queue->SetName(name);
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::GetDevice(REFIID riid, void** device) {
    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(ID3D12Object)
     || riid == __uuidof(ID3D12Device)
     || riid == __uuidof(ID3D12Device1)
     || riid == __uuidof(ID3D12Device2))
      return m_device->QueryInterface(riid, device);

    return m_queue->GetDevice(riid, device);
  }

  void STDMETHODCALLTYPE D3D12CommandQueueProxy::UpdateTileMappings(ID3D12Resource* resource, UINT numResourceRegions, const D3D12_TILED_RESOURCE_COORDINATE* resourceRegionStartCoordinates, const D3D12_TILE_REGION_SIZE* resourceRegionSizes, ID3D12Heap* heap, UINT numRanges, const D3D12_TILE_RANGE_FLAGS* rangeFlags, const UINT* heapRangeStartOffsets, const UINT* rangeTileCounts, D3D12_TILE_MAPPING_FLAGS flags) {
    m_queue->UpdateTileMappings(resource, numResourceRegions, resourceRegionStartCoordinates, resourceRegionSizes,
      heap, numRanges, rangeFlags, heapRangeStartOffsets, rangeTileCounts, flags);
  }

  void STDMETHODCALLTYPE D3D12CommandQueueProxy::CopyTileMappings(ID3D12Resource* dstResource, const D3D12_TILED_RESOURCE_COORDINATE* dstRegionStartCoordinate, ID3D12Resource* srcResource, const D3D12_TILED_RESOURCE_COORDINATE* srcRegionStartCoordinate, const D3D12_TILE_REGION_SIZE* regionSize, D3D12_TILE_MAPPING_FLAGS flags) {
    m_queue->CopyTileMappings(dstResource, dstRegionStartCoordinate, srcResource,
      srcRegionStartCoordinate, regionSize, flags);
  }

  void STDMETHODCALLTYPE D3D12CommandQueueProxy::ExecuteCommandLists(UINT numCommandLists, ID3D12CommandList* const* commandLists) {
    log("ID3D12CommandQueue::ExecuteCommandLists(count=%u, first=%p)",
      numCommandLists, numCommandLists && commandLists ? commandLists[0] : nullptr);
    m_queue->ExecuteCommandLists(numCommandLists, commandLists);
  }

  void STDMETHODCALLTYPE D3D12CommandQueueProxy::SetMarker(UINT metadata, const void* data, UINT size) {
    m_queue->SetMarker(metadata, data, size);
  }

  void STDMETHODCALLTYPE D3D12CommandQueueProxy::BeginEvent(UINT metadata, const void* data, UINT size) {
    m_queue->BeginEvent(metadata, data, size);
  }

  void STDMETHODCALLTYPE D3D12CommandQueueProxy::EndEvent() {
    m_queue->EndEvent();
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::Signal(ID3D12Fence* fence, UINT64 value) {
    HRESULT result = m_queue->Signal(fence, value);
    log("ID3D12CommandQueue::Signal(fence=%p, value=%llu, result=0x%08lx)",
      fence, value, static_cast<unsigned long>(result));
    recordD3D12FenceSignal(m_queue, fence, value, result);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::Wait(ID3D12Fence* fence, UINT64 value) {
    HRESULT result = m_queue->Wait(fence, value);
    log("ID3D12CommandQueue::Wait(fence=%p, value=%llu, result=0x%08lx)",
      fence, value, static_cast<unsigned long>(result));
    recordD3D12FenceWait(m_queue, fence, value, result);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::GetTimestampFrequency(UINT64* frequency) {
    HRESULT result = m_queue->GetTimestampFrequency(frequency);
    log("ID3D12CommandQueue::GetTimestampFrequency(result=0x%08lx, frequency=%llu)",
      static_cast<unsigned long>(result), frequency ? *frequency : 0);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12CommandQueueProxy::GetClockCalibration(UINT64* gpuTimestamp, UINT64* cpuTimestamp) {
    HRESULT result = m_queue->GetClockCalibration(gpuTimestamp, cpuTimestamp);
    log("ID3D12CommandQueue::GetClockCalibration(result=0x%08lx, gpu=%llu, cpu=%llu)",
      static_cast<unsigned long>(result),
      gpuTimestamp ? *gpuTimestamp : 0,
      cpuTimestamp ? *cpuTimestamp : 0);
    return result;
  }

  D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE D3D12CommandQueueProxy::GetDesc() {
    D3D12_COMMAND_QUEUE_DESC result = m_queue->GetDesc();
    log("ID3D12CommandQueue::GetDesc(type=%u, flags=0x%x, node=%u)",
      static_cast<unsigned>(result.Type), result.Flags, result.NodeMask);
    return result;
  }

}
