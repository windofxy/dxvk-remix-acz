#include "d3d12_device_proxy.h"

#include "d3d12_resource_registry.h"

#include "d3d12_command_queue_proxy.h"
#include "d3d12_command_queue_trace.h"
#include "d3d12_command_list_trace.h"
#include "d3d12_device_extension_trace.h"
#include "acz_camera_hook.h"
#include "d3d12_device_raw_trace.h"
#include "d3d12_runtime_adapter.h"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <new>
#include <unordered_set>

namespace remix {

  namespace {

    std::mutex g_shaderCaptureMutex;
    std::unordered_set<uint64_t> g_capturedShaderHashes;

    uint64_t shaderHash(const void* data, size_t size) {
      const auto* bytes = static_cast<const uint8_t*>(data);
      uint64_t hash = 1469598103934665603ull;
      for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
      }
      return hash;
    }

    std::filesystem::path shaderCaptureDirectory() {
      wchar_t modulePath[MAX_PATH] = {};
      DWORD length = GetModuleFileNameW(nullptr, modulePath, ARRAYSIZE(modulePath));
      if (!length || length >= ARRAYSIZE(modulePath))
        return {};

      std::filesystem::path directory(modulePath);
      directory = directory.parent_path() / L"remix-d3d12-shaders";
      std::error_code error;
      std::filesystem::create_directories(directory, error);
      return directory;
    }

    void captureShader(const char* stage, const D3D12_SHADER_BYTECODE& bytecode) {
      if (!stage || !bytecode.pShaderBytecode || !bytecode.BytecodeLength)
        return;

      const uint64_t hash = shaderHash(bytecode.pShaderBytecode, bytecode.BytecodeLength);
      {
        std::lock_guard lock(g_shaderCaptureMutex);
        if (!g_capturedShaderHashes.emplace(hash).second)
          return;
      }

      const std::filesystem::path directory = shaderCaptureDirectory();
      if (directory.empty())
        return;

      wchar_t filename[96] = {};
      swprintf_s(filename, L"%S_%016llX_%zu.bin", stage,
        static_cast<unsigned long long>(hash), bytecode.BytecodeLength);
      const auto path = directory / filename;
      std::ofstream file(path, std::ios::binary);
      if (file)
        file.write(static_cast<const char*>(bytecode.pShaderBytecode), bytecode.BytecodeLength);

      log("D3D12 shader capture: stage=%s hash=0x%016llx bytes=%zu path=%ls written=%u",
        stage,
        static_cast<unsigned long long>(hash),
        bytecode.BytecodeLength,
        path.c_str(),
        file.good() ? 1u : 0u);
    }

    const char* inputFormatName(DXGI_FORMAT format) {
      switch (format) {
      case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
      case DXGI_FORMAT_R32G32B32_FLOAT: return "R32G32B32_FLOAT";
      case DXGI_FORMAT_R32G32_FLOAT: return "R32G32_FLOAT";
      case DXGI_FORMAT_R32_FLOAT: return "R32_FLOAT";
      case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
      case DXGI_FORMAT_R16G16_FLOAT: return "R16G16_FLOAT";
      case DXGI_FORMAT_R16_FLOAT: return "R16_FLOAT";
      case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
      case DXGI_FORMAT_R8G8B8A8_UINT: return "R8G8B8A8_UINT";
      case DXGI_FORMAT_R16G16B16A16_UNORM: return "R16G16B16A16_UNORM";
      case DXGI_FORMAT_R16G16_UNORM: return "R16G16_UNORM";
      case DXGI_FORMAT_R16_UINT: return "R16_UINT";
      case DXGI_FORMAT_R32_UINT: return "R32_UINT";
      default: return "UNKNOWN";
      }
    }

    // These policies are intentionally compile-time constants so a missing or
    // incorrectly inherited process environment cannot hide whether a probe is active.
    // Do not enable raw-device tracing: replacing the native ID3D12Device vtable
    // can terminate the process during its first command-list bootstrap. The queue
    // trace observes otherwise-unwrapped lists at submission time without touching
    // the device object.
    constexpr bool kEnableRawDeviceTrace = true;
    constexpr bool kEnableCommandQueueTrace = true;
    constexpr bool kEnableCommandListTrace = true;
    constexpr bool kEnableCommandQueueProxy = false;

  }

  D3D12DeviceProxy::D3D12DeviceProxy(ID3D12Device* device)
  : m_device(device),
    m_device2(nullptr) {
    device->QueryInterface(__uuidof(ID3D12Device2), reinterpret_cast<void**>(&m_device2));
    initializeD3D12RuntimeAdapter();
    attachD3D12Device(device);
    if (kEnableRawDeviceTrace)
      installRawDeviceTrace(m_device);
    installAczCameraHook();
    log("D3D12 proxy policy: raw-device-trace=%s, command-list-trace=%s, command-queue-trace=%s, command-queue-proxy=disabled",
      kEnableRawDeviceTrace ? "on" : "off",
      kEnableCommandListTrace ? "on" : "off",
      kEnableCommandQueueTrace ? "on" : "off");
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject)
      return E_POINTER;

    *ppvObject = nullptr;
    if (riid == __uuidof(IUnknown)
     || riid == __uuidof(ID3D12Object)
     || riid == __uuidof(ID3D12Device)
     || riid == __uuidof(ID3D12Device1)
     || riid == __uuidof(ID3D12Device2)) {
      *ppvObject = static_cast<ID3D12Device2*>(this);
      AddRef();
      return S_OK;
    }

    // Handing out the real device for a higher interface bypasses every hook in
    // this layer: the caller then creates buffers and their views through methods
    // this proxy never sees, so those buffers never reach the address registry.
    // Refusing the higher interfaces makes the caller fall back to ID3D12Device2,
    // which this proxy implements, so creation stays observable.
    if (riid == __uuidof(ID3D12Device3) || isDevice4OrLaterIid(riid)) {
      log("ID3D12Device::QueryInterface(riid=%s) refused so creation stays proxied",
        formatIid(riid));
      return E_NOINTERFACE;
    }

    HRESULT result = m_device->QueryInterface(riid, ppvObject);
    bool traceSupported = isDevice4OrLaterIid(riid);
    bool traceInstalled = false;
    if (SUCCEEDED(result) && ppvObject && *ppvObject) {
      if (traceSupported)
        traceInstalled = installDevice4Trace(static_cast<ID3D12Device4*>(*ppvObject));
      // Higher device interfaces expose the same base Create* methods through
      // their own vtable. Patch those vtables too so SRV metadata is captured
      // when the game bypasses the ID3D12Device2 proxy.
      installRawDeviceTrace(static_cast<ID3D12Device*>(*ppvObject));
    }
    log("ID3D12Device::QueryInterface(riid=%s, result=0x%08lx, object=%p, device4TraceSupported=%u, device4TraceInstalled=%u)",
      formatIid(riid), static_cast<unsigned long>(result),
      ppvObject ? *ppvObject : nullptr,
      traceSupported ? 1u : 0u, traceInstalled ? 1u : 0u);
    return result;
  }

  ULONG STDMETHODCALLTYPE D3D12DeviceProxy::AddRef() {
    return ++m_refs;
  }

  ULONG STDMETHODCALLTYPE D3D12DeviceProxy::Release() {
    ULONG refs = --m_refs;
    if (!refs) {
      if (m_device2)
        m_device2->Release();
      m_device->Release();
      delete this;
    }
    return refs;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::GetPrivateData(REFGUID guid, UINT* dataSize, void* data) {
    return m_device->GetPrivateData(guid, dataSize, data);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::SetPrivateData(REFGUID guid, UINT dataSize, const void* data) {
    return m_device->SetPrivateData(guid, dataSize, data);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::SetPrivateDataInterface(REFGUID guid, const IUnknown* data) {
    return m_device->SetPrivateDataInterface(guid, data);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::SetName(LPCWSTR name) {
    return m_device->SetName(name);
  }

  UINT STDMETHODCALLTYPE D3D12DeviceProxy::GetNodeCount() {
    UINT result = m_device->GetNodeCount();
    log("ID3D12Device::GetNodeCount(result=%u)", result);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid, void** commandQueue) {
    HRESULT result = m_device->CreateCommandQueue(desc, riid, commandQueue);
    bool queueTraceInstalled = false;
    if (kEnableCommandQueueTrace && SUCCEEDED(result) && commandQueue && *commandQueue
     && riid == __uuidof(ID3D12CommandQueue)) {
      queueTraceInstalled = installCommandQueueTrace(
        static_cast<ID3D12CommandQueue*>(*commandQueue));
    }

    if (SUCCEEDED(result) && desc && commandQueue && *commandQueue
     && riid == __uuidof(ID3D12CommandQueue)) {
      attachD3D12CommandQueue(static_cast<ID3D12CommandQueue*>(*commandQueue), *desc);
    }

    if (SUCCEEDED(result) && kEnableCommandQueueProxy && commandQueue && *commandQueue
     && riid == __uuidof(ID3D12CommandQueue)) {
      auto* queue = static_cast<ID3D12CommandQueue*>(*commandQueue);
      auto* proxy = new (std::nothrow) D3D12CommandQueueProxy(
        queue, static_cast<ID3D12Device*>(this));
      if (proxy) {
        *commandQueue = proxy;
        log("Wrapped ID3D12CommandQueue at %p", proxy);
      } else {
        queue->Release();
        *commandQueue = nullptr;
        result = E_OUTOFMEMORY;
      }
    }
    log("ID3D12Device::CreateCommandQueue(type=%u, flags=0x%x, node=%u, riid=%s, result=0x%08lx, object=%p, trace=%s, traceInstalled=%u)",
      desc ? static_cast<unsigned>(desc->Type) : 0,
      desc ? desc->Flags : 0,
      desc ? desc->NodeMask : 0,
      formatIid(riid),
      static_cast<unsigned long>(result),
      commandQueue ? *commandQueue : nullptr,
      kEnableCommandQueueTrace ? "on" : "off",
      queueTraceInstalled ? 1u : 0u);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** commandAllocator) {
    HRESULT result = m_device->CreateCommandAllocator(type, riid, commandAllocator);
    log("ID3D12Device::CreateCommandAllocator(type=%u, result=0x%08lx, object=%p)",
      static_cast<unsigned>(type),
      static_cast<unsigned long>(result),
      commandAllocator ? *commandAllocator : nullptr);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc, REFIID riid, void** pipelineState) {
    HRESULT result = m_device->CreateGraphicsPipelineState(desc, riid, pipelineState);
    recordD3D12Pipeline(desc,
      pipelineState && *pipelineState ? static_cast<ID3D12PipelineState*>(*pipelineState) : nullptr,
      result);
    log("ID3D12Device::CreateGraphicsPipelineState(desc=%p, vs=%zu, ps=%zu, inputElements=%u, topology=%u, rtCount=%u, rt0=%u, dsv=%u, result=0x%08lx, object=%p)",
      desc,
      desc ? desc->VS.BytecodeLength : 0u,
      desc ? desc->PS.BytecodeLength : 0u,
      desc ? desc->InputLayout.NumElements : 0u,
      desc ? static_cast<unsigned>(desc->PrimitiveTopologyType) : 0u,
      desc ? desc->NumRenderTargets : 0u,
      desc ? static_cast<unsigned>(desc->RTVFormats[0]) : 0u,
      desc ? static_cast<unsigned>(desc->DSVFormat) : 0u,
      static_cast<unsigned long>(result), pipelineState ? *pipelineState : nullptr);

    if (desc && desc->InputLayout.pInputElementDescs) {
      for (UINT i = 0; i < desc->InputLayout.NumElements; ++i) {
        const D3D12_INPUT_ELEMENT_DESC& element = desc->InputLayout.pInputElementDescs[i];
        log("D3D12 input layout: pipeline=%p element=%u semantic=%s semanticIndex=%u format=%u(%s) offset=%u slot=%u class=%u stepRate=%u",
          pipelineState ? *pipelineState : nullptr,
          i,
          element.SemanticName ? element.SemanticName : "<null>",
          element.SemanticIndex,
          static_cast<unsigned>(element.Format),
          inputFormatName(element.Format),
          element.AlignedByteOffset,
          element.InputSlot,
          static_cast<unsigned>(element.InputSlotClass),
          element.InstanceDataStepRate);
      }
    }
    if (desc) {
      captureShader("vs", desc->VS);
      captureShader("ps", desc->PS);
    }
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID riid, void** pipelineState) {
    HRESULT result = m_device->CreateComputePipelineState(desc, riid, pipelineState);
    log("ID3D12Device::CreateComputePipelineState(desc=%p, result=0x%08lx, object=%p)",
      desc, static_cast<unsigned long>(result), pipelineState ? *pipelineState : nullptr);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateCommandList(UINT nodeMask, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* commandAllocator, ID3D12PipelineState* initialState, REFIID riid, void** commandList) {
    HRESULT result = m_device->CreateCommandList(nodeMask, type, commandAllocator, initialState, riid, commandList);
    bool traceSupported = isGraphicsCommandListIid(riid);
    bool traceInstalled = false;
    if (SUCCEEDED(result) && kEnableCommandListTrace && traceSupported
     && commandList && *commandList) {
      traceInstalled = installCommandListTrace(
        static_cast<ID3D12GraphicsCommandList*>(*commandList));
    }
    if (SUCCEEDED(result) && traceSupported && commandList && *commandList)
      attachD3D12CommandList(static_cast<ID3D12GraphicsCommandList*>(*commandList));
    log("ID3D12Device::CreateCommandList(node=%u, type=%u, allocator=%p, initialState=%p, riid=%s, result=0x%08lx, object=%p, trace=%s, traceSupported=%u, traceInstalled=%u)",
      nodeMask, static_cast<unsigned>(type), commandAllocator, initialState,
      formatIid(riid), static_cast<unsigned long>(result),
      commandList ? *commandList : nullptr,
      kEnableCommandListTrace ? "on" : "off",
      traceSupported ? 1u : 0u, traceInstalled ? 1u : 0u);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CheckFeatureSupport(D3D12_FEATURE feature, void* featureSupportData, UINT featureSupportDataSize) {
    HRESULT result = m_device->CheckFeatureSupport(feature, featureSupportData, featureSupportDataSize);
    log("ID3D12Device::CheckFeatureSupport(feature=%u, size=%u, result=0x%08lx)",
      static_cast<unsigned>(feature), featureSupportDataSize, static_cast<unsigned long>(result));
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC* descriptorHeapDesc, REFIID riid, void** heap) {
    HRESULT result = m_device->CreateDescriptorHeap(descriptorHeapDesc, riid, heap);
    if (SUCCEEDED(result) && descriptorHeapDesc && heap && *heap
     && riid == __uuidof(ID3D12DescriptorHeap)) {
      auto* descriptorHeap = static_cast<ID3D12DescriptorHeap*>(*heap);
      recordD3D12DescriptorHeap(
        descriptorHeap,
        *descriptorHeapDesc,
        descriptorHeap->GetCPUDescriptorHandleForHeapStart(),
        descriptorHeap->GetGPUDescriptorHandleForHeapStart(),
        m_device->GetDescriptorHandleIncrementSize(descriptorHeapDesc->Type));
    }
    log("ID3D12Device::CreateDescriptorHeap(desc=%p, result=0x%08lx, object=%p)",
      descriptorHeapDesc, static_cast<unsigned long>(result), heap ? *heap : nullptr);
    return result;
  }

  UINT STDMETHODCALLTYPE D3D12DeviceProxy::GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapType) {
    UINT result = m_device->GetDescriptorHandleIncrementSize(descriptorHeapType);
    log("ID3D12Device::GetDescriptorHandleIncrementSize(type=%u, result=%u)",
      static_cast<unsigned>(descriptorHeapType), result);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateRootSignature(UINT nodeMask, const void* blobWithRootSignature, SIZE_T blobLengthInBytes, REFIID riid, void** rootSignature) {
    HRESULT result = m_device->CreateRootSignature(nodeMask, blobWithRootSignature, blobLengthInBytes, riid, rootSignature);
    log("ID3D12Device::CreateRootSignature(node=%u, blob=%p, size=%zu, result=0x%08lx, object=%p)",
      nodeMask, blobWithRootSignature, blobLengthInBytes, static_cast<unsigned long>(result),
      rootSignature ? *rootSignature : nullptr);
    return result;
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
    m_device->CreateConstantBufferView(desc, destDescriptor);
    recordD3D12ConstantBufferView(destDescriptor, desc);
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CreateShaderResourceView(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
    m_device->CreateShaderResourceView(resource, desc, destDescriptor);
    recordD3D12View("SRV", destDescriptor, resource, desc ? desc->Format : DXGI_FORMAT_UNKNOWN,
      desc, desc ? sizeof(*desc) : 0);
    if (desc) {
      UINT64 firstElement = 0;
      UINT numElements = 0;
      UINT structureStride = 0;
      if (desc->ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
        firstElement = desc->Buffer.FirstElement;
        numElements = desc->Buffer.NumElements;
        structureStride = desc->Buffer.StructureByteStride;
      }
      log("D3D12 SRV description: cpu=0x%llx resource=%p dimension=%u format=%u first=%llu elements=%u stride=%u",
        static_cast<unsigned long long>(destDescriptor.ptr), resource,
        static_cast<unsigned>(desc->ViewDimension), static_cast<unsigned>(desc->Format),
        static_cast<unsigned long long>(firstElement), numElements, structureStride);
    }
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CreateUnorderedAccessView(ID3D12Resource* resource, ID3D12Resource* counterResource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
    m_device->CreateUnorderedAccessView(resource, counterResource, desc, destDescriptor);
    recordD3D12View("UAV", destDescriptor, resource, desc ? desc->Format : DXGI_FORMAT_UNKNOWN,
      desc, desc ? sizeof(*desc) : 0);
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CreateRenderTargetView(ID3D12Resource* resource, const D3D12_RENDER_TARGET_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
    m_device->CreateRenderTargetView(resource, desc, destDescriptor);
    recordD3D12View("RTV", destDescriptor, resource, desc ? desc->Format : DXGI_FORMAT_UNKNOWN,
      desc, desc ? sizeof(*desc) : 0);
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CreateDepthStencilView(ID3D12Resource* resource, const D3D12_DEPTH_STENCIL_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
    m_device->CreateDepthStencilView(resource, desc, destDescriptor);
    recordD3D12View("DSV", destDescriptor, resource, desc ? desc->Format : DXGI_FORMAT_UNKNOWN,
      desc, desc ? sizeof(*desc) : 0);
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CreateSampler(const D3D12_SAMPLER_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
    m_device->CreateSampler(desc, destDescriptor);
    recordD3D12Sampler(destDescriptor, desc);
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CopyDescriptors(UINT numDestDescriptorRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* destDescriptorRangeStarts, const UINT* destDescriptorRangeSizes, UINT numSrcDescriptorRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* srcDescriptorRangeStarts, const UINT* srcDescriptorRangeSizes, D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapsType) {
    m_device->CopyDescriptors(numDestDescriptorRanges, destDescriptorRangeStarts, destDescriptorRangeSizes,
      numSrcDescriptorRanges, srcDescriptorRangeStarts, srcDescriptorRangeSizes, descriptorHeapsType);
    recordD3D12CopyDescriptors(numDestDescriptorRanges, destDescriptorRangeStarts, destDescriptorRangeSizes,
      numSrcDescriptorRanges, srcDescriptorRangeStarts, srcDescriptorRangeSizes,
      m_device->GetDescriptorHandleIncrementSize(descriptorHeapsType));
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::CopyDescriptorsSimple(UINT numDescriptors, D3D12_CPU_DESCRIPTOR_HANDLE destDescriptorRangeStart, D3D12_CPU_DESCRIPTOR_HANDLE srcDescriptorRangeStart, D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapsType) {
    m_device->CopyDescriptorsSimple(numDescriptors, destDescriptorRangeStart, srcDescriptorRangeStart, descriptorHeapsType);
    recordD3D12CopyDescriptorsSimple(numDescriptors, destDescriptorRangeStart, srcDescriptorRangeStart,
      m_device->GetDescriptorHandleIncrementSize(descriptorHeapsType));
  }

  D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE D3D12DeviceProxy::GetResourceAllocationInfo(UINT visibleMask, UINT numResourceDescs, const D3D12_RESOURCE_DESC* resourceDescs) {
    return m_device->GetResourceAllocationInfo(visibleMask, numResourceDescs, resourceDescs);
  }

  D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE D3D12DeviceProxy::GetCustomHeapProperties(UINT nodeMask, D3D12_HEAP_TYPE heapType) {
    return m_device->GetCustomHeapProperties(nodeMask, heapType);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateCommittedResource(const D3D12_HEAP_PROPERTIES* heapProperties, D3D12_HEAP_FLAGS heapFlags, const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initialResourceState, const D3D12_CLEAR_VALUE* optimizedClearValue, REFIID riidResource, void** resource) {
    HRESULT result = m_device->CreateCommittedResource(heapProperties, heapFlags, desc, initialResourceState, optimizedClearValue, riidResource, resource);
    recordD3D12Resource(heapProperties, desc,
      resource && *resource ? static_cast<ID3D12Resource*>(*resource) : nullptr,
      result);
    log("ID3D12Device::CreateCommittedResource(desc=%p, dimension=%u, width=%llu, height=%u, depthOrArray=%u, mips=%u, format=%u, flags=0x%x, heapType=%u, state=0x%x, result=0x%08lx, object=%p)",
      desc,
      desc ? static_cast<unsigned>(desc->Dimension) : 0u,
      desc ? static_cast<unsigned long long>(desc->Width) : 0ull,
      desc ? desc->Height : 0u,
      desc ? desc->DepthOrArraySize : 0u,
      desc ? desc->MipLevels : 0u,
      desc ? static_cast<unsigned>(desc->Format) : 0u,
      desc ? static_cast<unsigned>(desc->Flags) : 0u,
      heapProperties ? static_cast<unsigned>(heapProperties->Type) : 0u,
      initialResourceState, static_cast<unsigned long>(result), resource ? *resource : nullptr);

    if (SUCCEEDED(result) && resource && *resource && heapProperties && desc) {
      ID3D12Resource* baseResource = nullptr;
      HRESULT queryResult = reinterpret_cast<IUnknown*>(*resource)->QueryInterface(
        __uuidof(ID3D12Resource), reinterpret_cast<void**>(&baseResource));
      if (SUCCEEDED(queryResult) && baseResource) {
        registerResource(baseResource, heapProperties, desc);
        baseResource->Release();
      }
    }

    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateHeap(const D3D12_HEAP_DESC* desc, REFIID riid, void** heap) {
    return m_device->CreateHeap(desc, riid, heap);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreatePlacedResource(ID3D12Heap* heap, UINT64 heapOffset, const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE* optimizedClearValue, REFIID riid, void** resource) {
    HRESULT result = m_device->CreatePlacedResource(heap, heapOffset, desc, initialState, optimizedClearValue, riid, resource);
    if (SUCCEEDED(result) && desc && resource && *resource && riid == __uuidof(ID3D12Resource)) {
      auto* placedResource = static_cast<ID3D12Resource*>(*resource);
      D3D12_HEAP_PROPERTIES heapProperties {};
      heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
      recordD3D12Resource(&heapProperties, desc, placedResource, result);
      registerResource(placedResource, &heapProperties, desc);
    }
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateReservedResource(const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE* optimizedClearValue, REFIID riid, void** resource) {
    HRESULT result = m_device->CreateReservedResource(desc, initialState, optimizedClearValue, riid, resource);
    if (SUCCEEDED(result) && desc && resource && *resource && riid == __uuidof(ID3D12Resource)) {
      auto* reservedResource = static_cast<ID3D12Resource*>(*resource);
      recordD3D12Resource(nullptr, desc, reservedResource, result);
      D3D12_HEAP_PROPERTIES heapProperties {};
      heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
      registerResource(reservedResource, &heapProperties, desc);
    }
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateSharedHandle(ID3D12DeviceChild* object, const SECURITY_ATTRIBUTES* attributes, DWORD access, LPCWSTR name, HANDLE* handle) {
    return m_device->CreateSharedHandle(object, attributes, access, name, handle);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::OpenSharedHandle(HANDLE ntHandle, REFIID riid, void** object) {
    return m_device->OpenSharedHandle(ntHandle, riid, object);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::OpenSharedHandleByName(LPCWSTR name, DWORD access, HANDLE* ntHandle) {
    return m_device->OpenSharedHandleByName(name, access, ntHandle);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::MakeResident(UINT numObjects, ID3D12Pageable* const* objects) {
    return m_device->MakeResident(numObjects, objects);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::Evict(UINT numObjects, ID3D12Pageable* const* objects) {
    return m_device->Evict(numObjects, objects);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateFence(UINT64 initialValue, D3D12_FENCE_FLAGS flags, REFIID riid, void** fence) {
    HRESULT result = m_device->CreateFence(initialValue, flags, riid, fence);
    log("ID3D12Device::CreateFence(initial=%llu, flags=0x%x, result=0x%08lx, object=%p)",
      initialValue, flags, static_cast<unsigned long>(result), fence ? *fence : nullptr);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::GetDeviceRemovedReason() {
    HRESULT result = m_device->GetDeviceRemovedReason();
    log("ID3D12Device::GetDeviceRemovedReason(result=0x%08lx)", static_cast<unsigned long>(result));
    return result;
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::GetCopyableFootprints(const D3D12_RESOURCE_DESC* resourceDesc, UINT firstSubresource, UINT numSubresources, UINT64 baseOffset, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts, UINT* numRows, UINT64* rowSizeInBytes, UINT64* totalBytes) {
    m_device->GetCopyableFootprints(resourceDesc, firstSubresource, numSubresources, baseOffset, layouts, numRows, rowSizeInBytes, totalBytes);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateQueryHeap(const D3D12_QUERY_HEAP_DESC* desc, REFIID riid, void** heap) {
    return m_device->CreateQueryHeap(desc, riid, heap);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::SetStablePowerState(BOOL enable) {
    return m_device->SetStablePowerState(enable);
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreateCommandSignature(const D3D12_COMMAND_SIGNATURE_DESC* desc, ID3D12RootSignature* rootSignature, REFIID riid, void** commandSignature) {
    return m_device->CreateCommandSignature(desc, rootSignature, riid, commandSignature);
  }

  void STDMETHODCALLTYPE D3D12DeviceProxy::GetResourceTiling(ID3D12Resource* tiledResource, UINT* numTilesForEntireResource, D3D12_PACKED_MIP_INFO* packedMipDesc, D3D12_TILE_SHAPE* standardTileShapeForNonPackedMips, UINT* numSubresourceTilings, UINT firstSubresourceTilingToGet, D3D12_SUBRESOURCE_TILING* subresourceTilingsForNonPackedMips) {
    m_device->GetResourceTiling(tiledResource, numTilesForEntireResource, packedMipDesc,
      standardTileShapeForNonPackedMips, numSubresourceTilings,
      firstSubresourceTilingToGet, subresourceTilingsForNonPackedMips);
  }

  LUID STDMETHODCALLTYPE D3D12DeviceProxy::GetAdapterLuid() {
    return m_device->GetAdapterLuid();
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreatePipelineLibrary(const void* libraryBlob, SIZE_T blobLength, REFIID riid, void** pipelineLibrary) {
    if (!m_device2)
      return E_NOINTERFACE;

    HRESULT result = m_device2->CreatePipelineLibrary(libraryBlob, blobLength, riid, pipelineLibrary);
    log("ID3D12Device1::CreatePipelineLibrary(blob=%p, size=%zu, result=0x%08lx, object=%p)",
      libraryBlob, blobLength, static_cast<unsigned long>(result),
      pipelineLibrary ? *pipelineLibrary : nullptr);
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::SetEventOnMultipleFenceCompletion(ID3D12Fence* const* fences, const UINT64* fenceValues, UINT numFences, D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE event) {
    if (!m_device2)
      return E_NOINTERFACE;

    HRESULT result = m_device2->SetEventOnMultipleFenceCompletion(fences, fenceValues, numFences, flags, event);
    log("ID3D12Device1::SetEventOnMultipleFenceCompletion(count=%u, flags=0x%x, result=0x%08lx)",
      numFences, flags, static_cast<unsigned long>(result));
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::SetResidencyPriority(UINT numObjects, ID3D12Pageable* const* objects, const D3D12_RESIDENCY_PRIORITY* priorities) {
    if (!m_device2)
      return E_NOINTERFACE;

    HRESULT result = m_device2->SetResidencyPriority(numObjects, objects, priorities);
    log("ID3D12Device1::SetResidencyPriority(count=%u, result=0x%08lx)",
      numObjects, static_cast<unsigned long>(result));
    return result;
  }

  HRESULT STDMETHODCALLTYPE D3D12DeviceProxy::CreatePipelineState(const D3D12_PIPELINE_STATE_STREAM_DESC* desc, REFIID riid, void** pipelineState) {
    if (!m_device2)
      return E_NOINTERFACE;

    HRESULT result = m_device2->CreatePipelineState(desc, riid, pipelineState);
    log("ID3D12Device2::CreatePipelineState(desc=%p, streamSize=%zu, result=0x%08lx, object=%p)",
      desc, desc ? desc->SizeInBytes : 0u,
      static_cast<unsigned long>(result), pipelineState ? *pipelineState : nullptr);
    return result;
  }

}
