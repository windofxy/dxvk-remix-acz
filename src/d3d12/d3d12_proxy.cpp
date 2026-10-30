#include <windows.h>
#include <d3d12.h>

#include "d3d12_device_proxy.h"
#include "d3d12_runtime_adapter.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <new>
#include <string>

namespace remix {

  HMODULE g_realD3d12 = nullptr;
  std::once_flag g_realD3d12Once;
  std::once_flag g_logFileOnce;
  HANDLE g_logFile = INVALID_HANDLE_VALUE;
  std::mutex g_logMutex;

  constexpr bool kVerboseLogging = false;

  bool isHighFrequencyLog(const char* format) {
    if (kVerboseLogging || !format)
      return false;
    constexpr const char* prefixes[] = {
      // Per-command-list chatter that dominates the log volume.
      "ID3D12GraphicsCommandList",
      "D3D12 root CBV",
      "D3D12 draw counters",
      "ID3D12CommandQueue::ExecuteCommandLists",
      "Reinstalled command list trace",
      "Installed command list trace",
      "Command list interface probe",
      // Frontend and backend bookkeeping.
      "D3D12 native frontend draw",
      "D3D12 native frontend pipeline",
      "D3D12 native frontend resource",
      "D3D12 native frontend view",
      "D3D12 native frontend descriptor",
      "D3D12 native frontend submit",
      "D3D12 native frontend frame summary",
      "D3D12 native frontend fence",
      "D3D12 native frontend command list",
      "D3D12 native frontend frame begin",
      "D3D12 resource registry",
      "D3D12 texture snapshot",
      "D3D12 transform lookup",
      "D3D12 native draw",
      "D3D12 rejected draw",
      "D3D12 binding state",
      "D3D12 native Remix draw",
      "D3D12 native Remix scene submit",
      "D3D12 native Remix submit state",
      "D3D12 native Remix prepare",
      "D3D12 native Remix scene state",
      "D3D12 native Remix external sync",
      "D3D12 native Remix output handoff",
      "D3D12 native Remix camera updated",
    };
    for (const char* prefix : prefixes) {
      if (std::strncmp(format, prefix, std::strlen(prefix)) == 0)
        return true;
    }
    return false;
  }

  const char* formatIid(REFIID iid) {
    static thread_local char text[64] = {};
    _snprintf_s(text, sizeof(text), _TRUNCATE,
      "%08lX-%04hX-%04hX-%02hX%02hX-%02hX%02hX%02hX%02hX%02hX%02hX",
      static_cast<unsigned long>(iid.Data1),
      iid.Data2,
      iid.Data3,
      iid.Data4[0], iid.Data4[1],
      iid.Data4[2], iid.Data4[3], iid.Data4[4], iid.Data4[5], iid.Data4[6], iid.Data4[7]);
    return text;
  }

  std::wstring getLogPath() {
    wchar_t modulePath[MAX_PATH] = {};
    DWORD length = GetModuleFileNameW(nullptr, modulePath, ARRAYSIZE(modulePath));
    if (!length || length >= ARRAYSIZE(modulePath))
      return L"remix-d3d12-proxy.log";

    wchar_t* separator = wcsrchr(modulePath, L'\\');
    if (separator)
      *(separator + 1) = L'\0';
    else
      modulePath[0] = L'\0';

    return std::wstring(modulePath) + L"remix-d3d12-proxy.log";
  }

  void log(const char* format, ...) {
    if (isHighFrequencyLog(format))
      return;

    char message[1024] = {};
    va_list args;
    va_start(args, format);
    _vsnprintf_s(message, sizeof(message), _TRUNCATE, format, args);
    va_end(args);

    OutputDebugStringA(message);
    OutputDebugStringA("\n");

    std::call_once(g_logFileOnce, [] {
      g_logFile = CreateFileW(
        getLogPath().c_str(),
        FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    });

    std::lock_guard lock(g_logMutex);
    if (g_logFile == INVALID_HANDLE_VALUE)
      return;

    DWORD written = 0;
    WriteFile(g_logFile, message, static_cast<DWORD>(strlen(message)), &written, nullptr);
    WriteFile(g_logFile, "\r\n", 2, &written, nullptr);
  }

  HMODULE loadRealD3d12() {
    std::call_once(g_realD3d12Once, [] {
      wchar_t systemPath[MAX_PATH] = {};
      UINT length = GetSystemDirectoryW(systemPath, ARRAYSIZE(systemPath));
      if (!length || length >= ARRAYSIZE(systemPath) - 11) {
        log("GetSystemDirectoryW failed: %lu", GetLastError());
        return;
      }

      wcscat_s(systemPath, L"\\d3d12.dll");
      g_realD3d12 = LoadLibraryExW(systemPath, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
      if (!g_realD3d12)
        log("LoadLibraryExW(%ls) failed: %lu", systemPath, GetLastError());
      else
        log("Loaded system D3D12: %ls", systemPath);
    });

    return g_realD3d12;
  }

  template<typename T>
  T resolve(const char* name) {
    HMODULE module = loadRealD3d12();
    if (!module)
      return nullptr;

    FARPROC address = GetProcAddress(module, name);
    if (!address) {
      log("GetProcAddress(%s) failed: %lu", name, GetLastError());
      return nullptr;
    }

    log("Resolved %s at %p", name, reinterpret_cast<void*>(address));
    return reinterpret_cast<T>(address);
  }

}

using remix::log;
using remix::formatIid;
using remix::resolve;

extern "C" {

  HRESULT WINAPI D3D12CreateDevice(
      IUnknown* pAdapter,
      D3D_FEATURE_LEVEL minimumFeatureLevel,
      REFIID riid,
      void** ppDevice) {
    log("D3D12CreateDevice begin(adapter=%p, featureLevel=0x%lx, riid=%s, output=%p)",
      pAdapter,
      static_cast<unsigned long>(minimumFeatureLevel),
      formatIid(riid),
      ppDevice);

    using Function = HRESULT (WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    Function function = resolve<Function>("D3D12CreateDevice");
    if (!function)
      return E_FAIL;

    HRESULT result = function(pAdapter, minimumFeatureLevel, riid, ppDevice);
    if (SUCCEEDED(result) && ppDevice && *ppDevice
     && (riid == __uuidof(ID3D12Device)
      || riid == __uuidof(ID3D12Device1)
      || riid == __uuidof(ID3D12Device2))) {
      ID3D12Device* device = static_cast<ID3D12Device*>(*ppDevice);
      auto* proxy = new (std::nothrow) remix::D3D12DeviceProxy(device);
      if (proxy) {
        *ppDevice = proxy;
        remix::initializeD3D12RuntimeAdapter();
        remix::attachD3D12Device(static_cast<ID3D12Device*>(proxy));
        log("Wrapped ID3D12Device at %p", proxy);
      } else {
        device->Release();
        *ppDevice = nullptr;
        result = E_OUTOFMEMORY;
      }
    }
    log("D3D12CreateDevice(adapter=%p, featureLevel=0x%lx, result=0x%08lx, device=%p)",
      pAdapter,
      static_cast<unsigned long>(minimumFeatureLevel),
      static_cast<unsigned long>(result),
      ppDevice ? *ppDevice : nullptr);
    return result;
  }

  HRESULT WINAPI D3D12GetDebugInterface(REFIID riid, void** ppvDebug) {
    using Function = HRESULT (WINAPI*)(REFIID, void**);
    Function function = resolve<Function>("D3D12GetDebugInterface");
    HRESULT result = function ? function(riid, ppvDebug) : E_FAIL;
    log("D3D12GetDebugInterface(riid=%p, output=%p, result=0x%08lx, object=%p)",
      &riid,
      ppvDebug,
      static_cast<unsigned long>(result),
      ppvDebug ? *ppvDebug : nullptr);
    return result;
  }

  HRESULT WINAPI D3D12GetInterface(REFCLSID rclsid, REFIID riid, void** ppv) {
    using Function = HRESULT (WINAPI*)(REFCLSID, REFIID, void**);
    Function function = resolve<Function>("D3D12GetInterface");
    HRESULT result = function ? function(rclsid, riid, ppv) : E_FAIL;
    log("D3D12GetInterface(class=%p, riid=%p, output=%p, result=0x%08lx, object=%p)",
      &rclsid,
      &riid,
      ppv,
      static_cast<unsigned long>(result),
      ppv ? *ppv : nullptr);
    return result;
  }

  HRESULT WINAPI D3D12EnableExperimentalFeatures(
      UINT numFeatures,
      const IID* iids,
      void* configurationStructs,
      UINT* configurationStructSizes) {
    using Function = HRESULT (WINAPI*)(UINT, const IID*, void*, UINT*);
    Function function = resolve<Function>("D3D12EnableExperimentalFeatures");
    HRESULT result = function
      ? function(numFeatures, iids, configurationStructs, configurationStructSizes)
      : E_FAIL;
    log("D3D12EnableExperimentalFeatures(count=%u, result=0x%08lx)",
      numFeatures,
      static_cast<unsigned long>(result));
    return result;
  }

  HRESULT WINAPI D3D12SerializeRootSignature(
      const D3D12_ROOT_SIGNATURE_DESC* rootSignature,
      D3D_ROOT_SIGNATURE_VERSION version,
      ID3DBlob** blob,
      ID3DBlob** errorBlob) {
    using Function = HRESULT (WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
    Function function = resolve<Function>("D3D12SerializeRootSignature");
    HRESULT result = function ? function(rootSignature, version, blob, errorBlob) : E_FAIL;
    log("D3D12SerializeRootSignature(desc=%p, version=%u, result=0x%08lx)",
      rootSignature,
      static_cast<unsigned>(version),
      static_cast<unsigned long>(result));
    return result;
  }

  HRESULT WINAPI D3D12SerializeVersionedRootSignature(
      const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* rootSignature,
      ID3DBlob** blob,
      ID3DBlob** errorBlob) {
    using Function = HRESULT (WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
    Function function = resolve<Function>("D3D12SerializeVersionedRootSignature");
    HRESULT result = function ? function(rootSignature, blob, errorBlob) : E_FAIL;
    log("D3D12SerializeVersionedRootSignature(desc=%p, result=0x%08lx)",
      rootSignature,
      static_cast<unsigned long>(result));
    return result;
  }

  HRESULT WINAPI D3D12CreateRootSignatureDeserializer(
      LPCVOID sourceData,
      SIZE_T sourceDataSizeInBytes,
      REFIID riid,
      void** ppvDeserializer) {
    using Function = HRESULT (WINAPI*)(LPCVOID, SIZE_T, REFIID, void**);
    Function function = resolve<Function>("D3D12CreateRootSignatureDeserializer");
    HRESULT result = function
      ? function(sourceData, sourceDataSizeInBytes, riid, ppvDeserializer)
      : E_FAIL;
    log("D3D12CreateRootSignatureDeserializer(data=%p, size=%zu, result=0x%08lx, object=%p)",
      sourceData,
      sourceDataSizeInBytes,
      static_cast<unsigned long>(result),
      ppvDeserializer ? *ppvDeserializer : nullptr);
    return result;
  }

  HRESULT WINAPI D3D12CreateVersionedRootSignatureDeserializer(
      LPCVOID sourceData,
      SIZE_T sourceDataSizeInBytes,
      REFIID riid,
      void** ppvDeserializer) {
    using Function = HRESULT (WINAPI*)(LPCVOID, SIZE_T, REFIID, void**);
    Function function = resolve<Function>("D3D12CreateVersionedRootSignatureDeserializer");
    HRESULT result = function
      ? function(sourceData, sourceDataSizeInBytes, riid, ppvDeserializer)
      : E_FAIL;
    log("D3D12CreateVersionedRootSignatureDeserializer(data=%p, size=%zu, result=0x%08lx, object=%p)",
      sourceData,
      sourceDataSizeInBytes,
      static_cast<unsigned long>(result),
      ppvDeserializer ? *ppvDeserializer : nullptr);
    return result;
  }

}
