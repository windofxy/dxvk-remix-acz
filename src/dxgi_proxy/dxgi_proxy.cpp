#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>

namespace {

  HMODULE g_realDxgi = nullptr;
  std::once_flag g_realDxgiOnce;
  std::mutex g_logMutex;
  std::mutex g_factoryHookMutex;

  using CreateSwapChainForHwndFunction = HRESULT (STDMETHODCALLTYPE*) (
    IDXGIFactory2*,
    IUnknown*,
    HWND,
    const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*,
    IDXGIOutput*,
    IDXGISwapChain1**);

  struct FactoryHook {
    void** originalVtable = nullptr;
    void** clonedVtable = nullptr;
    CreateSwapChainForHwndFunction originalCreateSwapChainForHwnd = nullptr;
  };

  using SwapChainPresentFunction = HRESULT (STDMETHODCALLTYPE*) (
    IDXGISwapChain*,
    UINT,
    UINT);

  using SwapChainGetBufferFunction = HRESULT (STDMETHODCALLTYPE*) (
    IDXGISwapChain*,
    UINT,
    REFIID,
    void**);

  using SwapChainGetCurrentBackBufferIndexFunction = UINT (STDMETHODCALLTYPE*) (
    IDXGISwapChain3*);

  using SwapChainResizeBuffersFunction = HRESULT (STDMETHODCALLTYPE*) (
    IDXGISwapChain*,
    UINT,
    UINT,
    UINT,
    DXGI_FORMAT,
    UINT);

  struct SwapChainHook {
    void** originalVtable = nullptr;
    void** clonedVtable = nullptr;
    size_t vtableEntries = 0;
    SwapChainPresentFunction originalPresent = nullptr;
    SwapChainGetBufferFunction originalGetBuffer = nullptr;
    SwapChainGetCurrentBackBufferIndexFunction originalGetCurrentBackBufferIndex = nullptr;
    SwapChainResizeBuffersFunction originalResizeBuffers = nullptr;
    std::atomic_uint64_t presentCount = 0;
    std::atomic_uint64_t backBufferIndexCount = 0;
  };

  std::unordered_map<void*, FactoryHook*> g_factoryHooks;
  std::unordered_map<void*, SwapChainHook*> g_swapChainHooks;

  void log(const char* format, ...);

  using D3D12RuntimePresentFunction = void (__cdecl*)(void*, UINT, UINT, HRESULT);
  using D3D12RuntimePresentBeginFunction = void (__cdecl*)(void*, UINT, ID3D12Resource*, const D3D12_RESOURCE_DESC*);

  D3D12RuntimePresentFunction resolveD3D12RuntimePresent() {
    static std::once_flag once;
    static D3D12RuntimePresentFunction function = nullptr;
    std::call_once(once, [] {
      HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
      if (d3d12)
        function = reinterpret_cast<D3D12RuntimePresentFunction>(
          GetProcAddress(d3d12, "remixD3D12RuntimePresent"));
      log("D3D12 native frontend Present callback %s", function ? "available" : "unavailable");
    });
    return function;
  }

  D3D12RuntimePresentBeginFunction resolveD3D12RuntimePresentBegin() {
    static std::once_flag once;
    static D3D12RuntimePresentBeginFunction function = nullptr;
    std::call_once(once, [] {
      HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
      if (d3d12)
        function = reinterpret_cast<D3D12RuntimePresentBeginFunction>(
          GetProcAddress(d3d12, "remixD3D12RuntimePresentBegin"));
      log("D3D12 native frontend Present-begin callback %s", function ? "available" : "unavailable");
    });
    return function;
  }

  constexpr size_t kFactory6VtableEntries = 31;
  constexpr size_t kCreateSwapChainForHwndIndex = 15;
  constexpr size_t kSwapChainBaseVtableEntries = 18;
  constexpr size_t kSwapChain3VtableEntries = 40;
  constexpr size_t kSwapChainPresentIndex = 8;
  constexpr size_t kSwapChainGetBufferIndex = 9;
  constexpr size_t kSwapChainGetCurrentBackBufferIndex = 36;
  constexpr size_t kSwapChainResizeBuffersIndex = 13;

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
      return L"remix-dxgi-proxy.log";

    wchar_t* separator = wcsrchr(modulePath, L'\\');
    if (separator)
      *(separator + 1) = L'\0';
    else
      modulePath[0] = L'\0';

    return std::wstring(modulePath) + L"remix-dxgi-proxy.log";
  }

  void log(const char* format, ...) {
    char message[1024] = {};
    va_list args;
    va_start(args, format);
    _vsnprintf_s(message, sizeof(message), _TRUNCATE, format, args);
    va_end(args);

    OutputDebugStringA(message);
    OutputDebugStringA("\n");

    std::lock_guard lock(g_logMutex);
    HANDLE file = CreateFileW(
      getLogPath().c_str(),
      FILE_APPEND_DATA,
      FILE_SHARE_READ | FILE_SHARE_WRITE,
      nullptr,
      OPEN_ALWAYS,
      FILE_ATTRIBUTE_NORMAL,
      nullptr);
    if (file == INVALID_HANDLE_VALUE)
      return;

    DWORD written = 0;
    WriteFile(file, message, static_cast<DWORD>(strlen(message)), &written, nullptr);
    WriteFile(file, "\r\n", 2, &written, nullptr);
    CloseHandle(file);
  }

  HRESULT STDMETHODCALLTYPE resizeBuffersHook(
      IDXGISwapChain* swapChain,
      UINT bufferCount,
      UINT width,
      UINT height,
      DXGI_FORMAT format,
      UINT flags) {
    SwapChainHook* hook = nullptr;
    {
      std::lock_guard lock(g_factoryHookMutex);
      auto entry = g_swapChainHooks.find(swapChain);
      if (entry != g_swapChainHooks.end())
        hook = entry->second;
    }

    if (!hook || !hook->originalResizeBuffers) {
      log("IDXGISwapChain::ResizeBuffers missing original (swapChain=%p)", swapChain);
      return E_UNEXPECTED;
    }

    log("IDXGISwapChain::ResizeBuffers(swapChain=%p, bufferCount=%u, width=%u, height=%u, format=%u, flags=0x%x)",
      swapChain,
      bufferCount,
      width,
      height,
      static_cast<unsigned>(format),
      flags);
    HRESULT result = hook->originalResizeBuffers(
      swapChain, bufferCount, width, height, format, flags);
    log("IDXGISwapChain::ResizeBuffers(result=0x%08lx)",
      static_cast<unsigned long>(result));
    return result;
  }

  UINT STDMETHODCALLTYPE getCurrentBackBufferIndexHook(IDXGISwapChain3* swapChain) {
    SwapChainHook* hook = nullptr;
    {
      std::lock_guard lock(g_factoryHookMutex);
      auto entry = g_swapChainHooks.find(swapChain);
      if (entry != g_swapChainHooks.end())
        hook = entry->second;
    }

    if (!hook || !hook->originalGetCurrentBackBufferIndex) {
      log("IDXGISwapChain3::GetCurrentBackBufferIndex missing original (swapChain=%p)", swapChain);
      return 0;
    }

    UINT index = hook->originalGetCurrentBackBufferIndex(swapChain);
    uint64_t callCount = ++hook->backBufferIndexCount;
    if (callCount <= 3 || (callCount % 60) == 0)
      log("IDXGISwapChain3::GetCurrentBackBufferIndex(count=%llu, index=%u)",
        static_cast<unsigned long long>(callCount), index);
    return index;
  }

  HRESULT STDMETHODCALLTYPE getBufferHook(
      IDXGISwapChain* swapChain,
      UINT buffer,
      REFIID riid,
      void** surface) {
    SwapChainHook* hook = nullptr;
    {
      std::lock_guard lock(g_factoryHookMutex);
      auto entry = g_swapChainHooks.find(swapChain);
      if (entry != g_swapChainHooks.end())
        hook = entry->second;
    }

    if (!hook || !hook->originalGetBuffer) {
      log("IDXGISwapChain::GetBuffer missing original (swapChain=%p)", swapChain);
      return E_UNEXPECTED;
    }

    HRESULT result = hook->originalGetBuffer(swapChain, buffer, riid, surface);
    log("IDXGISwapChain::GetBuffer(buffer=%u, riid=%s, result=0x%08lx, resource=%p)",
      buffer,
      formatIid(riid),
      static_cast<unsigned long>(result),
      surface ? *surface : nullptr);
    if (SUCCEEDED(result) && surface && *surface && riid == __uuidof(ID3D12Resource)) {
      auto* resource = static_cast<ID3D12Resource*>(*surface);
      D3D12_RESOURCE_DESC description = resource->GetDesc();
      log("IDXGISwapChain::GetBuffer desc(dimension=%u, width=%llu, height=%u, depthOrArray=%u, mips=%u, format=%u, sampleCount=%u, sampleQuality=%u, layout=%u, flags=0x%x)",
        static_cast<unsigned>(description.Dimension),
        static_cast<unsigned long long>(description.Width),
        description.Height,
        description.DepthOrArraySize,
        description.MipLevels,
        static_cast<unsigned>(description.Format),
        description.SampleDesc.Count,
        description.SampleDesc.Quality,
        static_cast<unsigned>(description.Layout),
        static_cast<unsigned>(description.Flags));
    }
    return result;
  }

  HRESULT STDMETHODCALLTYPE presentHook(
      IDXGISwapChain* swapChain,
      UINT syncInterval,
      UINT flags) {
    SwapChainHook* hook = nullptr;
    {
      std::lock_guard lock(g_factoryHookMutex);
      auto entry = g_swapChainHooks.find(swapChain);
      if (entry != g_swapChainHooks.end())
        hook = entry->second;
    }

    if (!hook || !hook->originalPresent) {
      log("IDXGISwapChain::Present missing original (swapChain=%p)", swapChain);
      return E_UNEXPECTED;
    }

    uint64_t presentCount = ++hook->presentCount;
    bool logPresent = presentCount <= 3 || (presentCount % 60) == 0;
    if (logPresent)
      log("IDXGISwapChain::Present begin(swapChain=%p, count=%llu, sync=%u, flags=0x%x)",
        swapChain,
        static_cast<unsigned long long>(presentCount),
        syncInterval,
        flags);

    UINT backBufferIndex = 0;
    ID3D12Resource* backBuffer = nullptr;
    D3D12_RESOURCE_DESC backBufferDescription {};
    if (hook->originalGetCurrentBackBufferIndex)
      backBufferIndex = hook->originalGetCurrentBackBufferIndex(
        reinterpret_cast<IDXGISwapChain3*>(swapChain));
    if (hook->originalGetBuffer
     && SUCCEEDED(hook->originalGetBuffer(
       swapChain, backBufferIndex, __uuidof(ID3D12Resource),
       reinterpret_cast<void**>(&backBuffer)))
     && backBuffer) {
      backBufferDescription = backBuffer->GetDesc();
    }
    if (auto callback = resolveD3D12RuntimePresentBegin())
      callback(swapChain, backBufferIndex, backBuffer, backBuffer ? &backBufferDescription : nullptr);
    if (backBuffer)
      backBuffer->Release();

    HRESULT result = hook->originalPresent(swapChain, syncInterval, flags);
    if (auto callback = resolveD3D12RuntimePresent())
      callback(swapChain, syncInterval, flags, result);
    if (logPresent)
      log("IDXGISwapChain::Present end(count=%llu, result=0x%08lx)",
        static_cast<unsigned long long>(presentCount),
        static_cast<unsigned long>(result));
    return result;
  }

  void installSwapChainHook(IDXGISwapChain1* swapChain) {
    if (!swapChain)
      return;

    std::lock_guard lock(g_factoryHookMutex);
    if (g_swapChainHooks.find(swapChain) != g_swapChainHooks.end())
      return;

    void** originalVtable = *reinterpret_cast<void***>(swapChain);
    if (!originalVtable)
      return;

    auto queryInterface = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*) (
      IDXGISwapChain1*, REFIID, void**)>(originalVtable[0]);
    void* swapChain3 = nullptr;
    HRESULT queryResult = queryInterface
      ? queryInterface(swapChain, __uuidof(IDXGISwapChain3), &swapChain3)
      : E_NOINTERFACE;
    bool supportsSwapChain3 = SUCCEEDED(queryResult) && swapChain3;
    if (swapChain3)
      reinterpret_cast<IUnknown*>(swapChain3)->Release();

    size_t vtableEntries = supportsSwapChain3
      ? kSwapChain3VtableEntries
      : kSwapChainBaseVtableEntries;
    void** clonedVtable = static_cast<void**>(VirtualAlloc(
      nullptr,
      vtableEntries * sizeof(void*),
      MEM_RESERVE | MEM_COMMIT,
      PAGE_READWRITE));
    if (!clonedVtable) {
      log("IDXGISwapChain hook allocation failed: %lu", GetLastError());
      return;
    }

    memcpy(clonedVtable, originalVtable, vtableEntries * sizeof(void*));
    auto originalPresent =
      reinterpret_cast<SwapChainPresentFunction>(clonedVtable[kSwapChainPresentIndex]);
    auto originalGetBuffer =
      reinterpret_cast<SwapChainGetBufferFunction>(clonedVtable[kSwapChainGetBufferIndex]);
    auto originalGetCurrentBackBufferIndex = supportsSwapChain3
      ? reinterpret_cast<SwapChainGetCurrentBackBufferIndexFunction>(clonedVtable[kSwapChainGetCurrentBackBufferIndex])
      : nullptr;
    auto originalResizeBuffers =
      reinterpret_cast<SwapChainResizeBuffersFunction>(clonedVtable[kSwapChainResizeBuffersIndex]);
    clonedVtable[kSwapChainPresentIndex] = reinterpret_cast<void*>(&presentHook);
    clonedVtable[kSwapChainGetBufferIndex] = reinterpret_cast<void*>(&getBufferHook);
    clonedVtable[kSwapChainResizeBuffersIndex] = reinterpret_cast<void*>(&resizeBuffersHook);
    if (supportsSwapChain3)
      clonedVtable[kSwapChainGetCurrentBackBufferIndex] = reinterpret_cast<void*>(&getCurrentBackBufferIndexHook);

    DWORD oldVtableProtection = 0;
    if (!VirtualProtect(
      clonedVtable,
      vtableEntries * sizeof(void*),
      PAGE_READONLY,
      &oldVtableProtection)) {
      log("IDXGISwapChain hook vtable protection failed: %lu", GetLastError());
      VirtualFree(clonedVtable, 0, MEM_RELEASE);
      return;
    }

    auto* hook = new (std::nothrow) SwapChainHook {
      originalVtable,
      clonedVtable,
      vtableEntries,
      originalPresent,
      originalGetBuffer,
      originalGetCurrentBackBufferIndex,
      originalResizeBuffers};
    if (!hook) {
      log("IDXGISwapChain hook metadata allocation failed");
      VirtualFree(clonedVtable, 0, MEM_RELEASE);
      return;
    }

    DWORD oldObjectProtection = 0;
    if (!VirtualProtect(swapChain, sizeof(void*), PAGE_READWRITE, &oldObjectProtection)) {
      log("IDXGISwapChain object protection failed: %lu", GetLastError());
      delete hook;
      VirtualFree(clonedVtable, 0, MEM_RELEASE);
      return;
    }
    *reinterpret_cast<void***>(swapChain) = clonedVtable;
    DWORD ignoredProtection = 0;
    VirtualProtect(swapChain, sizeof(void*), oldObjectProtection, &ignoredProtection);

    g_swapChainHooks.emplace(swapChain, hook);
    log("Installed IDXGISwapChain vtable hook at %p (interface=%s, Present=%p, GetBuffer=%p, GetCurrentBackBufferIndex=%p, ResizeBuffers=%p)",
      swapChain,
      supportsSwapChain3 ? "IDXGISwapChain3" : "IDXGISwapChain",
      reinterpret_cast<void*>(originalPresent),
      reinterpret_cast<void*>(originalGetBuffer),
      reinterpret_cast<void*>(originalGetCurrentBackBufferIndex),
      reinterpret_cast<void*>(originalResizeBuffers));
  }

  HRESULT STDMETHODCALLTYPE createSwapChainForHwndHook(
      IDXGIFactory2* factory,
      IUnknown* device,
      HWND window,
      const DXGI_SWAP_CHAIN_DESC1* description,
      const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreenDescription,
      IDXGIOutput* restrictToOutput,
      IDXGISwapChain1** swapChain) {
    FactoryHook* hook = nullptr;
    {
      std::lock_guard lock(g_factoryHookMutex);
      auto entry = g_factoryHooks.find(factory);
      if (entry != g_factoryHooks.end())
        hook = entry->second;
    }

    if (!hook || !hook->originalCreateSwapChainForHwnd) {
      log("IDXGIFactory6::CreateSwapChainForHwnd missing original (factory=%p)", factory);
      return E_UNEXPECTED;
    }

    log("IDXGIFactory6::CreateSwapChainForHwnd(factory=%p, device=%p, hwnd=%p, desc=%p, fullscreen=%p, output=%p)",
      factory, device, window, description, fullscreenDescription, restrictToOutput);
    if (description) {
      log("IDXGIFactory6::CreateSwapChainForHwnd desc(width=%u, height=%u, format=%u, bufferCount=%u, sampleCount=%u, sampleQuality=%u, swapEffect=%u, flags=0x%x)",
        description->Width,
        description->Height,
        static_cast<unsigned>(description->Format),
        description->BufferCount,
        description->SampleDesc.Count,
        description->SampleDesc.Quality,
        static_cast<unsigned>(description->SwapEffect),
        description->Flags);
    }

    HRESULT result = hook->originalCreateSwapChainForHwnd(
      factory,
      device,
      window,
      description,
      fullscreenDescription,
      restrictToOutput,
      swapChain);

    log("IDXGIFactory6::CreateSwapChainForHwnd(result=0x%08lx, swapChain=%p)",
      static_cast<unsigned long>(result),
      swapChain ? *swapChain : nullptr);
    if (SUCCEEDED(result) && swapChain && *swapChain)
      installSwapChainHook(*swapChain);
    return result;
  }

  void installFactory6Hook(void* factory) {
    if (!factory)
      return;

    std::lock_guard lock(g_factoryHookMutex);
    if (g_factoryHooks.find(factory) != g_factoryHooks.end())
      return;

    void** originalVtable = *reinterpret_cast<void***>(factory);
    if (!originalVtable)
      return;

    void** clonedVtable = static_cast<void**>(VirtualAlloc(
      nullptr,
      kFactory6VtableEntries * sizeof(void*),
      MEM_RESERVE | MEM_COMMIT,
      PAGE_READWRITE));
    if (!clonedVtable) {
      log("IDXGIFactory6 hook allocation failed: %lu", GetLastError());
      return;
    }

    memcpy(clonedVtable, originalVtable, kFactory6VtableEntries * sizeof(void*));
    auto originalCreateSwapChainForHwnd =
      reinterpret_cast<CreateSwapChainForHwndFunction>(clonedVtable[kCreateSwapChainForHwndIndex]);
    clonedVtable[kCreateSwapChainForHwndIndex] = reinterpret_cast<void*>(&createSwapChainForHwndHook);

    DWORD oldVtableProtection = 0;
    if (!VirtualProtect(
      clonedVtable,
      kFactory6VtableEntries * sizeof(void*),
      PAGE_READONLY,
      &oldVtableProtection)) {
      log("IDXGIFactory6 hook vtable protection failed: %lu", GetLastError());
      VirtualFree(clonedVtable, 0, MEM_RELEASE);
      return;
    }

    DWORD oldObjectProtection = 0;
    if (!VirtualProtect(factory, sizeof(void*), PAGE_READWRITE, &oldObjectProtection)) {
      log("IDXGIFactory6 object protection failed: %lu", GetLastError());
      VirtualFree(clonedVtable, 0, MEM_RELEASE);
      return;
    }
    *reinterpret_cast<void***>(factory) = clonedVtable;
    DWORD ignoredProtection = 0;
    VirtualProtect(factory, sizeof(void*), oldObjectProtection, &ignoredProtection);

    auto* hook = new (std::nothrow) FactoryHook {
      originalVtable,
      clonedVtable,
      originalCreateSwapChainForHwnd};
    if (!hook) {
      log("IDXGIFactory6 hook metadata allocation failed");
      return;
    }
    g_factoryHooks.emplace(factory, hook);
    log("Installed IDXGIFactory6 vtable hook at %p (CreateSwapChainForHwnd=%p)",
      factory, reinterpret_cast<void*>(originalCreateSwapChainForHwnd));
  }

  HMODULE loadRealDxgi() {
    std::call_once(g_realDxgiOnce, [] {
      wchar_t systemPath[MAX_PATH] = {};
      UINT length = GetSystemDirectoryW(systemPath, ARRAYSIZE(systemPath));
      if (!length || length >= ARRAYSIZE(systemPath) - 10) {
        log("GetSystemDirectoryW failed: %lu", GetLastError());
        return;
      }

      wcscat_s(systemPath, L"\\dxgi.dll");
      g_realDxgi = LoadLibraryExW(systemPath, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
      if (!g_realDxgi)
        log("LoadLibraryExW(%ls) failed: %lu", systemPath, GetLastError());
      else
        log("Loaded system DXGI: %ls", systemPath);
    });

    return g_realDxgi;
  }

  template<typename T>
  T resolve(const char* name) {
    HMODULE module = loadRealDxgi();
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

extern "C" {

  HRESULT WINAPI CreateDXGIFactory(REFIID riid, void** ppFactory) {
    using Function = HRESULT (WINAPI*)(REFIID, void**);
    Function function = resolve<Function>("CreateDXGIFactory");
    HRESULT result = function ? function(riid, ppFactory) : E_FAIL;
    log("CreateDXGIFactory(riid=%s, output=%p, result=0x%08lx, object=%p)",
      formatIid(riid), ppFactory, static_cast<unsigned long>(result),
      ppFactory ? *ppFactory : nullptr);
    return result;
  }

  HRESULT WINAPI CreateDXGIFactory1(REFIID riid, void** ppFactory) {
    using Function = HRESULT (WINAPI*)(REFIID, void**);
    Function function = resolve<Function>("CreateDXGIFactory1");
    HRESULT result = function ? function(riid, ppFactory) : E_FAIL;
    log("CreateDXGIFactory1(riid=%s, output=%p, result=0x%08lx, object=%p)",
      formatIid(riid), ppFactory, static_cast<unsigned long>(result),
      ppFactory ? *ppFactory : nullptr);
    return result;
  }

  HRESULT WINAPI CreateDXGIFactory2(UINT flags, REFIID riid, void** ppFactory) {
    using Function = HRESULT (WINAPI*)(UINT, REFIID, void**);
    Function function = resolve<Function>("CreateDXGIFactory2");
    HRESULT result = function ? function(flags, riid, ppFactory) : E_FAIL;
    log("CreateDXGIFactory2(flags=0x%x, riid=%s, output=%p, result=0x%08lx, object=%p)",
      flags, formatIid(riid), ppFactory, static_cast<unsigned long>(result),
      ppFactory ? *ppFactory : nullptr);
    if (SUCCEEDED(result) && ppFactory && *ppFactory && riid == __uuidof(IDXGIFactory6))
      installFactory6Hook(*ppFactory);
    return result;
  }

  HRESULT WINAPI DXGIDeclareAdapterRemovalSupport() {
    using Function = HRESULT (WINAPI*)();
    Function function = resolve<Function>("DXGIDeclareAdapterRemovalSupport");
    HRESULT result = function ? function() : E_FAIL;
    log("DXGIDeclareAdapterRemovalSupport(result=0x%08lx)",
      static_cast<unsigned long>(result));
    return result;
  }

  HRESULT WINAPI DXGIGetDebugInterface1(UINT flags, REFIID riid, void** ppDebug) {
    using Function = HRESULT (WINAPI*)(UINT, REFIID, void**);
    Function function = resolve<Function>("DXGIGetDebugInterface1");
    HRESULT result = function ? function(flags, riid, ppDebug) : E_FAIL;
    log("DXGIGetDebugInterface1(flags=0x%x, riid=%s, output=%p, result=0x%08lx, object=%p)",
      flags, formatIid(riid), ppDebug, static_cast<unsigned long>(result),
      ppDebug ? *ppDebug : nullptr);
    return result;
  }

}
