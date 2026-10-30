#pragma once

#include <d3d12.h>

namespace remix {

  bool isDevice4OrLaterIid(REFIID iid);
  bool installDevice4Trace(ID3D12Device4* device);

}
