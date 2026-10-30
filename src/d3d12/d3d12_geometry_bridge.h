#pragma once

#include <d3d12.h>

namespace remix {

  void captureIndexedGeometry(
      ID3D12GraphicsCommandList* commandList,
      const D3D12_VERTEX_BUFFER_VIEW* vertexBuffer,
      const D3D12_INDEX_BUFFER_VIEW* indexBuffer,
      D3D12_PRIMITIVE_TOPOLOGY topology,
      ID3D12PipelineState* pipelineState,
      UINT indexCount,
      UINT instanceCount,
      UINT startIndexLocation,
      INT baseVertexLocation,
      UINT startInstanceLocation);

}
