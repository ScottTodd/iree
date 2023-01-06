// This program reads a value from a push constant then splats that value into
// an output buffer.

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>

hal.executable.source public @executable {
  hal.executable.export public @splat_push_constant layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @splat_push_constant() {
      // Input push constant.
      %input = hal.interface.constant.load[0] : i32

      // Output buffer.
      %c0 = arith.constant 0 : index
      %out = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<4xi32>>

      // Splat the input value into a tensor.
      %tensor = tensor.empty() : tensor<4xi32>
      %splat_tensor = linalg.fill ins(%input : i32) outs(%tensor : tensor<4xi32>) -> tensor<4xi32>

      // Store into the output buffer.
      flow.dispatch.tensor.store %splat_tensor, %out, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>

      return
    }
  }
}
