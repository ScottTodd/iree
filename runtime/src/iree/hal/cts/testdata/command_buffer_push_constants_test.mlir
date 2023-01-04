// TODO(scotttodd): update this
// This program reads a value from an input buffer at the offset specified by
// a push constant then stores that value into an output buffer.

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>

hal.executable.source public @executable {
  hal.executable.export public @splat_push_constant layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @splat_push_constant() {
      %c0 = arith.constant 0 : index
      // %c1 = arith.constant 1 : index
      // %c2 = arith.constant 2 : index
      // %c3 = arith.constant 3 : index
      // %c4 = arith.constant 4 : index

      // Read the input push constants.
      %input_0 = hal.interface.constant.load[0] : i32
      // %input_1 = hal.interface.constant.load[1] : i32
      // %input_2 = hal.interface.constant.load[2] : i32
      // %input_3 = hal.interface.constant.load[3] : i32

      // %input_splat = flow.tensor.splat %input_0 : tensor<4xi32>

      %tensor = tensor.empty() : tensor<4xi32>
      %input_fill = linalg.fill ins(%input_0 : i32) outs(%tensor : tensor<4xi32>) -> tensor<4xi32>

      // %input_empty = flow.tensor.empty : tensor<4xi32>
      // %input_0_stored = flow.tensor.store %input_0,    %input_empty[%c0] : tensor<4xi32>
      // %input_1_stored = flow.tensor.store %input_1, %input_0_stored[%c1] : tensor<4xi32>
      // %input_2_stored = flow.tensor.store %input_2, %input_1_stored[%c2] : tensor<4xi32>
      // %input_3_stored = flow.tensor.store %input_3, %input_2_stored[%c3] : tensor<4xi32>

      // %push_constants = tensor.from_elements %input_0, %input_1, %input_2, %input_3 : tensor<4xi32>

      // Write into the output buffer.
      %out = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<4xi32>>

      // flow.dispatch.tensor.store %input_3_stored, %out, offsets = [0], sizes = [1], strides = [0] : tensor<4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>
      // flow.dispatch.tensor.store %input_fill, %out, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>
      flow.dispatch.tensor.store %tensor, %out, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>

      // // Load from the input buffer at the index in the push constant data.
      // %push_constant_index = arith.index_castui %push_constant_i32 : i32 to index
      // %loaded_value = flow.dispatch.tensor.load %in, offsets = [%push_constant_index], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<f32>

      // // Store into the output buffer.
      // flow.dispatch.tensor.store %loaded_value, %out, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>

      return
    }
  }
}
