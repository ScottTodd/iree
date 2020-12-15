// TODO(scotttodd): convert into a test, or merge into existing tests

module  {
  hal.variable @_executable_linked_vmla mutable : !hal.executable attributes {sym_visibility = "private"}
  func @side_effects_test() {
    %c1 = constant 1 : index
    %dev = hal.ex.shared_device : !hal.device
    %cmd = hal.command_buffer.create %dev, "OneShot", "Transfer|Dispatch" : !hal.command_buffer
    hal.command_buffer.begin %cmd
    %0 = hal.variable.load @_executable_linked_vmla : !hal.executable
    hal.command_buffer.dispatch %cmd, %0, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
    %1 = hal.variable.load @_executable_linked_vmla : !hal.executable
    hal.command_buffer.dispatch %cmd, %1, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
    %2 = hal.variable.load @_executable_linked_vmla : !hal.executable
    hal.command_buffer.dispatch %cmd, %2, entry_point = 0, workgroup_xyz = [%c1, %c1, %c1]
    hal.command_buffer.end %cmd
    hal.ex.submit_and_wait %dev, %cmd
    return
  }
}
