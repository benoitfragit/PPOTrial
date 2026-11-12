// Link regression: the existing BlueDefense backend uses the default namespace.
#include <rl_tools/operations/cpu_mux.h>

bool OtherRltoolsBackendCompiles()
{
    auto engine=rl_tools::random::default_engine(rl_tools::devices::random::CPU{},17);
    return engine()!=0 && !rl_tools::devices::cpu::sanitize_file_name("flight-control").empty();
}
