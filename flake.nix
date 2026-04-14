{
  description = "Optimal control tools to achieve force feedback in MPC.";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        extraDevPyPackages = [ "force-feedback-mpc" ];
        overrideAttrs.force-feedback-mpc = {
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./benchmarks
              ./bindings
              ./demos
              ./include
              ./python
              ./src
              ./tests
              ./CMakeLists.txt
              ./package.xml
            ];
          };
        };
      }
    );
}
