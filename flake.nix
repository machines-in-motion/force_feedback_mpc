{
  description = "Optimal control tools to achieve force feedback in MPC.";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    flake-parts.follows = "gepetto/flake-parts";
    nixpkgs.follows = "gepetto/nixpkgs";
    nix-ros-overlay.follows = "gepetto/nix-ros-overlay";
    systems.follows = "gepetto/systems";
    treefmt-nix.follows = "gepetto/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, self, ... }:
      {
        systems = import inputs.systems;
        imports = [
          inputs.gepetto.flakeModule
          { gepetto-pkgs.overlays = [ self.overlays.default ]; }
        ];
        flake.overlays.default = _final: prev: {
          force-feedback-mpc = prev.force-feedback-mpc.overrideAttrs {
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
        };
        perSystem =
          { pkgs, self', ... }:
          {
            packages = {
              default = self'.packages.force-feedback-mpc;
              force-feedback-mpc = pkgs.python3Packages.force-feedback-mpc.override { standalone = false; };
            };
          };
        }
      }
    );
}
