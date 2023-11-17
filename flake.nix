{
  description = "onnion";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication overrides;
        pkgs = import nixpkgs { inherit system; };
        py = pkgs.python39;
        customOverrides = self: super: {
          platformdirs = super.platformdirs.overridePythonAttrs (
            old: {
              postPatch = "";
            }
          );

          onnxruntime = super.onnxruntime.overridePythonAttrs (
            old: {
              nativeBuildInputs = [ ];
              postFixup =
                let rPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
                in
                ''
                  rrPath=${rPath}
                  find $out/lib -name '*.so' -exec patchelf --add-rpath "$rrPath" {} \;
                '';
            }
          );
        };
      in
      {
        packages.onnion = mkPoetryApplication {
          projectDir = ./compiler;
          python = py;
          overrides = overrides.withDefaults (
            customOverrides
          );
          preferWheels = true;
        };

        packages.dockerimage = pkgs.dockerTools.buildImage {
          name = "idein/onnion";
          tag = "latest";
          created = "now";
          copyToRoot = [ self.packages.${system}.onnion ];
          config = {
            Entrypoint = [ "/bin/onnion" ];
            WorkingDir = "/work";
          };
        };


        defaultPackage = self.packages.${system}.onnion;

        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.protobuf
            pkgs.zlib

            py
            pkgs.poetry
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ]}
          '';
        };
      }
    );
}
