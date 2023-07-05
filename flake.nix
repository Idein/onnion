{
  description = "onnion";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
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
        packages.onnion = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = ./compiler;
          python = py;
          overrides = pkgs.poetry2nix.overrides.withDefaults (
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
