{
  description = "onnion";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-21.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        py = pkgs.python38;
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

        defaultPackage = self.packages.${system}.onnion;

        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.protobuf

            py
            py.pkgs.jedi-language-server
            py.pkgs.poetry
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}
          '';
        };
      }
    );
}
