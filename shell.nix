{ sources ? import ./nix/sources.nix
, pkgs ? import sources.nixpkgs {}
}:


let
  py = pkgs.python38;
in
pkgs.mkShell {
  buildInputs = [
    pkgs.protobuf

    py
    py.pkgs.python-language-server
    py.pkgs.jedi
    py.pkgs.poetry
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}
  '';
}
