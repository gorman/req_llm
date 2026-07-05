# Local dev shell, not committed. Same toolchain as cabochon's `nix develop .#elixir`:
# nixpkgs pinned to cabochon's flake.lock rev so the store paths are already present.
let
  nixpkgs = import (fetchTarball {
    url = "https://github.com/nixos/nixpkgs/archive/569d578509928497eddc3fdbf94a799027050be4.tar.gz";
    sha256 = "sha256-ZGP04e+Q6WyQJGA9ZvI5CL6+heGQldbAG9U1T9NGvmU=";
  }) { };
in
nixpkgs.mkShell {
  packages = [
    nixpkgs.erlang_27
    nixpkgs.beam.packages.erlang_27.elixir_1_20
  ];
}
