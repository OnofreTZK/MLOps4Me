let
  nixpkgs = import <nixpkgs> {};
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) {};
  pyEnv = mach-nix.mkPython rec {

    requirements = ''
        pip
        gunicorn
        mlflow
        pandas
        sklearn
        matplotlib
        keras
        tensorflow
    '';

  };
in
mach-nix.nixpkgs.mkShell {

  buildInputs = [
    pyEnv
  ] ;

  shellHook = ''
  '';
}
