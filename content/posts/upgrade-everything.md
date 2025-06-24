+++
date = '2025-06-24T15:25:00+02:00'
title = 'Upgrade Everything !'
+++

Keeping your system up to date is important because developers push security vulnerabilities to new versions of software.
The issue is that you may have a lot of different things to update: System packages managed by `apt`, `Vim` plugins or `VSCode` extensions, `snap` packages, `flatpaks`, `Nix` channels and `Home Manager` and derivations, `rustup` toolchain, `Docker` containers, `uv` tools, `Helix` grammars and many other things.

Luckily, there is an amazing and simple tool called ✨ [**TOPGRADE**](https://github.com/topgrade-rs/topgrade/) ✨

Install it the way you want (I used `cargo binstall topgrade`) and simply run `topgrade`. It will scan your system for all the tools it can upgrade, and it will upgrade everything. Sweet!


