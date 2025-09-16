build
```bash
winget install Ninja-build.Ninja

winget install -e --id Microsoft.VisualStudio.2022.BuildTools
```

in the `x64 Native Tools Command Prompt for VS 2022`

```bash
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```bash
ninja -C build -j %NUMBER_OF_PROCESSORS%
```
