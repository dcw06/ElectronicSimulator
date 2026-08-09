# GPU-Accelerated 2D Dirichlet Solver

Monte Carlo random-walk solver for the 2D Laplace equation. The CUDA mapping follows
the supplied framework: one block per interior grid point, multiple trajectories per
thread, Philox RNG, and a shared-memory reduction producing potential and standard error.

The initial boundary condition has exact solution `u(x,y)=x`. Electric field is computed
as `E=-grad(V)`, so the exact interior field is `Ex=-1, Ey=0, |E|=1`.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

To build without CUDA (useful on a machine without an NVIDIA GPU):

```powershell
cmake -S . -B build-cpu -DMC_ENABLE_CUDA=OFF
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

## Run

```powershell
build\Release\mc_field_solver.exe --nx 64 --ny 64 --walks 4096 --backend cpu --output output
build\Release\mc_field_solver.exe --nx 256 --ny 256 --walks 4096 --backend cuda --output output
```

Output files are `potential.csv`, `stderr.csv`, `Ex.csv`, `Ey.csv`, and `E.csv`.

## MUSA on academy.mthreads

The MUSA build targets the MTT S4000 (`mp_22`) and uses the Toolkit installed at
`/usr/local/musa` by default.

```bash
export MUSA_HOME=/usr/local/musa
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:${LD_LIBRARY_PATH:-}"

./build_musa.sh
./build-musa/mc_field_solver --nx 64 --ny 64 --walks 4096 \
  --backend musa --output output-musa
```

Override `MUSA_ARCH` if the Academy instance provides a different MT GPU.

## Visualize

```powershell
python -m pip install numpy matplotlib
python python\visualize.py output --save output\fields.png
```

The CUDA executable can be compiled on this computer, but CUDA execution requires an
NVIDIA GPU and compatible driver. The current Intel-only system can run the CPU backend.

## Local web interface

```powershell
.\.venv\Scripts\python.exe web\app.py
```

Open <http://127.0.0.1:8000>. The API validates grid and sampling limits, invokes the
native solver, and returns potential, standard error, and electric-field matrices.

To use the MUSA executable in the web interface on Academy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Flask
MC_FIELD_SOLVER="$PWD/build-musa/mc_field_solver" MC_BACKEND=musa python web/app.py
```
