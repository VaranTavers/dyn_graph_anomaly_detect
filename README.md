# disszertacio


## Instructions to run

You need Julia installed, to run this application. (https://julialang.org/)

Aside from Julia you need Pluto.jl to be installed, but that can be done using the Julia shell:

```
julia> using Pkg
julia> Pkg.add("Pluto")
```

After you have it installed, you should start a Pluto server:

```
julia> using Pluto
julia> Pluto.run()
```

This opens a new tab in the browser, in which you can open the "implementation.jl" file.

The notebook will run automatically upon opening, and will refresh any affected cells upon editing, and rerunning a cell.
