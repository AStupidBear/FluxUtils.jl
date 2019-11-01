# Extensions to Flux.jl

## Installation

```julia
using Pkg
pkg"add FluxUtils"
```

## Usage

### Sklearn Interface

```jl
fit!(est::Estimator, x, y, w = nothing; kws...)
predict!(ŷ, est::Estimator, x)
```

### Distributed Trading

```jl

```

### Utility Functions

```
using Flux, FluxUtils
```

```
notrack(m)
```

tensor contraction

```
A *ᶜ B
```

batch matrix multiplication

```
bmm(A, B)
```

```
pσ
```

```
net2vec(m)
vec2net!(m, x)
```

```
namedparams
states
loadstates!(m, xs)
```

## Performance Tips for Inference of RNN

### Broadcasting

```
```

### 