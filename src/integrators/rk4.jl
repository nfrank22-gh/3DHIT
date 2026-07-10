"""
    RK4

Classic 4th-order Runge–Kutta with preallocated stage buffers.

Construct from an existing state so the buffers match its size, eltype, and
backend:

    scheme = RK4(û)

TODO later: low-storage variants (e.g. 2N-storage SSPRK3 / Carpenter–Kennedy)
as sibling structs implementing the same `step!` interface.
"""
struct RK4{A} <: AbstractScheme
    k1::A
    k2::A
    k3::A
    k4::A
    tmp::A   # holds û + c*dt*kᵢ between stages
end

RK4(û::AbstractArray) = RK4(ntuple(_ -> similar(û), 5)...)

function step!(û, r::AbstractRHS, s::RK4, dt, t)
    rhs!(s.k1, û, r, t)
    @. s.tmp = û + (dt / 2) * s.k1
    rhs!(s.k2, s.tmp, r, t + dt / 2)
    @. s.tmp = û + (dt / 2) * s.k2
    rhs!(s.k3, s.tmp, r, t + dt / 2)
    @. s.tmp = û + dt * s.k3
    rhs!(s.k4, s.tmp, r, t + dt)
    @. û += (dt / 6) * (s.k1 + 2 * s.k2 + 2 * s.k3 + s.k4)
    return û
end
