@require CuArrays begin
    using CUDAdrv
    function device!(n = myid())
        ngpu = length(CUDAdrv.devices())
        CUDAdrv.CuContext(CUDAdrv.CuDevice(mod1(n, ngpu) - 1))
    end
    device!()
end