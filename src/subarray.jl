Base.vcat(a::Tracker.TrackedArray, b::SubArray) = Tracker.track(vcat, a, b)
Base.vcat(a::SubArray, b::Tracker.TrackedArray) = Tracker.track(vcat, a, b)
Base.hcat(a::Tracker.TrackedArray, b::SubArray) = Tracker.track(hcat, a, b)
Base.hcat(a::SubArray, b::Tracker.TrackedArray) = Tracker.track(hcat, a, b)
