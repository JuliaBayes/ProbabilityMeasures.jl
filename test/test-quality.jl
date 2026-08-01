using ProbabilityMeasures
using Aqua
using Test

# Formatting and explicit-import hygiene are enforced by pre-commit
# (.pre-commit-config.yaml), so they are deliberately not duplicated here.
@testset "Aqua" begin
    Aqua.test_all(ProbabilityMeasures)
end
