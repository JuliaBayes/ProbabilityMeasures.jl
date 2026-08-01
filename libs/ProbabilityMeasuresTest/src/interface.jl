"""
    testpoint(d, i = 1)

A deterministic point in the support of `d`, used by the interface predicates.
"""
testpoint(d, i::Int=1) = rand(Xoshiro(i), d)

# Predicates receive only the object, so any test point has to be derived from it.
@interface MeasureInterface AbstractProbabilityMeasure (
    mandatory=(
        logdensityof=(
            "returns a real number" => d -> logdensityof(d, testpoint(d)) isa Real,
            "agrees with densityof" =>
                d -> exp(logdensityof(d, testpoint(d))) ≈ densityof(d, testpoint(d)),
            "is total at ±Inf and NaN (never throws)" =>
                d -> all(x -> (logdensityof(d, x); true), (Inf, -Inf, NaN)),
        ),
        rand=(
            "rand(rng, d) has type eltype(d)" => d -> rand(Xoshiro(1), d) isa eltype(d),
            "rand(rng, d, n) yields n draws" => d -> length(rand(Xoshiro(1), d, 3)) == 3,
            "rand is reproducible for a fixed seed" =>
                d -> rand(Xoshiro(1), d) == rand(Xoshiro(1), d),
        ),
        eltype="eltype(d) is concrete" => d -> isconcretetype(eltype(d)),
        support="support(d) is a Support" => d -> support(d) isa Support,
        insupport="insupport(d, x) is a Bool" => d -> insupport(d, testpoint(d)) isa Bool,
        params=(
            "params(d) is a NamedTuple" => d -> params(d) isa NamedTuple,
            "params(d) names the fields" => d -> keys(params(d)) === fieldnames(typeof(d)),
        ),
        reference="reference(d) is a ReferenceMeasure" =>
            d -> reference(d) isa ReferenceMeasure,
        checkparams="checkparams(d) is a Bool" => d -> checkparams(d) isa Bool,
        broadcast="d broadcasts as a scalar" =>
            d -> length(logdensityof.(d, [testpoint(d, 1), testpoint(d, 2)])) == 2,
    ),
    optional=(
        cdf=(
            "cdf lies in [0, 1]" => d -> 0 <= cdf(d, testpoint(d)) <= 1,
            "cdf and ccdf sum to one" =>
                d -> cdf(d, testpoint(d)) + ccdf(d, testpoint(d)) ≈ 1,
            "logcdf agrees with log(cdf)" =>
                d -> logcdf(d, testpoint(d)) ≈ log(cdf(d, testpoint(d))),
            "logccdf agrees with log(ccdf)" =>
                d -> logccdf(d, testpoint(d)) ≈ log(ccdf(d, testpoint(d))),
        ),
        quantile=(
            "quantile inverts cdf" => d -> quantile(d, cdf(d, testpoint(d))) ≈ testpoint(d),
            "quantile is monotone" => d -> quantile(d, 0.25) <= quantile(d, 0.75),
        ),
        mean=d -> mean(d) isa Real,
        var="var is non-negative" => d -> var(d) >= 0,
        std="std is the square root of var" => d -> std(d) ≈ sqrt(var(d)),
        median=d -> median(d) isa Real,
        mode=d -> mode(d) isa Real,
        entropy=d -> entropy(d) isa Real,
        skewness=d -> skewness(d) isa Real,
        kurtosis=d -> kurtosis(d) isa Real,
        mgf="mgf(d, 0) == 1" => d -> mgf(d, 0) ≈ 1,
        cf="cf(d, 0) == 1" => d -> cf(d, 0) ≈ 1,
    ),
) """
The interface of a normalized probability measure.

Mandatory components are those `AbstractProbabilityMeasure` documents as required
plus the invariants that make a measure usable inside a PPL: `logdensityof` must be
total, and the measure must broadcast as a scalar so that batched evaluation fuses
into a single kernel.

Optional components are the summaries and the distribution function -- meaningful
for a univariate measure, not for every measure.
"""
