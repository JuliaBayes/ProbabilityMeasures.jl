"""
    testpoint(d, i = 1)

A repeatable point in the support of `d`.
"""
testpoint(d, i::Int=1) = rand(Xoshiro(i), d)

# Interface checks receive only `d`, so derive their test points from it.
@interface MeasureInterface AbstractProbabilityMeasure (
    mandatory=(
        logdensityof=(
            "returns a real number" => d -> logdensityof(d, testpoint(d)) isa Real,
            "agrees with densityof" =>
                d -> exp(logdensityof(d, testpoint(d))) ≈ densityof(d, testpoint(d)),
            "is total at extreme arguments (never throws)" =>
                d -> all(x -> (logdensityof(d, x); true), _extremepoints(d)),
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
        entropy=d -> entropy(d) isa Real,
        # Multivariate summaries.
        meanvector=(
            "mean is a vector" => d -> mean(d) isa AbstractVector,
            "mean has the length of a draw" => d -> length(mean(d)) == length(testpoint(d)),
        ),
        cov=(
            "cov is square, with the length of a draw" =>
                d -> size(cov(d)) == (length(testpoint(d)), length(testpoint(d))),
            "cov is symmetric" => d -> cov(d) ≈ transpose(cov(d)),
            "var is the diagonal of cov" =>
                d -> var(d) ≈ [cov(d)[i, i] for i in axes(cov(d), 1)],
            "std is the elementwise square root of var" => d -> std(d) ≈ sqrt.(var(d)),
        ),
    ),
) """
Checks the required measure methods and any declared distribution functions or
summaries.
"""
