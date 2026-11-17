# Optimal classification trees with Interpretable AI (IAI), called from interpretable_model_V1.ipynb.
#
# IAI is preloaded in the IAI system image: do NOT add `using IAI`. Run with the IAI channel:
#   julia +1.11.6/IAI/3.2.2 helper_train_iai.jl <dir> [max_depth=5] [minbucket=20]
# (plain `julia` uses the default channel, which has no IAI, so `IAI.` calls would fail)
#
# Input  (in <dir>): train.csv, val.csv, test.csv — feature columns plus a 0/1 target column `y`.
# Output (in <dir>): p_val.csv, p_test.csv — one column per depth (`depth_1` …), the predicted
#                    probability that y = 1; tree_depth_<d>.html (picture), .txt (text) and .json (reloadable model).
# Depth is NOT chosen here: the notebook picks it on lockbox (val) AUC, like the other models.
# For each depth, IAI tunes the complexity penalty `cp` on an internal split of train only.

using CSV, DataFrames

dir       = ARGS[1]
max_depth = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
minbucket = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20

load(name) = CSV.read(joinpath(dir, "$name.csv"), DataFrame)
train, val, test = load("train"), load("val"), load("test")
features(d) = select(d, Not(:y))

p_val, p_test = DataFrame(), DataFrame()
for depth in 1:max_depth
    t = @elapsed begin
        grid = IAI.GridSearch(
            IAI.OptimalTreeClassifier(random_seed=1, minbucket=minbucket, criterion=:gini),
            max_depth=depth,
        )
        IAI.fit!(grid, features(train), train.y, validation_criterion=:auc)
        lnr = IAI.get_learner(grid)
    end
    prob1(d) = IAI.predict_proba(lnr, features(d))[!, Symbol(1)]   # probability of y = 1
    p_val[!, "depth_$depth"]  = prob1(val)
    p_test[!, "depth_$depth"] = prob1(test)
    IAI.write_html(joinpath(dir, "tree_depth_$depth.html"), lnr)
    IAI.write_json(joinpath(dir, "tree_depth_$depth.json"), lnr)
    write(joinpath(dir, "tree_depth_$depth.txt"), sprint(show, MIME("text/plain"), lnr))
    println("depth $depth: $(IAI.get_num_nodes(lnr)) nodes, $(round(t, digits=1)) s")
end

CSV.write(joinpath(dir, "p_val.csv"), p_val)
CSV.write(joinpath(dir, "p_test.csv"), p_test)
