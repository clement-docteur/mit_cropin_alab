# Optimal classification trees with Interpretable AI (IAI), with spatially blocked CV.
# Called from interpretable_model_V2_v4_OCT_RF.ipynb.
#
# IAI is preloaded in the IAI system image: do NOT add `using IAI`. Run with the IAI channel:
#   julia +1.11.6/IAI/3.2.2 helper_train_iai_cv.jl <dir> [max_depth=5] [minbucket=20]
#
# Input  (in <dir>): train.csv — feature columns, a 0/1 target `y` and a CV fold id `fold`;
#                    test.csv  — the same feature columns plus `y`.
# Output (in <dir>): oof.csv — out-of-fold P(y = 1) for every training row, one column per depth
#                    (`depth_1` …); p_train.csv / p_test.csv — P(y = 1) on train (in-sample) and test from the tree refit on all training rows;
#                    tree_depth_<d>_nodes.csv (one row per node, for plotting) and
#                    importance_depth_<d>.csv (IAI variable importance) for each refit tree.
# Depth is NOT chosen here: the notebook picks it on out-of-fold AUC.
# For each fit, IAI tunes the complexity penalty `cp` on an internal split of that fit's training rows.

using CSV, DataFrames

dir       = ARGS[1]
max_depth = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
minbucket = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20

train = CSV.read(joinpath(dir, "train.csv"), DataFrame)
test  = CSV.read(joinpath(dir, "test.csv"), DataFrame)
features(d) = select(d, Not(intersect(names(d), ["y", "fold"])))

function fit_tree(X, y, depth)
    grid = IAI.GridSearch(
        IAI.OptimalTreeClassifier(random_seed=1, minbucket=minbucket, criterion=:gini),
        max_depth=depth,
    )
    IAI.fit!(grid, X, y, validation_criterion=:auc)
    IAI.get_learner(grid)
end
prob1(lnr, d) = IAI.predict_proba(lnr, features(d))[!, Symbol(1)]   # probability of y = 1

# One row per node, so Python can draw the tree like sklearn's plot_tree.
# Points with feature < threshold go to `lower` (left), the others to `upper` (right).
function tree_nodes(lnr)
    rows = map(1:IAI.get_num_nodes(lnr)) do t
        leaf = IAI.is_leaf(lnr, t)
        (node = t, depth = IAI.get_depth(lnr, t), is_leaf = leaf,
         feature   = leaf ? "" : string(IAI.get_split_feature(lnr, t)),
         threshold = leaf ? NaN : IAI.get_split_threshold(lnr, t),
         lower     = leaf ? 0 : IAI.get_lower_child(lnr, t),
         upper     = leaf ? 0 : IAI.get_upper_child(lnr, t),
         n_train   = IAI.get_num_samples(lnr, t),
         p1        = get(IAI.get_classification_proba(lnr, t, check_leaf=false), 1, 0.0))
    end
    DataFrame(rows)
end

folds = sort(unique(train.fold))
oof, p_train, p_test = DataFrame(), DataFrame(), DataFrame()
for depth in 1:max_depth
    t = @elapsed begin
        p = zeros(nrow(train))
        for k in folds
            fit_rows, out_rows = train.fold .!= k, train.fold .== k
            lnr = fit_tree(features(train[fit_rows, :]), train.y[fit_rows], depth)
            p[out_rows] = prob1(lnr, train[out_rows, :])
        end
        oof[!, "depth_$depth"] = p
        lnr = fit_tree(features(train), train.y, depth)   # refit on all training rows
        p_train[!, "depth_$depth"] = prob1(lnr, train)   # in-sample: used only to place the cutoff on this tree's leaves
        p_test[!, "depth_$depth"] = prob1(lnr, test)
    end
    CSV.write(joinpath(dir, "tree_depth_$(depth)_nodes.csv"), tree_nodes(lnr))
    CSV.write(joinpath(dir, "importance_depth_$(depth).csv"), IAI.variable_importance(lnr))
    println("depth $depth: $(IAI.get_num_nodes(lnr)) nodes in the refit tree, $(round(t, digits=1)) s for CV + refit")
end

CSV.write(joinpath(dir, "oof.csv"), oof)
CSV.write(joinpath(dir, "p_train.csv"), p_train)
CSV.write(joinpath(dir, "p_test.csv"), p_test)
